#!/usr/bin/env python3
"""
성능 테스트 실행 스크립트

Locust를 사용하여 마할라노비스 분석 API의 성능을 테스트합니다.
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# 프로젝트 루트 디렉토리 추가
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    TEST_SCENARIOS, PERFORMANCE_THRESHOLDS, REPORT_CONFIG,
    get_scenario_config, get_performance_thresholds
)


class PerformanceTestRunner:
    """성능 테스트 실행기"""

    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.locust_file = Path(__file__).parent / "locustfile.py"
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

    def check_backend_health(self) -> bool:
        """백엔드 서버 상태 확인"""
        import requests

        try:
            response = requests.get(f"{self.backend_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
        except Exception as e:
            print(f"❌ 백엔드 서버 상태 확인 실패: {e}")
            return False

        return False

    def run_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """특정 시나리오 실행"""
        print(f"🚀 {scenario_name} 시나리오 테스트 시작...")

        if not self.check_backend_health():
            raise RuntimeError("백엔드 서버가 정상 동작하지 않습니다.")

        scenario = get_scenario_config(scenario_name)
        user_class = scenario["user_class"]
        users = scenario["users"]
        spawn_rate = scenario["spawn_rate"]
        duration = scenario["duration"]

        # Locust 명령어 구성
        cmd = [
            "locust",
            "-f", str(self.locust_file),
            "--host", self.backend_url,
            "--users", str(users),
            "--spawn-rate", str(spawn_rate),
            "--run-time", f"{duration}s",
            "--headless",  # GUI 없이 실행
            "--only-summary",  # 요약만 출력
            "--json"  # JSON 출력
        ]

        # 사용자 클래스 지정 (기본값 외의 경우)
        if user_class != "MahalanobisAnalysisUser":
            cmd.extend(["--class-picker", user_class])

        print(f"실행 명령어: {' '.join(cmd)}")

        try:
            # Locust 실행
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )

            if result.returncode == 0:
                print("✅ 테스트 완료")
                return self.parse_locust_output(result.stdout, result.stderr)
            else:
                print(f"❌ 테스트 실패 (코드: {result.returncode})")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return {"error": f"테스트 실패: {result.stderr}"}

        except Exception as e:
            print(f"❌ 테스트 실행 중 오류: {e}")
            return {"error": str(e)}

    def parse_locust_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Locust 출력 파싱"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "errors": [],
            "raw_output": stdout
        }

        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        lines = stdout.split('\n')

        for line in lines:
            if "Requests/sec" in line and "Failures/sec" in line:
                # 요약 라인 파싱
                parts = line.split()
                if len(parts) >= 8:
                    result["summary"] = {
                        "requests_per_second": float(parts[1]),
                        "failures_per_second": float(parts[3]),
                        "response_time_50p": float(parts[5]),
                        "response_time_95p": float(parts[7])
                    }

        return result

    def validate_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """성능 기준치 검증"""
        thresholds = get_performance_thresholds()
        validation_results = {
            "passed": True,
            "violations": [],
            "scores": {}
        }

        if "summary" not in results:
            validation_results["passed"] = False
            validation_results["violations"].append("테스트 결과 파싱 실패")
            return validation_results

        summary = results["summary"]

        # 응답 시간 검증
        if summary.get("response_time_95p", 0) > thresholds["response_time_95p"]:
            validation_results["passed"] = False
            validation_results["violations"].append(
                f"95% 응답 시간 초과: {summary['response_time_95p']}ms > {thresholds['response_time_95p']}ms"
            )

        # 에러율 검증
        total_requests = summary.get("requests_per_second", 0) * 60  # 1분 기준 추정
        if total_requests > 0:
            error_rate = summary.get("failures_per_second", 0) / summary.get("requests_per_second", 1)
            if error_rate > thresholds["error_rate"]:
                validation_results["passed"] = False
                validation_results["violations"].append(
                    f"에러율 초과: {error_rate:.1%} > {thresholds['error_rate']:.1%}"
                )

        # 성능 점수 계산
        validation_results["scores"] = {
            "response_time_score": min(100, 5000 / max(summary.get("response_time_95p", 1), 1)),
            "throughput_score": min(100, summary.get("requests_per_second", 0) * 2),
            "reliability_score": max(0, 100 - (error_rate * 100)) if 'error_rate' in locals() else 100
        }

        return validation_results

    def save_results(self, scenario_name: str, results: Dict[str, Any], validation: Dict[str, Any]):
        """테스트 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scenario_name}_{timestamp}.json"
        filepath = self.results_dir / filename

        report_data = {
            "scenario": scenario_name,
            "timestamp": datetime.now().isoformat(),
            "backend_url": self.backend_url,
            "results": results,
            "validation": validation,
            "performance_thresholds": get_performance_thresholds()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"📊 테스트 결과 저장: {filepath}")
        return filepath

    def generate_report(self, results_file: Path) -> str:
        """HTML 보고서 생성"""
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>성능 테스트 보고서 - {data['scenario']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric {{ background: white; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric.pass {{ border-color: #4CAF50; }}
                .metric.fail {{ border-color: #f44336; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .violations {{ background: #ffebee; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>성능 테스트 보고서</h1>
                <p><strong>시나리오:</strong> {data['scenario']}</p>
                <p><strong>실행 시간:</strong> {data['timestamp']}</p>
                <p><strong>백엔드 URL:</strong> {data['backend_url']}</p>
            </div>

            <h2>성능 메트릭</h2>
            <div class="metrics">
        """

        summary = data.get('results', {}).get('summary', {})

        metrics = [
            ("응답 시간 (95p)", f"{summary.get('response_time_95p', 0)}ms", "response_time_95p"),
            ("초당 요청 수", f"{summary.get('requests_per_second', 0):.1f}", "requests_per_second"),
            ("실패율", f"{summary.get('failures_per_second', 0):.3f}/s", "failures_per_second")
        ]

        thresholds = data.get('performance_thresholds', {})

        for name, value, key in metrics:
            status_class = "pass"
            if key == "response_time_95p" and summary.get(key, 0) > thresholds.get("response_time_95p", 2000):
                status_class = "fail"
            elif key == "failures_per_second" and summary.get(key, 0) > 0:
                status_class = "fail"

            html_content += f"""
                <div class="metric {status_class}">
                    <h3>{name}</h3>
                    <div class="score">{value}</div>
                </div>
            """

        html_content += """
            </div>

            <h2>성능 점수</h2>
            <div class="metrics">
        """

        scores = data.get('validation', {}).get('scores', {})
        for score_name, score_value in scores.items():
            html_content += f"""
                <div class="metric pass">
                    <h3>{score_name.replace('_', ' ').title()}</h3>
                    <div class="score">{score_value:.1f}</div>
                </div>
            """

        html_content += """
            </div>
        """

        violations = data.get('validation', {}).get('violations', [])
        if violations:
            html_content += """
                <div class="violations">
                    <h2>🚨 성능 기준 위반 사항</h2>
                    <ul>
            """
            for violation in violations:
                html_content += f"<li>{violation}</li>"
            html_content += "</ul></div>"

        html_content += """
        </body>
        </html>
        """

        report_file = results_file.with_suffix('.html')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📈 HTML 보고서 생성: {report_file}")
        return str(report_file)


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="마할라노비스 분석 API 성능 테스트")
    parser.add_argument(
        "--scenario",
        choices=list(TEST_SCENARIOS.keys()),
        default="normal_load",
        help="실행할 테스트 시나리오"
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="백엔드 서버 URL"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="HTML 보고서 생성"
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="모든 시나리오 실행"
    )

    args = parser.parse_args()

    runner = PerformanceTestRunner(args.backend_url)

    if args.all_scenarios:
        print("🔄 모든 시나리오 테스트 실행...")
        all_results = {}

        for scenario_name in TEST_SCENARIOS.keys():
            try:
                results = runner.run_scenario(scenario_name)
                validation = runner.validate_performance(results)
                runner.save_results(scenario_name, results, validation)

                all_results[scenario_name] = {
                    "results": results,
                    "validation": validation
                }

                status = "✅ 통과" if validation["passed"] else "❌ 실패"
                print(f"{scenario_name}: {status}")

            except Exception as e:
                print(f"❌ {scenario_name} 시나리오 실행 실패: {e}")
                all_results[scenario_name] = {"error": str(e)}

        # 전체 요약 출력
        print("\n📊 전체 테스트 요약:")
        for scenario, result in all_results.items():
            if "error" in result:
                print(f"  {scenario}: ❌ 오류 - {result['error']}")
            else:
                passed = result['validation']['passed']
                status = "✅ 통과" if passed else "❌ 실패"
                score = result['validation']['scores'].get('response_time_score', 0)
                print(".1f"
        if args.generate_report:
            print("\n📈 개별 HTML 보고서들이 생성되었습니다.")

    else:
        # 단일 시나리오 실행
        try:
            results = runner.run_scenario(args.scenario)
            validation = runner.validate_performance(results)
            results_file = runner.save_results(args.scenario, results, validation)

            print(f"\n📊 테스트 결과:")
            print(f"시나리오: {args.scenario}")
            print(f"상태: {'✅ 통과' if validation['passed'] else '❌ 실패'}")

            if validation['violations']:
                print("🚨 위반 사항:")
                for violation in validation['violations']:
                    print(f"  - {violation}")

            if args.generate_report:
                report_file = runner.generate_report(Path(results_file))
                print(f"📈 HTML 보고서: {report_file}")

        except Exception as e:
            print(f"❌ 테스트 실행 실패: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()


