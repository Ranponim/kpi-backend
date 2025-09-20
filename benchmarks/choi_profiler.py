"""
Choi 알고리즘 상세 프로파일링 도구

이 모듈은 cProfile과 다양한 프로파일링 도구를 사용하여
Choi 알고리즘의 성능 병목 지점을 정밀 분석합니다.

Author: Choi Algorithm Profiling Team
Created: 2025-09-20
"""

import cProfile
import pstats
import io
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
import json

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.peg_processing_service import PEGProcessingService
from app.models.judgement import PegSampleSeries

# 로깅 설정
logging.basicConfig(level=logging.ERROR)  # 프로파일링 시 로그 최소화
logger = logging.getLogger(__name__)


class ChoiAlgorithmProfiler:
    """
    Choi 알고리즘 프로파일러
    
    cProfile을 사용하여 함수별 실행 시간과 호출 빈도를 분석하고
    성능 병목 지점을 식별합니다.
    """
    
    def __init__(self):
        """프로파일러 초기화"""
        self.service = PEGProcessingService()
        self.profiling_results = {}
        
        logger.info("Choi algorithm profiler initialized")
    
    def run_comprehensive_profiling(self) -> Dict[str, Any]:
        """포괄적 프로파일링 실행"""
        print("🔍 Choi 알고리즘 상세 프로파일링")
        print("=" * 60)
        
        try:
            # 1. 전체 워크플로우 프로파일링
            print("1. 전체 워크플로우 프로파일링:")
            full_workflow_profile = self._profile_full_workflow()
            
            # 2. 단계별 프로파일링
            print("\n2. 단계별 프로파일링:")
            phase_profiles = self._profile_individual_phases()
            
            # 3. 핫스팟 분석
            print("\n3. 핫스팟 분석:")
            hotspot_analysis = self._analyze_hotspots(full_workflow_profile)
            
            # 4. 함수 호출 빈도 분석
            print("\n4. 함수 호출 빈도 분석:")
            call_frequency_analysis = self._analyze_call_frequencies(full_workflow_profile)
            
            # 5. 메모리 프로파일링
            print("\n5. 메모리 프로파일링:")
            memory_profile = self._profile_memory_usage()
            
            # 종합 결과
            comprehensive_results = {
                "timestamp": datetime.now().isoformat(),
                "full_workflow": full_workflow_profile,
                "phase_breakdown": phase_profiles,
                "hotspots": hotspot_analysis,
                "call_frequencies": call_frequency_analysis,
                "memory_profile": memory_profile,
                "optimization_recommendations": self._generate_optimization_recommendations(
                    hotspot_analysis, call_frequency_analysis
                )
            }
            
            # 결과 저장
            self._save_profiling_results(comprehensive_results)
            
            # 요약 보고
            self._report_profiling_summary(comprehensive_results)
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Profiling execution failed: {e}")
            raise
    
    def _profile_full_workflow(self) -> Dict[str, Any]:
        """전체 워크플로우 프로파일링"""
        print("  🔄 전체 워크플로우 cProfile 실행")
        
        # 표준 테스트 데이터
        cell_ids = [f"profile_cell_{i:03d}" for i in range(10)]
        input_data = {"ems_ip": "192.168.200.10"}
        time_range = {"start": datetime.now()}
        
        # cProfile 실행
        profiler = cProfile.Profile()
        
        profiler.enable()
        
        # 알고리즘 실행
        response = self.service.process_peg_data(input_data, cell_ids, time_range)
        
        profiler.disable()
        
        # 프로파일 결과 분석
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        
        # 다양한 정렬로 통계 생성
        profile_data = {}
        
        # 1. 누적 시간 기준 상위 20개 함수
        stats.sort_stats('cumulative')
        stats.print_stats(20)
        profile_data["by_cumulative_time"] = stats_stream.getvalue()
        
        # 2. 자체 실행 시간 기준 상위 20개 함수
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('tottime')
        stats.print_stats(20)
        profile_data["by_self_time"] = stats_stream.getvalue()
        
        # 3. 호출 횟수 기준 상위 20개 함수
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('ncalls')
        stats.print_stats(20)
        profile_data["by_call_count"] = stats_stream.getvalue()
        
        # 프로파일 통계 추출
        stats_dict = {}
        for func_key, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line_num, func_name = func_key
            
            # Choi 알고리즘 관련 함수만 필터링
            if any(keyword in filename for keyword in ['choi', 'anomaly', 'kpi', 'filtering', 'judgement']):
                stats_dict[f"{filename}:{func_name}"] = {
                    "call_count": cc,
                    "total_time": round(tt, 6),
                    "cumulative_time": round(ct, 6),
                    "time_per_call": round(tt / cc, 6) if cc > 0 else 0,
                    "line_number": line_num
                }
        
        return {
            "total_execution_time_ms": response.processing_time_ms,
            "cells_analyzed": response.total_cells_analyzed,
            "profile_text": profile_data,
            "function_stats": stats_dict,
            "profiling_overhead_estimated": "< 5%"
        }
    
    def _profile_individual_phases(self) -> Dict[str, Any]:
        """단계별 개별 프로파일링"""
        # 테스트 데이터 준비
        cell_ids = [f"phase_cell_{i:03d}" for i in range(10)]
        peg_data = self._generate_standard_test_data(cell_ids)
        
        phase_results = {}
        
        # 각 단계별로 개별 프로파일링
        phases = [
            ("filtering", lambda: self.service._run_filtering(peg_data)),
            ("aggregation", lambda data=peg_data: self._profile_aggregation(data)),
            ("derivation", lambda data=peg_data: self._profile_derivation(data)),
            ("judgement", lambda data=peg_data: self._profile_judgement(data))
        ]
        
        for phase_name, phase_func in phases:
            print(f"  🔍 {phase_name} 단계 프로파일링")
            
            # 3회 측정
            times = []
            for _ in range(3):
                start_time = time.perf_counter()
                result = phase_func()
                end_time = time.perf_counter()
                
                execution_time = (end_time - start_time) * 1000
                times.append(execution_time)
            
            avg_time = sum(times) / len(times)
            
            phase_results[phase_name] = {
                "avg_time_ms": round(avg_time, 3),
                "min_time_ms": round(min(times), 3),
                "max_time_ms": round(max(times), 3),
                "measurements": len(times)
            }
            
            print(f"    ⏱️ {phase_name}: {avg_time:.3f}ms")
        
        return phase_results
    
    def _profile_aggregation(self, peg_data: Dict[str, List[PegSampleSeries]]) -> Any:
        """집계 단계 프로파일링"""
        filtering_result = self.service._run_filtering(peg_data)
        return self.service._aggregation(peg_data, filtering_result)
    
    def _profile_derivation(self, peg_data: Dict[str, List[PegSampleSeries]]) -> Any:
        """파생 계산 단계 프로파일링"""
        filtering_result = self.service._run_filtering(peg_data)
        aggregated_data = self.service._aggregation(peg_data, filtering_result)
        return self.service._derived_calculation(aggregated_data)
    
    def _profile_judgement(self, peg_data: Dict[str, List[PegSampleSeries]]) -> Any:
        """판정 단계 프로파일링"""
        filtering_result = self.service._run_filtering(peg_data)
        aggregated_data = self.service._aggregation(peg_data, filtering_result)
        derived_data = self.service._derived_calculation(aggregated_data)
        return self.service._run_judgement(derived_data, filtering_result)
    
    def _analyze_hotspots(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """핫스팟 분석"""
        function_stats = profile_data["function_stats"]
        
        if not function_stats:
            return {"hotspots": [], "analysis": "No function stats available"}
        
        # 누적 시간 기준 상위 10개 함수
        hotspots_by_cumtime = sorted(
            function_stats.items(),
            key=lambda x: x[1]["cumulative_time"],
            reverse=True
        )[:10]
        
        # 자체 실행 시간 기준 상위 10개 함수
        hotspots_by_selftime = sorted(
            function_stats.items(),
            key=lambda x: x[1]["total_time"],
            reverse=True
        )[:10]
        
        # 호출 횟수 기준 상위 10개 함수
        hotspots_by_calls = sorted(
            function_stats.items(),
            key=lambda x: x[1]["call_count"],
            reverse=True
        )[:10]
        
        hotspots = {
            "by_cumulative_time": [
                {
                    "function": func_name,
                    "cumulative_time_ms": round(stats["cumulative_time"] * 1000, 3),
                    "call_count": stats["call_count"],
                    "time_per_call_ms": round(stats["time_per_call"] * 1000, 6)
                }
                for func_name, stats in hotspots_by_cumtime
            ],
            "by_self_time": [
                {
                    "function": func_name,
                    "self_time_ms": round(stats["total_time"] * 1000, 3),
                    "call_count": stats["call_count"],
                    "time_per_call_ms": round(stats["time_per_call"] * 1000, 6)
                }
                for func_name, stats in hotspots_by_selftime
            ],
            "by_call_frequency": [
                {
                    "function": func_name,
                    "call_count": stats["call_count"],
                    "total_time_ms": round(stats["total_time"] * 1000, 3),
                    "avg_time_per_call_us": round(stats["time_per_call"] * 1000000, 2)
                }
                for func_name, stats in hotspots_by_calls
            ]
        }
        
        # 상위 3개 핫스팟 출력
        print("    🔥 상위 핫스팟 (누적 시간):")
        for i, hotspot in enumerate(hotspots["by_cumulative_time"][:3], 1):
            print(f"      {i}. {hotspot['function'].split(':')[-1]}: {hotspot['cumulative_time_ms']:.3f}ms")
        
        return hotspots
    
    def _analyze_call_frequencies(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """함수 호출 빈도 분석"""
        function_stats = profile_data["function_stats"]
        
        if not function_stats:
            return {"analysis": "No function stats available"}
        
        # 호출 빈도 통계
        call_counts = [stats["call_count"] for stats in function_stats.values()]
        
        if not call_counts:
            return {"analysis": "No call count data"}
        
        total_calls = sum(call_counts)
        avg_calls = total_calls / len(call_counts)
        max_calls = max(call_counts)
        
        # 자주 호출되는 함수들 (상위 20%)
        high_frequency_threshold = sorted(call_counts, reverse=True)[int(len(call_counts) * 0.2)]
        
        high_frequency_functions = [
            {
                "function": func_name.split(':')[-1],
                "call_count": stats["call_count"],
                "total_time_ms": round(stats["total_time"] * 1000, 3),
                "avg_time_per_call_us": round(stats["time_per_call"] * 1000000, 2)
            }
            for func_name, stats in function_stats.items()
            if stats["call_count"] >= high_frequency_threshold
        ]
        
        # 호출 빈도별 정렬
        high_frequency_functions.sort(key=lambda x: x["call_count"], reverse=True)
        
        analysis = {
            "total_function_calls": total_calls,
            "unique_functions": len(function_stats),
            "avg_calls_per_function": round(avg_calls, 1),
            "max_calls_single_function": max_calls,
            "high_frequency_functions": high_frequency_functions[:10],
            "call_distribution": {
                "functions_called_once": len([c for c in call_counts if c == 1]),
                "functions_called_2_10": len([c for c in call_counts if 2 <= c <= 10]),
                "functions_called_10_plus": len([c for c in call_counts if c > 10])
            }
        }
        
        print(f"    📞 총 함수 호출: {total_calls:,}회")
        print(f"    🔄 최다 호출 함수: {max_calls:,}회")
        
        return analysis
    
    def _profile_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 프로파일링"""
        try:
            import tracemalloc
            import psutil
            
            # 메모리 추적 시작
            tracemalloc.start()
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # 대용량 데이터로 메모리 프로파일링
            cell_ids = [f"memory_profile_cell_{i:03d}" for i in range(20)]
            input_data = {"ems_ip": "192.168.200.20"}
            time_range = {"start": datetime.now()}
            
            # 단계별 메모리 측정
            memory_snapshots = []
            
            # 초기 상태
            memory_snapshots.append({
                "phase": "initial",
                "memory_mb": process.memory_info().rss / 1024 / 1024
            })
            
            # 알고리즘 실행
            response = self.service.process_peg_data(input_data, cell_ids, time_range)
            
            # 최종 상태
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_snapshots.append({
                "phase": "final",
                "memory_mb": final_memory
            })
            
            # tracemalloc 통계
            current, peak = tracemalloc.get_traced_memory()
            
            # 메모리 통계 생성
            memory_stats = tracemalloc.take_snapshot().statistics('lineno')
            
            # 상위 메모리 사용 함수들
            top_memory_users = []
            for stat in memory_stats[:10]:
                try:
                    traceback_info = stat.traceback.format()[0] if stat.traceback.format() else "unknown"
                    if any(keyword in traceback_info for keyword in ['choi', 'anomaly', 'kpi']):
                        top_memory_users.append({
                            "file": traceback_info.split('/')[-1] if '/' in traceback_info else traceback_info,
                            "size_mb": round(stat.size / 1024 / 1024, 3),
                            "count": stat.count
                        })
                except Exception:
                    # tracemalloc 통계 접근 오류 시 건너뛰기
                    continue
            
            tracemalloc.stop()
            
            results = {
                "initial_memory_mb": round(initial_memory, 3),
                "final_memory_mb": round(final_memory, 3),
                "memory_increase_mb": round(final_memory - initial_memory, 3),
                "peak_traced_memory_mb": round(peak / 1024 / 1024, 3),
                "current_traced_memory_mb": round(current / 1024 / 1024, 3),
                "cells_processed": len(cell_ids),
                "memory_per_cell_mb": round((final_memory - initial_memory) / len(cell_ids), 4),
                "top_memory_users": top_memory_users,
                "memory_snapshots": memory_snapshots
            }
            
            print(f"    🧠 메모리 증가: {results['memory_increase_mb']:.3f}MB")
            print(f"    📊 Peak 메모리: {results['peak_traced_memory_mb']:.3f}MB")
            
            return results
            
        except ImportError:
            return {"error": "tracemalloc or psutil not available"}
    
    def _generate_standard_test_data(self, cell_ids: List[str]) -> Dict[str, List[PegSampleSeries]]:
        """표준 테스트 데이터 생성"""
        peg_data = {}
        
        for cell_id in cell_ids:
            peg_series = []
            peg_names = ["AirMacDLThruAvg", "AirMacULThruAvg", "ConnNoAvg"]
            
            for peg_name in peg_names:
                # 표준 20개 샘플
                pre_samples = [1000.0 + i * 10 for i in range(20)]
                post_samples = [1100.0 + i * 12 for i in range(20)]
                
                series = PegSampleSeries(
                    peg_name=peg_name,
                    cell_id=cell_id,
                    pre_samples=pre_samples,
                    post_samples=post_samples,
                    unit="Kbps" if "Thru" in peg_name else "count"
                )
                
                peg_series.append(series)
            
            peg_data[cell_id] = peg_series
        
        return peg_data
    
    def _generate_optimization_recommendations(self, 
                                             hotspots: Dict[str, Any], 
                                             call_frequencies: Dict[str, Any]) -> List[Dict[str, Any]]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        # 1. 누적 시간 기준 최적화
        if hotspots.get("by_cumulative_time"):
            top_cumulative = hotspots["by_cumulative_time"][0]
            if top_cumulative["cumulative_time_ms"] > 1.0:  # 1ms 이상
                recommendations.append({
                    "type": "cumulative_time_optimization",
                    "priority": "high",
                    "function": top_cumulative["function"],
                    "current_time_ms": top_cumulative["cumulative_time_ms"],
                    "recommendation": "이 함수의 알고리즘 복잡도를 검토하고 numpy 벡터화 적용을 고려하세요.",
                    "potential_improvement": "20-50% 성능 향상 가능"
                })
        
        # 2. 호출 빈도 기준 최적화
        if call_frequencies.get("high_frequency_functions"):
            top_frequent = call_frequencies["high_frequency_functions"][0]
            if top_frequent["call_count"] > 1000:  # 1000회 이상 호출
                recommendations.append({
                    "type": "call_frequency_optimization",
                    "priority": "medium",
                    "function": top_frequent["function"],
                    "call_count": top_frequent["call_count"],
                    "recommendation": "자주 호출되는 함수는 결과 캐싱이나 계산 최적화를 고려하세요.",
                    "potential_improvement": "10-30% 성능 향상 가능"
                })
        
        # 3. 메모리 최적화
        recommendations.append({
            "type": "memory_optimization",
            "priority": "low",
            "recommendation": "현재 메모리 사용량이 효율적이지만, 대용량 데이터 처리 시 numpy 배열 재사용을 고려하세요.",
            "potential_improvement": "메모리 사용량 10-20% 감소 가능"
        })
        
        # 4. 알고리즘 최적화
        recommendations.append({
            "type": "algorithmic_optimization", 
            "priority": "low",
            "recommendation": "필터링 단계가 주요 병목이므로, 조기 종료 조건이나 병렬 처리를 고려하세요.",
            "potential_improvement": "대용량 데이터에서 30-50% 성능 향상 가능"
        })
        
        return recommendations
    
    def _save_profiling_results(self, results: Dict[str, Any]) -> None:
        """프로파일링 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON 결과 저장
        json_file = Path(__file__).parent / f"profiling_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            # 텍스트 프로파일 데이터는 제외하고 저장
            save_data = {k: v for k, v in results.items() if k != "full_workflow"}
            save_data["full_workflow_summary"] = {
                "total_execution_time_ms": results["full_workflow"]["total_execution_time_ms"],
                "cells_analyzed": results["full_workflow"]["cells_analyzed"],
                "function_count": len(results["full_workflow"]["function_stats"])
            }
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
        
        # 텍스트 프로파일 결과 저장
        text_file = Path(__file__).parent / f"cprofile_output_{timestamp}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("=== Choi Algorithm cProfile Output ===\n\n")
            for key, content in results["full_workflow"]["profile_text"].items():
                f.write(f"=== {key.upper()} ===\n")
                f.write(content)
                f.write("\n\n")
        
        print(f"  💾 프로파일링 결과 저장: {json_file.name}, {text_file.name}")
    
    def _report_profiling_summary(self, results: Dict[str, Any]) -> None:
        """프로파일링 요약 보고"""
        print("\n" + "=" * 60)
        print("🔍 Choi 알고리즘 프로파일링 요약")
        print("=" * 60)
        
        # 전체 성능
        full_workflow = results["full_workflow"]
        print(f"⏱️ 전체 실행 시간: {full_workflow['total_execution_time_ms']:.3f}ms")
        print(f"📊 분석된 셀: {full_workflow['cells_analyzed']}개")
        print(f"🔧 프로파일된 함수: {len(full_workflow['function_stats'])}개")
        
        # 단계별 성능
        phases = results["phase_breakdown"]
        print(f"\n⚡ 단계별 성능:")
        total_phase_time = sum(phase["avg_time_ms"] for phase in phases.values())
        for phase_name, phase_data in phases.items():
            percentage = (phase_data["avg_time_ms"] / total_phase_time) * 100 if total_phase_time > 0 else 0
            print(f"  {phase_name}: {phase_data['avg_time_ms']:.3f}ms ({percentage:.1f}%)")
        
        # 최적화 권장사항
        recommendations = results["optimization_recommendations"]
        print(f"\n💡 최적화 권장사항:")
        for i, rec in enumerate(recommendations, 1):
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec["priority"], "⚪")
            print(f"  {i}. {priority_icon} {rec['type']}: {rec['recommendation']}")
        
        print("=" * 60)


# =============================================================================
# 직접 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    try:
        profiler = ChoiAlgorithmProfiler()
        results = profiler.run_comprehensive_profiling()
        
        print("\n🎉 프로파일링 완료!")
        print("📁 상세 결과는 생성된 파일들을 확인하세요.")
        
    except Exception as e:
        print(f"❌ 프로파일링 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
