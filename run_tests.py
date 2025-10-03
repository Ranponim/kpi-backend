#!/usr/bin/env python3
"""
PEG 비교분석 테스트 실행 스크립트

이 스크립트는 PEG 비교분석 기능의 테스트를 실행하고 결과를 보고합니다.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def run_command(command, description):
    """명령어 실행 및 결과 반환"""
    print(f"\n{'='*60}")
    print(f"실행 중: {description}")
    print(f"명령어: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"실행 시간: {duration:.2f}초")
        print(f"반환 코드: {result.returncode}")
        
        if result.stdout:
            print("\n표준 출력:")
            print(result.stdout)
        
        if result.stderr:
            print("\n표준 오류:")
            print(result.stderr)
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except Exception as e:
        print(f"명령어 실행 중 오류 발생: {e}")
        return False, "", str(e)


def run_unit_tests():
    """단위 테스트 실행"""
    command = "python -m pytest app/tests/test_*.py -m 'not slow' -v"
    return run_command(command, "단위 테스트")


def run_integration_tests():
    """통합 테스트 실행"""
    command = "python -m pytest app/tests/integration/ -v"
    return run_command(command, "통합 테스트")


def run_performance_tests():
    """성능 테스트 실행"""
    command = "python -m pytest app/tests/performance/ -v"
    return run_command(command, "성능 테스트")


def run_all_tests():
    """모든 테스트 실행"""
    command = "python -m pytest app/tests/ -v"
    return run_command(command, "전체 테스트")


def run_specific_test(test_path):
    """특정 테스트 실행"""
    command = f"python -m pytest {test_path} -v"
    return run_command(command, f"특정 테스트: {test_path}")


def run_tests_with_coverage():
    """커버리지와 함께 테스트 실행"""
    command = "python -m pytest app/tests/ --cov=app --cov-report=html --cov-report=term-missing -v"
    return run_command(command, "커버리지 테스트")


def run_tests_parallel():
    """병렬 테스트 실행"""
    command = "python -m pytest app/tests/ -n auto -v"
    return run_command(command, "병렬 테스트")


def check_dependencies():
    """의존성 확인"""
    print("\n의존성 확인 중...")
    
    required_packages = [
        "pytest",
        "pytest-asyncio",
        "pytest-mock",
        "fastapi",
        "pydantic",
        "aiohttp",
        "cryptography"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (누락)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n누락된 패키지: {', '.join(missing_packages)}")
        print("다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n모든 의존성이 설치되어 있습니다.")
    return True


def generate_test_report():
    """테스트 보고서 생성"""
    print("\n테스트 보고서 생성 중...")
    
    # HTML 보고서 생성
    command = "python -m pytest app/tests/ --html=test_report.html --self-contained-html -v"
    success, stdout, stderr = run_command(command, "HTML 보고서 생성")
    
    if success:
        print("HTML 보고서가 test_report.html에 생성되었습니다.")
    
    # JUnit XML 보고서 생성
    command = "python -m pytest app/tests/ --junitxml=test_results.xml -v"
    success, stdout, stderr = run_command(command, "JUnit XML 보고서 생성")
    
    if success:
        print("JUnit XML 보고서가 test_results.xml에 생성되었습니다.")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="PEG 비교분석 테스트 실행 스크립트")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "performance", "all", "coverage", "parallel"],
        default="unit",
        help="실행할 테스트 타입"
    )
    parser.add_argument(
        "--test",
        type=str,
        help="실행할 특정 테스트 파일 경로"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="테스트 보고서 생성"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="의존성 확인"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 출력"
    )
    
    args = parser.parse_args()
    
    print("PEG 비교분석 테스트 실행 스크립트")
    print("=" * 50)
    
    # 의존성 확인
    if args.check_deps:
        if not check_dependencies():
            sys.exit(1)
        return
    
    # 특정 테스트 실행
    if args.test:
        success, stdout, stderr = run_specific_test(args.test)
        if not success:
            sys.exit(1)
        return
    
    # 테스트 타입에 따른 실행
    success = False
    
    if args.type == "unit":
        success, stdout, stderr = run_unit_tests()
    elif args.type == "integration":
        success, stdout, stderr = run_integration_tests()
    elif args.type == "performance":
        success, stdout, stderr = run_performance_tests()
    elif args.type == "all":
        success, stdout, stderr = run_all_tests()
    elif args.type == "coverage":
        success, stdout, stderr = run_tests_with_coverage()
    elif args.type == "parallel":
        success, stdout, stderr = run_tests_parallel()
    
    # 테스트 보고서 생성
    if args.report:
        generate_test_report()
    
    # 결과 출력
    print(f"\n{'='*60}")
    if success:
        print("✓ 모든 테스트가 성공적으로 완료되었습니다.")
    else:
        print("✗ 일부 테스트가 실패했습니다.")
        sys.exit(1)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()







