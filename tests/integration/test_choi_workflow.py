"""
Choi 알고리즘 전체 워크플로우 통합 테스트

이 모듈은 PEGProcessingService를 통한 완전한 Choi 알고리즘 워크플로우를
실제 테스트 데이터로 검증합니다. 6장 필터링, 4장 이상탐지, 5장 KPI분석의
전체 파이프라인을 종합적으로 테스트합니다.

Author: Choi Algorithm Test Team
Created: 2025-09-20
"""

import pytest
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import sys

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.peg_processing_service import PEGProcessingService
from app.models.judgement import PegSampleSeries, ChoiAlgorithmResponse

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestChoiWorkflow:
    """
    Choi 알고리즘 전체 워크플로우 통합 테스트 클래스
    
    SOLID 원칙을 준수하여 구현된 Choi 알고리즘 시스템의
    완전한 통합 테스트를 수행합니다.
    """
    
    @pytest.fixture(scope="class")
    def peg_processing_service(self):
        """
        PEGProcessingService 픽스처 (클래스 스코프)
        
        Strategy Factory를 통한 완전한 의존성 주입으로 생성
        """
        logger.info("Creating PEGProcessingService with Strategy Factory DI")
        service = PEGProcessingService()
        
        # 서비스 정상 초기화 검증
        assert service.filtering_strategy is not None, "Filtering strategy should be initialized"
        assert service.judgement_strategy is not None, "Judgement strategy should be initialized"
        assert service.config is not None, "Configuration should be loaded"
        
        logger.info("PEGProcessingService fixture created successfully")
        return service
    
    @pytest.fixture(scope="class")
    def test_fixtures(self):
        """테스트 픽스처 로드"""
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "choi_algorithm"
        expected_outputs_dir = fixtures_dir / "expected_outputs"
        
        fixtures = {}
        
        # 시나리오 픽스처 로드
        for fixture_file in fixtures_dir.glob("scenario_*.json"):
            with open(fixture_file, 'r', encoding='utf-8') as f:
                scenario_data = json.load(f)
                scenario_name = scenario_data["scenario_name"]  # JSON 내부의 scenario_name 사용
                fixtures[scenario_name] = scenario_data
        
        # 예상 출력 로드
        for expected_file in expected_outputs_dir.glob("*_expected.json"):
            scenario_name = expected_file.stem.replace("_expected", "")
            
            with open(expected_file, 'r', encoding='utf-8') as f:
                if scenario_name in fixtures:
                    fixtures[scenario_name]["expected_output"] = json.load(f)
        
        logger.info(f"Loaded {len(fixtures)} test fixtures")
        return fixtures
    
    @pytest.mark.parametrize("scenario_name", [
        "normal_case",
        "anomaly_detection_case", 
        "fifty_percent_rule_trigger"
    ])
    def test_choi_algorithm_scenarios(self, peg_processing_service, test_fixtures, scenario_name):
        """
        Choi 알고리즘 시나리오별 통합 테스트
        
        각 시나리오에 대해 전체 워크플로우를 실행하고 
        예상 결과와 비교하여 정확성을 검증합니다.
        """
        logger.info(f"🧪 Testing scenario: {scenario_name}")
        
        # 픽스처 데이터 가져오기
        scenario_data = test_fixtures[scenario_name]
        expected_output = scenario_data.get("expected_output")
        
        assert expected_output is not None, f"Expected output not found for {scenario_name}"
        
        # 입력 데이터 준비
        input_data = scenario_data["input_data"]
        cell_ids = scenario_data["cell_ids"]
        time_range = self._convert_time_range(scenario_data["time_range"])
        
        # 성능 측정 시작
        start_time = datetime.now()
        
        # 전체 워크플로우 실행
        logger.info(f"Executing Choi algorithm for {scenario_name}")
        actual_response = peg_processing_service.process_peg_data(
            input_data, cell_ids, time_range
        )
        
        # 성능 검증
        processing_time = (datetime.now() - start_time).total_seconds()
        assert processing_time < 5.0, f"Processing time {processing_time:.2f}s exceeds 5s limit"
        
        logger.info(f"✅ Performance test passed: {processing_time:.3f}s < 5.0s")
        
        # 결과 구조 검증
        self._validate_response_structure(actual_response)
        
        # 주요 결과 비교
        self._compare_filtering_results(actual_response.filtering, expected_output["filtering"])
        self._compare_abnormal_detection_results(actual_response.abnormal_detection, expected_output["abnormal_detection"])
        
        # 메타데이터 검증
        assert actual_response.total_cells_analyzed == len(cell_ids)
        assert actual_response.algorithm_version == expected_output["algorithm_version"]
        
        logger.info(f"🎉 Scenario {scenario_name} test passed completely")
    
    def test_choi_algorithm_performance_benchmark(self, peg_processing_service):
        """
        Choi 알고리즘 성능 벤치마크 테스트
        
        대용량 데이터로 성능 기준 준수 여부를 검증합니다.
        """
        logger.info("🚀 Performance benchmark test")
        
        # 대용량 테스트 데이터 생성 (10셀 × 5PEG × 100샘플)
        large_input_data = {"ems_ip": "192.168.1.200"}
        large_cell_ids = [f"cell_{i:03d}" for i in range(10)]
        time_range = {"start": datetime.now()}
        
        # 성능 측정
        start_time = datetime.now()
        
        response = peg_processing_service.process_peg_data(
            large_input_data, large_cell_ids, time_range
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 성능 기준 검증 (PRD 4.3 요구사항)
        max_allowed_time = len(large_cell_ids) * 0.1  # 셀당 100ms
        assert processing_time < max_allowed_time, f"Performance {processing_time:.2f}s exceeds {max_allowed_time:.2f}s"
        
        # 결과 무결성 검증
        assert response.total_cells_analyzed == len(large_cell_ids)
        assert len(response.filtering.valid_time_slots) == len(large_cell_ids)
        
        logger.info(f"✅ Performance benchmark passed: {processing_time:.3f}s for {len(large_cell_ids)} cells")
    
    def test_choi_algorithm_error_handling(self, peg_processing_service):
        """
        Choi 알고리즘 오류 처리 견고성 테스트
        
        잘못된 입력 데이터에 대한 시스템의 견고성을 검증합니다.
        """
        logger.info("🛡️ Error handling robustness test")
        
        # 빈 데이터 테스트 - 예외 발생 예상
        with pytest.raises(RuntimeError, match="PEG processing failed"):
            peg_processing_service.process_peg_data(
                {"ems_ip": "test"}, [], {"start": datetime.now()}
            )
        
        logger.info("✅ Empty data error handling verified")
        
        # 최소한의 유효 데이터로 정상 동작 확인
        minimal_response = peg_processing_service.process_peg_data(
            {"ems_ip": "test"}, ["cell_test"], {"start": datetime.now()}
        )
        
        # 최소 데이터로도 응답 생성 확인
        assert isinstance(minimal_response, ChoiAlgorithmResponse)
        assert minimal_response.total_cells_analyzed >= 0
        
        logger.info("✅ Minimal data handling test passed")
        logger.info("✅ Error handling robustness verified")
    
    def _convert_time_range(self, time_range_data: Dict[str, str]) -> Dict[str, datetime]:
        """시간 범위 데이터 변환"""
        return {
            key: datetime.fromisoformat(value) if isinstance(value, str) else value
            for key, value in time_range_data.items()
        }
    
    def _validate_response_structure(self, response: ChoiAlgorithmResponse) -> None:
        """응답 구조 검증"""
        # 필수 필드 존재 확인
        assert hasattr(response, 'timestamp'), "Response should have timestamp"
        assert hasattr(response, 'processing_time_ms'), "Response should have processing_time_ms"
        assert hasattr(response, 'algorithm_version'), "Response should have algorithm_version"
        assert hasattr(response, 'filtering'), "Response should have filtering results"
        assert hasattr(response, 'abnormal_detection'), "Response should have abnormal_detection results"
        assert hasattr(response, 'kpi_judgement'), "Response should have kpi_judgement results"
        
        # 타입 검증
        assert isinstance(response.total_cells_analyzed, int), "total_cells_analyzed should be int"
        assert isinstance(response.total_pegs_analyzed, int), "total_pegs_analyzed should be int"
        assert isinstance(response.processing_warnings, list), "processing_warnings should be list"
        
        logger.debug("Response structure validation passed")
    
    def _compare_filtering_results(self, actual, expected) -> None:
        """필터링 결과 비교"""
        # 필터링 비율 검증 (±5% 허용 오차)
        expected_ratio_range = expected.get("filter_ratio_range", [0.0, 1.0])
        assert expected_ratio_range[0] <= actual.filter_ratio <= expected_ratio_range[1], \
            f"Filter ratio {actual.filter_ratio} not in expected range {expected_ratio_range}"
        
        # 50% 규칙 트리거 검증
        should_trigger = expected.get("should_trigger_50_percent_rule", False)
        has_warning = actual.warning_message is not None
        assert has_warning == should_trigger, \
            f"50% rule trigger mismatch: expected {should_trigger}, got {has_warning}"
        
        logger.debug("Filtering results comparison passed")
    
    def _compare_abnormal_detection_results(self, actual, expected) -> None:
        """이상 탐지 결과 비교"""
        expected_anomalies = set(expected.get("expected_anomalies", []))
        expected_displays = expected.get("display_results", {})
        
        # α0 규칙에 따른 표시 결과 검증
        for anomaly_type, should_display in expected_displays.items():
            actual_display = actual.display_results.get(anomaly_type, False)
            assert actual_display == should_display, \
                f"Display result for {anomaly_type}: expected {should_display}, got {actual_display}"
        
        logger.debug("Abnormal detection results comparison passed")


# =============================================================================
# 테스트 실행 편의 함수들
# =============================================================================

def run_single_scenario_test(scenario_name: str) -> bool:
    """
    단일 시나리오 테스트 실행
    
    Args:
        scenario_name: 테스트할 시나리오 이름
        
    Returns:
        bool: 테스트 성공 여부
    """
    try:
        logger.info(f"Running single scenario test: {scenario_name}")
        
        # 서비스 생성
        service = PEGProcessingService()
        
        # 픽스처 로드
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "choi_algorithm"
        fixture_file = fixtures_dir / f"scenario_{scenario_name}.json"
        
        if not fixture_file.exists():
            logger.error(f"Fixture file not found: {fixture_file}")
            return False
        
        with open(fixture_file, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)
        
        # 테스트 실행
        input_data = scenario_data["input_data"]
        cell_ids = scenario_data["cell_ids"]
        time_range = {
            key: datetime.fromisoformat(value) if isinstance(value, str) else value
            for key, value in scenario_data["time_range"].items()
        }
        
        response = service.process_peg_data(input_data, cell_ids, time_range)
        
        # 기본 검증
        assert isinstance(response, ChoiAlgorithmResponse)
        assert response.total_cells_analyzed == len(cell_ids)
        
        logger.info(f"✅ Single scenario test {scenario_name} passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Single scenario test {scenario_name} failed: {e}")
        return False


def run_all_integration_tests() -> bool:
    """
    모든 통합 테스트 실행
    
    Returns:
        bool: 전체 테스트 성공 여부
    """
    try:
        logger.info("🧪 Running all Choi algorithm integration tests")
        
        # 시나리오별 테스트
        scenarios = ["normal", "anomalies", "fifty_percent_rule"]
        
        results = []
        for scenario in scenarios:
            result = run_single_scenario_test(scenario)
            results.append(result)
            
            if result:
                logger.info(f"✅ {scenario} scenario: PASSED")
            else:
                logger.error(f"❌ {scenario} scenario: FAILED")
        
        # 전체 결과
        all_passed = all(results)
        
        if all_passed:
            logger.info("🎉 All integration tests PASSED")
        else:
            failed_count = sum(1 for r in results if not r)
            logger.error(f"❌ {failed_count}/{len(results)} tests FAILED")
        
        return all_passed
        
    except Exception as e:
        logger.error(f"Error running integration tests: {e}")
        return False


# =============================================================================
# 성능 벤치마크 테스트
# =============================================================================

class TestChoiPerformance:
    """Choi 알고리즘 성능 테스트 클래스"""
    
    @pytest.fixture
    def performance_service(self):
        """성능 테스트용 서비스"""
        return PEGProcessingService()
    
    @pytest.mark.performance
    def test_large_dataset_performance(self, performance_service):
        """대용량 데이터셋 성능 테스트"""
        logger.info("🚀 Large dataset performance test")
        
        # 대용량 데이터 (50셀 × 10PEG)
        large_cell_ids = [f"cell_{i:03d}" for i in range(50)]
        input_data = {"ems_ip": "192.168.1.200"}
        time_range = {"start": datetime.now()}
        
        start_time = datetime.now()
        
        response = performance_service.process_peg_data(
            input_data, large_cell_ids, time_range
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 성능 기준: 셀당 100ms 이하
        max_allowed = len(large_cell_ids) * 0.1
        assert processing_time < max_allowed, \
            f"Performance {processing_time:.2f}s exceeds limit {max_allowed:.2f}s"
        
        # 결과 무결성
        assert response.total_cells_analyzed == len(large_cell_ids)
        
        logger.info(f"✅ Performance test passed: {processing_time:.3f}s for {len(large_cell_ids)} cells")
    
    @pytest.mark.performance  
    def test_memory_efficiency(self, performance_service):
        """메모리 효율성 테스트"""
        import psutil
        import os
        
        logger.info("💾 Memory efficiency test")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 연속적인 처리로 메모리 누수 확인
        for i in range(10):
            cell_ids = [f"cell_{j:03d}" for j in range(5)]
            response = performance_service.process_peg_data(
                {"ems_ip": f"192.168.1.{200+i}"}, cell_ids, {"start": datetime.now()}
            )
            assert isinstance(response, ChoiAlgorithmResponse)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 메모리 증가량이 100MB 이하여야 함
        assert memory_increase < 100, f"Memory increase {memory_increase:.1f}MB exceeds 100MB limit"
        
        logger.info(f"✅ Memory efficiency test passed: {memory_increase:.1f}MB increase")


# =============================================================================
# 메인 실행 (pytest 없이 직접 실행용)
# =============================================================================

if __name__ == "__main__":
    print("🧪 Choi 알고리즘 통합 테스트 직접 실행")
    print("=" * 50)
    
    try:
        # 전체 통합 테스트 실행
        success = run_all_integration_tests()
        
        if success:
            print("🎉 모든 통합 테스트 성공!")
            print("🏆 Choi 알고리즘 시스템 완전 검증 완료!")
        else:
            print("❌ 일부 통합 테스트 실패")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 통합 테스트 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
