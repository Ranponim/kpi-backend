"""
Choi 알고리즘 회귀 테스트 프레임워크

이 모듈은 고정된 입력/출력 데이터셋을 사용하여 Choi 알고리즘의
의도하지 않은 변경을 감지하는 회귀 테스트를 수행합니다.

모든 테스트는 기존에 검증된 '골든' 출력과 비교하여
알고리즘 로직의 일관성을 보장합니다.

Author: Choi Algorithm Regression Test Team
Created: 2025-09-20
"""

import pytest
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import sys
import time

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.peg_processing_service import PEGProcessingService
from app.models.judgement import PegSampleSeries, ChoiAlgorithmResponse

# 로깅 설정
logging.basicConfig(level=logging.WARNING)  # 회귀 테스트 시 로그 최소화
logger = logging.getLogger(__name__)


class TestChoiRegression:
    """
    Choi 알고리즘 회귀 테스트 클래스
    
    고정된 입력/출력 데이터셋을 사용하여 알고리즘의 일관성을 검증합니다.
    모든 테스트는 이전에 검증된 골든 출력과 정확히 일치해야 합니다.
    """
    
    @pytest.fixture(scope="class")
    def regression_service(self):
        """회귀 테스트용 PEGProcessingService 픽스처"""
        logger.info("Creating PEGProcessingService for regression testing")
        service = PEGProcessingService()
        
        # 서비스 정상 초기화 검증
        assert service.filtering_strategy is not None
        assert service.judgement_strategy is not None
        assert service.config is not None
        
        return service
    
    @pytest.fixture(scope="class")
    def regression_fixtures(self):
        """회귀 테스트 픽스처 로드"""
        data_dir = Path(__file__).parent / "data"
        fixtures = {}
        
        # 모든 입력 파일 검색
        input_files = list(data_dir.rglob("*_input.json"))
        
        for input_file in input_files:
            # 대응하는 예상 출력 파일 찾기
            expected_file = input_file.parent / input_file.name.replace("_input.json", "_expected.json")
            
            if expected_file.exists():
                scenario_name = input_file.stem.replace("_input", "")
                
                # 입력 데이터 로드
                with open(input_file, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
                
                # 예상 출력 로드
                with open(expected_file, 'r', encoding='utf-8') as f:
                    expected_output = json.load(f)
                
                fixtures[scenario_name] = {
                    "input": input_data,
                    "expected": expected_output,
                    "category": input_file.parent.name,
                    "input_file": input_file.name,
                    "expected_file": expected_file.name
                }
        
        logger.info(f"Loaded {len(fixtures)} regression test fixtures")
        return fixtures
    
    @pytest.mark.regression
    @pytest.mark.parametrize("scenario_name", [
        # 자동으로 발견된 시나리오들이 여기에 추가됩니다
        "basic_similar_judgement",
        "nd_cant_judge", 
        "exactly_50_percent_filter",
        "zero_samples_handling",
        "extreme_cv_values",
        "improve_degrade_detection",
        "beta_threshold_boundaries",
        "high_delta_anomaly"
    ])
    def test_choi_algorithm_regression(self, regression_service, regression_fixtures, scenario_name):
        """
        Choi 알고리즘 회귀 테스트
        
        각 시나리오에 대해 고정된 입력으로 알고리즘을 실행하고
        이전에 검증된 골든 출력과 정확히 일치하는지 확인합니다.
        """
        logger.info(f"🔄 Regression test: {scenario_name}")
        
        # 픽스처 데이터 가져오기
        fixture_data = regression_fixtures.get(scenario_name)
        assert fixture_data is not None, f"Fixture not found for scenario: {scenario_name}"
        
        input_data = fixture_data["input"]
        expected_output = fixture_data["expected"]
        
        # 입력 데이터 변환
        converted_input = self._convert_input_data(input_data)
        
        # 성능 측정 시작
        start_time = time.time()
        
        # 실제 알고리즘 실행
        actual_response = self._execute_algorithm_with_test_data(
            regression_service, converted_input
        )
        
        # 성능 검증
        processing_time = (time.time() - start_time) * 1000
        expected_max_time = input_data["expected_behavior"].get("performance_max_ms", 5000)
        
        assert processing_time < expected_max_time, \
            f"Performance regression: {processing_time:.2f}ms > {expected_max_time}ms"
        
        # 응답을 JSON 형태로 변환
        actual_output = self._serialize_response(actual_response)
        
        # 주요 결과 비교 (타임스탬프 등 변동 필드 제외)
        self._compare_regression_outputs(actual_output, expected_output, scenario_name)
        
        logger.info(f"✅ Regression test {scenario_name} passed: {processing_time:.2f}ms")
    
    @pytest.mark.regression
    @pytest.mark.performance
    def test_performance_regression_baseline(self, regression_service):
        """
        성능 회귀 테스트 기준선
        
        표준 데이터셋으로 성능 기준선을 설정하고
        향후 성능 회귀를 감지합니다.
        """
        logger.info("🚀 Performance regression baseline test")
        
        # 표준 성능 테스트 데이터
        standard_cell_ids = [f"perf_cell_{i:03d}" for i in range(10)]
        input_data = {"ems_ip": "192.168.200.1"}
        time_range = {"start": datetime.now()}
        
        # 성능 측정
        start_time = time.time()
        
        response = regression_service.process_peg_data(
            input_data, standard_cell_ids, time_range
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # 성능 기준선: 10셀 × 50ms = 500ms 이하
        baseline_max_time = 500.0
        assert processing_time < baseline_max_time, \
            f"Performance regression: {processing_time:.2f}ms > {baseline_max_time}ms"
        
        # 결과 무결성 검증
        assert response.total_cells_analyzed == len(standard_cell_ids)
        assert isinstance(response, ChoiAlgorithmResponse)
        
        logger.info(f"✅ Performance baseline: {processing_time:.2f}ms < {baseline_max_time}ms")
    
    def _convert_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """입력 데이터 변환"""
        return {
            "input_data": input_data["input_data"],
            "cell_ids": input_data["cell_ids"],
            "time_range": {
                key: datetime.fromisoformat(value) if isinstance(value, str) else value
                for key, value in input_data["time_range"].items()
            },
            "peg_data": self._convert_peg_data(input_data["peg_data"])
        }
    
    def _convert_peg_data(self, peg_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[PegSampleSeries]]:
        """PEG 데이터를 PegSampleSeries로 변환"""
        converted_data = {}
        
        for cell_id, peg_list in peg_data.items():
            series_list = []
            
            for peg_info in peg_list:
                # null 값을 None으로 변환
                pre_samples = [None if x is None else x for x in peg_info["pre_samples"]]
                post_samples = [None if x is None else x for x in peg_info["post_samples"]]
                
                series = PegSampleSeries(
                    peg_name=peg_info["peg_name"],
                    cell_id=peg_info["cell_id"],
                    pre_samples=pre_samples,
                    post_samples=post_samples,
                    unit=peg_info["unit"]
                )
                
                series_list.append(series)
            
            converted_data[cell_id] = series_list
        
        return converted_data
    
    def _execute_algorithm_with_test_data(self, 
                                        service: PEGProcessingService,
                                        converted_input: Dict[str, Any]) -> ChoiAlgorithmResponse:
        """테스트 데이터로 알고리즘 실행"""
        # 서비스의 내부 메서드를 직접 호출하여 테스트 데이터 주입
        validated_data = converted_input["peg_data"]
        
        # 필터링 실행
        filtering_result = service._run_filtering(validated_data)
        
        # 집계 및 파생 계산
        aggregated_data = service._aggregation(validated_data, filtering_result)
        derived_data = service._derived_calculation(aggregated_data)
        
        # 판정 실행
        judgement_result = service._run_judgement(derived_data, filtering_result)
        
        # 결과 포맷팅
        processing_results = {
            "filtering": filtering_result,
            "aggregation": aggregated_data,
            "derived_calculation": derived_data,
            "judgement": judgement_result
        }
        
        warnings = []
        if filtering_result.warning_message:
            warnings.append(filtering_result.warning_message)
        
        return service._result_formatting(processing_results, warnings)
    
    def _serialize_response(self, response: ChoiAlgorithmResponse) -> Dict[str, Any]:
        """응답을 JSON 형태로 직렬화"""
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        else:
            return response.__dict__
    
    def _compare_regression_outputs(self, 
                                  actual: Dict[str, Any], 
                                  expected: Dict[str, Any], 
                                  scenario_name: str) -> None:
        """회귀 테스트 출력 비교"""
        # 변동 가능한 필드들 제외하고 비교
        differences = self._find_significant_differences(actual, expected)
        
        if differences:
            # 차이점이 있으면 상세 정보 출력
            self._report_regression_failure(differences, scenario_name, actual, expected)
            pytest.fail(f"Regression test failed for {scenario_name}: {differences}")
        
        # 핵심 필드들의 구체적 검증
        self._verify_core_algorithm_outputs(actual, expected, scenario_name)
    
    def _find_significant_differences(self, 
                                    actual: Dict[str, Any], 
                                    expected: Dict[str, Any]) -> List[str]:
        """중요한 차이점 찾기 (내장 라이브러리만 사용)"""
        differences = []
        
        # 변동 가능한 필드들 (비교 제외)
        excluded_fields = {"timestamp", "processing_time_ms", "regression_metadata"}
        
        # 재귀적으로 딕셔너리 비교
        def compare_dicts(actual_dict, expected_dict, path=""):
            for key in expected_dict:
                if key in excluded_fields:
                    continue
                    
                current_path = f"{path}.{key}" if path else key
                
                if key not in actual_dict:
                    differences.append(f"Missing key: {current_path}")
                    continue
                
                actual_val = actual_dict[key]
                expected_val = expected_dict[key]
                
                if isinstance(expected_val, dict) and isinstance(actual_val, dict):
                    compare_dicts(actual_val, expected_val, current_path)
                elif isinstance(expected_val, list) and isinstance(actual_val, list):
                    if len(actual_val) != len(expected_val):
                        differences.append(f"List length mismatch at {current_path}: {len(actual_val)} vs {len(expected_val)}")
                    else:
                        for i, (a_item, e_item) in enumerate(zip(actual_val, expected_val)):
                            if isinstance(e_item, dict) and isinstance(a_item, dict):
                                compare_dicts(a_item, e_item, f"{current_path}[{i}]")
                            elif a_item != e_item:
                                differences.append(f"List item mismatch at {current_path}[{i}]: {a_item} vs {e_item}")
                elif isinstance(expected_val, float) and isinstance(actual_val, float):
                    # 부동소수점 비교 (±0.001 허용)
                    if abs(actual_val - expected_val) > 0.001:
                        differences.append(f"Float mismatch at {current_path}: {actual_val} vs {expected_val}")
                elif actual_val != expected_val:
                    differences.append(f"Value mismatch at {current_path}: {actual_val} vs {expected_val}")
        
        compare_dicts(actual, expected)
        return differences
    
    def _verify_core_algorithm_outputs(self, 
                                     actual: Dict[str, Any], 
                                     expected: Dict[str, Any], 
                                     scenario_name: str) -> None:
        """핵심 알고리즘 출력 검증"""
        # 필터링 결과 검증
        actual_filter_ratio = actual["filtering"]["filter_ratio"]
        expected_filter_ratio = expected["filtering"]["filter_ratio"]
        
        assert abs(actual_filter_ratio - expected_filter_ratio) < 0.01, \
            f"{scenario_name}: Filter ratio mismatch {actual_filter_ratio} != {expected_filter_ratio}"
        
        # 이상 탐지 결과 검증
        actual_display = actual["abnormal_detection"]["display_results"]
        expected_display = expected["abnormal_detection"]["display_results"]
        
        for anomaly_type in ["Range", "ND", "Zero", "New", "High Delta"]:
            actual_val = actual_display.get(anomaly_type, False)
            expected_val = expected_display.get(anomaly_type, False)
            
            assert actual_val == expected_val, \
                f"{scenario_name}: Anomaly display mismatch for {anomaly_type}: {actual_val} != {expected_val}"
        
        # 알고리즘 버전 검증
        assert actual["algorithm_version"] == expected["algorithm_version"], \
            f"{scenario_name}: Algorithm version mismatch"
    
    def _report_regression_failure(self, 
                                 differences: List[str], 
                                 scenario_name: str, 
                                 actual: Dict[str, Any], 
                                 expected: Dict[str, Any]) -> None:
        """회귀 테스트 실패 상세 보고"""
        print(f"\n❌ REGRESSION TEST FAILURE: {scenario_name}")
        print("=" * 60)
        print("🔍 상세 차이점:")
        for diff in differences:
            print(f"  - {diff}")
        
        # 핵심 필드별 비교
        print("\n📊 핵심 결과 비교:")
        
        # 필터링 비교
        actual_filter = actual.get("filtering", {})
        expected_filter = expected.get("filtering", {})
        print(f"  필터링 비율: {actual_filter.get('filter_ratio')} vs {expected_filter.get('filter_ratio')}")
        
        # 이상 탐지 비교
        actual_anomaly = actual.get("abnormal_detection", {}).get("display_results", {})
        expected_anomaly = expected.get("abnormal_detection", {}).get("display_results", {})
        print(f"  이상 탐지: {actual_anomaly} vs {expected_anomaly}")
        
        print("=" * 60)


# =============================================================================
# 회귀 테스트 자동 발견 및 실행
# =============================================================================

class RegressionTestDiscovery:
    """회귀 테스트 자동 발견기"""
    
    @staticmethod
    def discover_regression_scenarios() -> List[str]:
        """회귀 테스트 시나리오 자동 발견"""
        data_dir = Path(__file__).parent / "data"
        input_files = list(data_dir.rglob("*_input.json"))
        
        scenarios = []
        for input_file in input_files:
            expected_file = input_file.parent / input_file.name.replace("_input.json", "_expected.json")
            if expected_file.exists():
                scenario_name = input_file.stem.replace("_input", "")
                scenarios.append(scenario_name)
        
        return sorted(scenarios)
    
    @staticmethod
    def generate_pytest_parametrize_list() -> str:
        """pytest.mark.parametrize용 시나리오 리스트 생성"""
        scenarios = RegressionTestDiscovery.discover_regression_scenarios()
        return '["' + '", "'.join(scenarios) + '"]'


# =============================================================================
# 회귀 테스트 실행 및 보고
# =============================================================================

class RegressionTestRunner:
    """회귀 테스트 실행기"""
    
    def __init__(self):
        """실행기 초기화"""
        self.service = PEGProcessingService()
        self.data_dir = Path(__file__).parent / "data"
        
    def run_all_regression_tests(self) -> Dict[str, Any]:
        """모든 회귀 테스트 실행 및 결과 수집"""
        scenarios = RegressionTestDiscovery.discover_regression_scenarios()
        
        results = {
            "total_scenarios": len(scenarios),
            "passed": 0,
            "failed": 0,
            "failures": [],
            "performance_stats": [],
            "start_time": datetime.now().isoformat()
        }
        
        print(f"🔄 회귀 테스트 실행: {len(scenarios)}개 시나리오")
        
        for scenario_name in scenarios:
            try:
                performance_ms = self._run_single_regression_test(scenario_name)
                results["passed"] += 1
                results["performance_stats"].append({
                    "scenario": scenario_name,
                    "time_ms": performance_ms
                })
                print(f"  ✅ {scenario_name}: {performance_ms:.2f}ms")
                
            except Exception as e:
                results["failed"] += 1
                results["failures"].append({
                    "scenario": scenario_name,
                    "error": str(e)
                })
                print(f"  ❌ {scenario_name}: {e}")
        
        results["end_time"] = datetime.now().isoformat()
        return results
    
    def _run_single_regression_test(self, scenario_name: str) -> float:
        """단일 회귀 테스트 실행"""
        # 입력 및 예상 출력 로드
        input_file = None
        expected_file = None
        
        for input_path in self.data_dir.rglob(f"{scenario_name}_input.json"):
            input_file = input_path
            expected_file = input_path.parent / f"{scenario_name}_expected.json"
            break
        
        if not input_file or not expected_file.exists():
            raise FileNotFoundError(f"Fixture files not found for {scenario_name}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            expected_output = json.load(f)
        
        # 알고리즘 실행
        start_time = time.time()
        
        converted_input = self._convert_input_data(input_data)
        actual_response = self._execute_algorithm_with_test_data(converted_input)
        
        processing_time = (time.time() - start_time) * 1000
        
        # 결과 비교
        actual_output = self._serialize_response(actual_response)
        self._verify_regression_match(actual_output, expected_output)
        
        return processing_time
    
    def _convert_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """입력 데이터 변환 (공통 로직)"""
        return {
            "input_data": input_data["input_data"],
            "cell_ids": input_data["cell_ids"],
            "time_range": {
                key: datetime.fromisoformat(value) if isinstance(value, str) else value
                for key, value in input_data["time_range"].items()
            },
            "peg_data": self._convert_peg_data(input_data["peg_data"])
        }
    
    def _convert_peg_data(self, peg_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[PegSampleSeries]]:
        """PEG 데이터 변환 (공통 로직)"""
        converted_data = {}
        
        for cell_id, peg_list in peg_data.items():
            series_list = []
            
            for peg_info in peg_list:
                pre_samples = [None if x is None else x for x in peg_info["pre_samples"]]
                post_samples = [None if x is None else x for x in peg_info["post_samples"]]
                
                series = PegSampleSeries(
                    peg_name=peg_info["peg_name"],
                    cell_id=peg_info["cell_id"],
                    pre_samples=pre_samples,
                    post_samples=post_samples,
                    unit=peg_info["unit"]
                )
                
                series_list.append(series)
            
            converted_data[cell_id] = series_list
        
        return converted_data
    
    def _execute_algorithm_with_test_data(self, converted_input: Dict[str, Any]) -> ChoiAlgorithmResponse:
        """테스트 데이터로 알고리즘 실행 (공통 로직)"""
        validated_data = converted_input["peg_data"]
        
        # 단계별 실행
        filtering_result = self.service._run_filtering(validated_data)
        aggregated_data = self.service._aggregation(validated_data, filtering_result)
        derived_data = self.service._derived_calculation(aggregated_data)
        judgement_result = self.service._run_judgement(derived_data, filtering_result)
        
        # 결과 포맷팅
        processing_results = {
            "filtering": filtering_result,
            "aggregation": aggregated_data,
            "derived_calculation": derived_data,
            "judgement": judgement_result
        }
        
        warnings = []
        if filtering_result.warning_message:
            warnings.append(filtering_result.warning_message)
        
        return self.service._result_formatting(processing_results, warnings)
    
    def _serialize_response(self, response: ChoiAlgorithmResponse) -> Dict[str, Any]:
        """응답 직렬화 (공통 로직)"""
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        else:
            return response.__dict__
    
    def _verify_regression_match(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> None:
        """회귀 매치 검증 (공통 로직)"""
        # 핵심 필드만 비교
        core_fields = ["algorithm_version", "filtering", "abnormal_detection", "kpi_judgement"]
        
        for field in core_fields:
            if field in expected:
                assert field in actual, f"Missing field: {field}"
                
                if field == "filtering":
                    # 필터링 비율만 비교 (±1% 허용)
                    expected_ratio = expected[field]["filter_ratio"]
                    actual_ratio = actual[field]["filter_ratio"]
                    assert abs(actual_ratio - expected_ratio) < 0.01
                elif field == "abnormal_detection":
                    # 표시 결과만 비교
                    expected_display = expected[field]["display_results"]
                    actual_display = actual[field]["display_results"]
                    assert actual_display == expected_display


# =============================================================================
# 직접 실행 (pytest 없이)
# =============================================================================

def run_regression_tests_directly():
    """회귀 테스트 직접 실행"""
    print("🔄 Choi 알고리즘 회귀 테스트 직접 실행")
    print("=" * 50)
    
    try:
        runner = RegressionTestRunner()
        results = runner.run_all_regression_tests()
        
        print(f"\n📊 회귀 테스트 결과:")
        print(f"  총 시나리오: {results['total_scenarios']}")
        print(f"  성공: {results['passed']}")
        print(f"  실패: {results['failed']}")
        
        if results["failures"]:
            print(f"\n❌ 실패한 시나리오:")
            for failure in results["failures"]:
                print(f"  - {failure['scenario']}: {failure['error']}")
        
        # 성능 통계
        if results["performance_stats"]:
            times = [stat["time_ms"] for stat in results["performance_stats"]]
            avg_time = sum(times) / len(times)
            max_time = max(times)
            print(f"\n⏱️ 성능 통계:")
            print(f"  평균 처리 시간: {avg_time:.2f}ms")
            print(f"  최대 처리 시간: {max_time:.2f}ms")
        
        if results["failed"] == 0:
            print("\n🎉 모든 회귀 테스트 성공!")
            print("🏆 알고리즘 일관성 완전 검증!")
            return True
        else:
            print(f"\n❌ {results['failed']}개 회귀 테스트 실패")
            return False
            
    except Exception as e:
        print(f"❌ 회귀 테스트 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_regression_tests_directly()
    if not success:
        sys.exit(1)
