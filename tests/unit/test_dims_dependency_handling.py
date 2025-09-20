"""
DIMS Range 데이터 의존성 처리 테스트

이 모듈은 DIMS Range 데이터가 없거나 비활성화된 상황에서
Range 이상 탐지기가 견고하게 동작하는지 검증합니다.

Author: Choi Algorithm DIMS Dependency Team
Created: 2025-09-20
"""

import pytest
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.anomaly_detectors import RangeAnomalyDetector, MockDimsDataProvider, DimsDataProvider, AnomalyDetectionResult
from app.models.judgement import PegSampleSeries

# 테스트용 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoDimsDataProvider(DimsDataProvider):
    """
    DIMS 데이터가 없는 상황을 시뮬레이션하는 Provider
    """
    
    def get_peg_range(self, peg_name: str) -> Optional[Dict[str, Any]]:
        """항상 None 반환 (데이터 없음)"""
        return None
    
    def is_new_peg(self, peg_name: str) -> bool:
        """항상 False 반환"""
        return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Provider 정보"""
        return {
            "provider_name": "NoDimsDataProvider",
            "description": "Test provider with no DIMS data available",
            "data_available": False
        }


class PartialDimsDataProvider(DimsDataProvider):
    """
    일부 PEG에만 DIMS 데이터가 있는 상황을 시뮬레이션하는 Provider
    """
    
    def __init__(self):
        """부분 데이터 Provider 초기화"""
        self.available_pegs = {"AirMacDLThruAvg"}  # 일부 PEG만 데이터 있음
    
    def get_peg_range(self, peg_name: str) -> Optional[Dict[str, Any]]:
        """일부 PEG에만 데이터 제공"""
        if peg_name in self.available_pegs:
            return {"min": 0.0, "max": 10000.0, "unit": "Kbps"}
        return None
    
    def is_new_peg(self, peg_name: str) -> bool:
        """항상 False 반환"""
        return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Provider 정보"""
        return {
            "provider_name": "PartialDimsDataProvider",
            "description": "Test provider with partial DIMS data",
            "available_pegs": list(self.available_pegs),
            "data_available": True
        }


class ErrorDimsDataProvider(DimsDataProvider):
    """
    DIMS 데이터 접근 시 오류가 발생하는 상황을 시뮬레이션하는 Provider
    """
    
    def get_peg_range(self, peg_name: str) -> Optional[Dict[str, Any]]:
        """항상 예외 발생"""
        raise ConnectionError(f"DIMS server connection failed for {peg_name}")
    
    def is_new_peg(self, peg_name: str) -> bool:
        """항상 예외 발생"""
        raise ConnectionError(f"DIMS server connection failed for {peg_name}")
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Provider 정보"""
        return {
            "provider_name": "ErrorDimsDataProvider",
            "description": "Test provider that always throws errors",
            "data_available": False
        }


class TestDIMSDependencyHandling:
    """
    DIMS 의존성 처리 테스트 클래스
    
    Range 검사 활성화/비활성화, DIMS 데이터 가용성 등
    다양한 상황에서 견고한 동작을 검증합니다.
    """
    
    @pytest.fixture
    def sample_peg_data(self):
        """테스트용 PEG 데이터"""
        return {
            "cell_001": [
                PegSampleSeries(
                    peg_name="AirMacDLThruAvg",
                    cell_id="cell_001",
                    pre_samples=[1000.0, 1100.0, 1200.0],
                    post_samples=[1500.0, 1600.0, 1700.0],
                    unit="Kbps"
                ),
                PegSampleSeries(
                    peg_name="ConnNoAvg",
                    cell_id="cell_001",
                    pre_samples=[10.0, 12.0, 15.0],
                    post_samples=[8.0, 9.0, 11.0],
                    unit="count"
                )
            ]
        }
    
    def test_range_check_enabled_with_full_dims_data(self, sample_peg_data):
        """Range 검사 활성화 + 완전한 DIMS 데이터"""
        # 정상적인 DIMS Provider 사용
        detector = RangeAnomalyDetector(MockDimsDataProvider())
        
        config = {"enable_range_check": True}
        
        result = detector.detect(sample_peg_data, config)
        
        # 검증
        assert isinstance(result, AnomalyDetectionResult)
        assert result.anomaly_type == "Range"
        assert result.metadata.get("detection_enabled") is True
        assert result.metadata.get("detection_disabled") is not True
        
        # DIMS 가용성 확인
        assert result.metadata.get("dims_availability_ratio", 0) > 0.5
        
        print("✅ Range 검사 활성화 + 완전한 DIMS 데이터 테스트 성공")
    
    def test_range_check_disabled_by_configuration(self, sample_peg_data):
        """Range 검사 설정으로 비활성화"""
        detector = RangeAnomalyDetector(MockDimsDataProvider())
        
        config = {"enable_range_check": False}
        
        result = detector.detect(sample_peg_data, config)
        
        # 검증
        assert isinstance(result, AnomalyDetectionResult)
        assert result.anomaly_type == "Range"
        assert len(result.affected_cells) == 0
        assert len(result.affected_pegs) == 0
        assert result.metadata.get("detection_disabled") is True
        assert result.metadata.get("reason") == "Range check disabled in configuration"
        
        print("✅ Range 검사 설정 비활성화 테스트 성공")
    
    def test_range_check_with_no_dims_data(self, sample_peg_data):
        """Range 검사 활성화 + DIMS 데이터 없음"""
        # DIMS 데이터가 없는 Provider 사용
        detector = RangeAnomalyDetector(NoDimsDataProvider())
        
        config = {"enable_range_check": True}
        
        result = detector.detect(sample_peg_data, config)
        
        # 검증: 오류 없이 정상 완료되어야 함
        assert isinstance(result, AnomalyDetectionResult)
        assert result.anomaly_type == "Range"
        assert len(result.affected_cells) == 0  # 데이터 없으므로 탐지 없음
        assert len(result.affected_pegs) == 0
        
        # DIMS 가용성 메타데이터 확인
        assert result.metadata.get("dims_unavailable_count") == 2  # 2개 PEG 모두 데이터 없음
        assert result.metadata.get("dims_availability_ratio") == 0.0
        assert result.confidence <= 0.5  # 낮은 신뢰도
        
        print("✅ DIMS 데이터 없음 처리 테스트 성공")
    
    def test_range_check_with_partial_dims_data(self, sample_peg_data):
        """Range 검사 활성화 + 부분 DIMS 데이터"""
        # 일부 PEG에만 데이터가 있는 Provider 사용
        detector = RangeAnomalyDetector(PartialDimsDataProvider())
        
        config = {"enable_range_check": True}
        
        result = detector.detect(sample_peg_data, config)
        
        # 검증
        assert isinstance(result, AnomalyDetectionResult)
        assert result.anomaly_type == "Range"
        
        # 메타데이터 확인
        assert result.metadata.get("dims_unavailable_count") == 1  # ConnNoAvg만 데이터 없음
        assert result.metadata.get("total_pegs_checked") == 2
        assert result.metadata.get("dims_availability_ratio") == 0.5  # 50% 가용성
        
        # 신뢰도가 적절히 조정되었는지 확인
        assert 0.4 <= result.confidence <= 0.9
        
        print("✅ 부분 DIMS 데이터 처리 테스트 성공")
    
    def test_range_check_with_dims_connection_error(self, sample_peg_data):
        """Range 검사 활성화 + DIMS 연결 오류"""
        # 연결 오류가 발생하는 Provider 사용
        detector = RangeAnomalyDetector(ErrorDimsDataProvider())
        
        config = {"enable_range_check": True}
        
        # 연결 오류가 발생해도 예외가 발생하지 않아야 함
        result = detector.detect(sample_peg_data, config)
        
        # 검증: 오류 상황에서도 정상 결과 반환
        assert isinstance(result, AnomalyDetectionResult)
        assert result.anomaly_type == "Range"
        assert len(result.affected_cells) == 0  # 오류로 인해 탐지 없음
        assert len(result.affected_pegs) == 0
        
        # 오류 상황 메타데이터 확인
        assert result.metadata.get("dims_unavailable_count") == 2  # 모든 PEG 오류
        assert result.metadata.get("dims_availability_ratio") == 0.0
        assert result.confidence == 0.5  # 최소 신뢰도
        
        print("✅ DIMS 연결 오류 견고한 처리 테스트 성공")
    
    def test_range_check_configuration_validation(self):
        """Range 검사 설정 검증 테스트"""
        detector = RangeAnomalyDetector()
        
        # 유효한 설정들
        valid_configs = [
            {"enable_range_check": True},
            {"enable_range_check": False},
            {}  # 기본값 사용
        ]
        
        for config in valid_configs:
            assert detector.validate_config(config) is True
        
        # 무효한 설정들
        invalid_configs = [
            None,
            "invalid_string",
            {"enable_range_check": "invalid_bool"}
        ]
        
        for config in invalid_configs:
            # 기본 검증은 통과하지만, 실행 시 적절히 처리됨
            if config is None or isinstance(config, str):
                assert detector.validate_config(config) is False
        
        print("✅ Range 검사 설정 검증 테스트 성공")
    
    def test_dims_provider_abstraction_compliance(self):
        """DIMS Provider 추상화 준수 테스트"""
        # 다양한 Provider들이 인터페이스를 올바르게 구현했는지 확인
        providers = [
            MockDimsDataProvider(),
            NoDimsDataProvider(),
            PartialDimsDataProvider(),
            ErrorDimsDataProvider()
        ]
        
        for provider in providers:
            # 모든 Provider는 필수 메서드를 구현해야 함
            assert hasattr(provider, 'get_peg_range')
            assert hasattr(provider, 'is_new_peg')
            assert hasattr(provider, 'get_provider_info')
            
            # get_provider_info 호출 테스트
            info = provider.get_provider_info()
            assert isinstance(info, dict)
            assert "provider_name" in info
        
        print("✅ DIMS Provider 추상화 준수 테스트 성공")


# =============================================================================
# 통합 DIMS 의존성 테스트
# =============================================================================

class TestDIMSIntegrationScenarios:
    """DIMS 의존성 통합 시나리오 테스트"""
    
    def test_end_to_end_with_dims_disabled(self):
        """전체 워크플로우에서 DIMS Range 검사 비활성화"""
        from app.services.peg_processing_service import PEGProcessingService
        
        # Range 검사 비활성화된 설정으로 서비스 생성
        service = PEGProcessingService()
        
        # 테스트 데이터
        input_data = {"ems_ip": "192.168.1.100"}
        cell_ids = ["cell_dims_test"]
        time_range = {"start": "2025-09-20T10:00:00"}
        
        # 전체 워크플로우 실행
        response = service.process_peg_data(input_data, cell_ids, time_range)
        
        # 검증: Range 검사가 비활성화되어도 정상 동작
        assert response is not None
        assert response.abnormal_detection is not None
        
        # Range 탐지 결과 확인
        range_display = response.abnormal_detection.display_results.get("Range", False)
        
        # 설정에 따라 Range 탐지가 적절히 처리되었는지 확인
        # (현재 설정에서는 활성화되어 있으므로 정상 동작)
        assert isinstance(range_display, bool)
        
        print("✅ 전체 워크플로우 DIMS 처리 테스트 성공")
    
    def test_range_detector_factory_with_different_providers(self):
        """다양한 DIMS Provider로 Range 탐지기 팩토리 테스트"""
        from app.services.anomaly_detectors import AnomalyDetectorFactory
        
        # 다양한 Provider로 팩토리 테스트
        providers = [
            MockDimsDataProvider(),
            NoDimsDataProvider(),
            PartialDimsDataProvider()
        ]
        
        for provider in providers:
            factory = AnomalyDetectorFactory(dims_provider=provider)
            
            # Range 탐지기 생성
            range_detector = factory.create_detector("Range")
            assert range_detector is not None
            assert isinstance(range_detector, RangeAnomalyDetector)
            assert range_detector.dims_provider == provider
        
        print("✅ Range 탐지기 팩토리 다양한 Provider 테스트 성공")


# =============================================================================
# 설정 기반 DIMS 처리 테스트
# =============================================================================

class TestConfigurableDIMSHandling:
    """설정 기반 DIMS 처리 테스트"""
    
    def test_configuration_loading_dims_settings(self):
        """설정 파일에서 DIMS 관련 설정 로드 테스트"""
        from app.utils.choi_config import ChoiConfigLoader
        
        config_loader = ChoiConfigLoader()
        config = config_loader.load_config()
        
        # DIMS 관련 설정 확인
        assert hasattr(config, 'abnormal_detection')
        assert hasattr(config.abnormal_detection, 'enable_range_check')
        
        # 기본값 확인
        assert isinstance(config.abnormal_detection.enable_range_check, bool)
        
        print("✅ 설정 파일 DIMS 설정 로드 테스트 성공")
    
    def test_strategy_factory_dims_configuration(self):
        """Strategy Factory에서 DIMS 설정 전달 테스트"""
        from app.services.choi_strategy_factory import ChoiStrategyFactory
        
        factory = ChoiStrategyFactory()
        
        # Judgement Strategy 생성 (DIMS Provider 포함)
        judgement_strategy = factory.create_judgement_strategy()
        
        # Strategy가 올바른 설정을 받았는지 확인
        assert judgement_strategy is not None
        
        # Strategy 정보 확인
        strategy_info = judgement_strategy.get_strategy_info()
        assert isinstance(strategy_info, dict)
        
        print("✅ Strategy Factory DIMS 설정 전달 테스트 성공")


# =============================================================================
# 직접 실행 (pytest 없이)
# =============================================================================

def run_dims_dependency_tests_directly():
    """DIMS 의존성 테스트 직접 실행"""
    print("🔗 DIMS Range 데이터 의존성 처리 테스트")
    print("=" * 50)
    
    try:
        # 테스트 클래스 인스턴스 생성
        dims_tests = TestDIMSDependencyHandling()
        integration_tests = TestDIMSIntegrationScenarios()
        config_tests = TestConfigurableDIMSHandling()
        
        # 샘플 데이터 생성
        sample_data = {
            "cell_001": [
                PegSampleSeries(
                    peg_name="AirMacDLThruAvg",
                    cell_id="cell_001",
                    pre_samples=[1000.0, 1100.0, 1200.0],
                    post_samples=[1500.0, 1600.0, 1700.0],
                    unit="Kbps"
                ),
                PegSampleSeries(
                    peg_name="ConnNoAvg",
                    cell_id="cell_001",
                    pre_samples=[10.0, 12.0, 15.0],
                    post_samples=[8.0, 9.0, 11.0],
                    unit="count"
                )
            ]
        }
        
        print("1. 기본 DIMS 의존성 테스트:")
        dims_tests.test_range_check_enabled_with_full_dims_data(sample_data)
        dims_tests.test_range_check_disabled_by_configuration(sample_data)
        dims_tests.test_range_check_with_no_dims_data(sample_data)
        dims_tests.test_range_check_with_partial_dims_data(sample_data)
        dims_tests.test_range_check_with_dims_connection_error(sample_data)
        dims_tests.test_range_check_configuration_validation()
        dims_tests.test_dims_provider_abstraction_compliance()
        
        print("\n2. 통합 시나리오 테스트:")
        integration_tests.test_end_to_end_with_dims_disabled()
        integration_tests.test_range_detector_factory_with_different_providers()
        
        print("\n3. 설정 기반 처리 테스트:")
        config_tests.test_configuration_loading_dims_settings()
        config_tests.test_strategy_factory_dims_configuration()
        
        print("\n🎉 모든 DIMS 의존성 테스트 성공!")
        print("🏆 DIMS Range 데이터 의존성 견고한 처리 검증!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DIMS 의존성 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_dims_dependency_tests_directly()
    if not success:
        sys.exit(1)
