"""
Choi 알고리즘 예상 출력 계산 스크립트

이 스크립트는 테스트 픽스처 데이터를 실제 Choi 알고리즘으로 처리하여
예상 출력 JSON을 생성합니다. 이를 통해 골든 테스트 데이터를 확보합니다.

Author: Choi Algorithm Test Team
Created: 2025-09-20
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.peg_processing_service import PEGProcessingService
from app.models.judgement import PegSampleSeries

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChoiTestOutputCalculator:
    """
    Choi 알고리즘 테스트 출력 계산기
    
    테스트 픽스처를 실제 알고리즘으로 처리하여 예상 출력을 생성합니다.
    """
    
    def __init__(self):
        """계산기 초기화"""
        self.service = PEGProcessingService()
        self.fixtures_dir = Path(__file__).parent
        self.expected_outputs_dir = self.fixtures_dir / "expected_outputs"
        self.expected_outputs_dir.mkdir(exist_ok=True)
        
        logger.info("Choi Test Output Calculator initialized")
    
    def calculate_all_scenarios(self) -> None:
        """모든 시나리오의 예상 출력 계산"""
        try:
            # 픽스처 파일들 찾기
            fixture_files = list(self.fixtures_dir.glob("scenario_*.json"))
            
            logger.info(f"Found {len(fixture_files)} scenario fixtures")
            
            for fixture_file in fixture_files:
                logger.info(f"Processing scenario: {fixture_file.name}")
                
                try:
                    self.calculate_scenario_output(fixture_file)
                    logger.info(f"✅ {fixture_file.name} processed successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process {fixture_file.name}: {e}")
                    continue
            
            logger.info("All scenarios processed")
            
        except Exception as e:
            logger.error(f"Error in calculate_all_scenarios: {e}")
            raise
    
    def calculate_scenario_output(self, fixture_file: Path) -> None:
        """개별 시나리오의 예상 출력 계산"""
        try:
            # 픽스처 로드
            with open(fixture_file, 'r', encoding='utf-8') as f:
                scenario_data = json.load(f)
            
            scenario_name = scenario_data["scenario_name"]
            logger.info(f"Calculating expected output for: {scenario_name}")
            
            # 입력 데이터 변환
            input_data = scenario_data["input_data"]
            cell_ids = scenario_data["cell_ids"]
            time_range = self._convert_time_range(scenario_data["time_range"])
            
            # PEG 데이터를 PegSampleSeries로 변환
            peg_data = self._convert_peg_data(scenario_data["peg_data"])
            
            # 실제 알고리즘 실행
            logger.info(f"Running Choi algorithm for {scenario_name}")
            
            # PEGProcessingService의 내부 메서드를 직접 사용하여 정확한 결과 생성
            response = self._run_choi_algorithm_directly(input_data, cell_ids, time_range, peg_data)
            
            # 예상 출력 저장
            output_file = self.expected_outputs_dir / f"{scenario_name}_expected.json"
            
            # 응답을 JSON serializable 형태로 변환
            expected_output = self._convert_response_to_json(response)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(expected_output, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Expected output saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error calculating scenario output: {e}")
            raise
    
    def _convert_time_range(self, time_range_data: Dict[str, str]) -> Dict[str, datetime]:
        """시간 범위 데이터 변환"""
        return {
            key: datetime.fromisoformat(value) if isinstance(value, str) else value
            for key, value in time_range_data.items()
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
    
    def _run_choi_algorithm_directly(self, 
                                   input_data: Dict[str, Any],
                                   cell_ids: List[str],
                                   time_range: Dict[str, datetime],
                                   peg_data: Dict[str, List[PegSampleSeries]]) -> Any:
        """
        Choi 알고리즘을 직접 실행하여 정확한 결과 생성
        
        실제 PEGProcessingService를 사용하되, 테스트 데이터를 직접 주입
        """
        try:
            logger.info("Running Choi algorithm with test data injection")
            
            # 서비스의 내부 메서드들을 순차적으로 실행
            
            # 1-2단계: 데이터 검증 (테스트 데이터 직접 사용)
            validated_data = peg_data
            
            # 3단계: 필터링 실행
            filtering_result = self.service._run_filtering(validated_data)
            
            # 4-5단계: 집계 및 파생 계산 (Mock)
            aggregated_data = self.service._aggregation(validated_data, filtering_result)
            derived_data = self.service._derived_calculation(aggregated_data)
            
            # 6단계: 판정 실행
            judgement_result = self.service._run_judgement(derived_data, filtering_result)
            
            # 7단계: 결과 포맷팅
            processing_results = {
                "filtering": filtering_result,
                "aggregation": aggregated_data,
                "derived_calculation": derived_data,
                "judgement": judgement_result
            }
            
            warnings = []
            if filtering_result.warning_message:
                warnings.append(filtering_result.warning_message)
            
            response = self.service._result_formatting(processing_results, warnings)
            
            logger.info("Choi algorithm execution completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error running Choi algorithm: {e}")
            raise
    
    def _convert_response_to_json(self, response: Any) -> Dict[str, Any]:
        """응답 객체를 JSON serializable 딕셔너리로 변환"""
        try:
            # Pydantic 모델인 경우 model_dump 사용
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            
            # 일반 객체인 경우 __dict__ 사용
            elif hasattr(response, '__dict__'):
                return self._serialize_dict(response.__dict__)
            
            # 기본 타입인 경우 그대로 반환
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error converting response to JSON: {e}")
            return {"error": str(e)}
    
    def _serialize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """딕셔너리를 JSON serializable 형태로 변환"""
        result = {}
        
        for key, value in data.items():
            if hasattr(value, 'model_dump'):
                result[key] = value.model_dump()
            elif isinstance(value, dict):
                result[key] = self._serialize_dict(value)
            elif isinstance(value, list):
                result[key] = [self._serialize_dict(item) if isinstance(item, dict) else item for item in value]
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        
        return result


def main():
    """메인 실행 함수"""
    try:
        print("🧮 Choi 알고리즘 예상 출력 계산 시작")
        print("=" * 50)
        
        calculator = ChoiTestOutputCalculator()
        calculator.calculate_all_scenarios()
        
        print("✅ 모든 시나리오 예상 출력 계산 완료")
        print("📁 결과 저장 위치: tests/fixtures/choi_algorithm/expected_outputs/")
        
    except Exception as e:
        print(f"❌ 예상 출력 계산 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
