"""
회귀 테스트 예상 출력 계산 스크립트

이 스크립트는 회귀 테스트용 입력 데이터를 실제 Choi 알고리즘으로 처리하여
골든 출력 데이터를 생성합니다. 이는 향후 알고리즘 변경 시 회귀 검증의 기준이 됩니다.

Author: Choi Algorithm Regression Test Team
Created: 2025-09-20
"""

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
from app.models.judgement import PegSampleSeries

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegressionOutputCalculator:
    """
    회귀 테스트 출력 계산기
    
    회귀 테스트용 입력 데이터를 실제 알고리즘으로 처리하여
    정확한 골든 출력을 생성합니다.
    """
    
    def __init__(self):
        """계산기 초기화"""
        self.service = PEGProcessingService()
        self.data_dir = Path(__file__).parent / "data"
        
        logger.info("Regression output calculator initialized")
    
    def calculate_all_regression_outputs(self) -> None:
        """모든 회귀 테스트 시나리오의 출력 계산"""
        try:
            # 모든 입력 파일 찾기
            input_files = list(self.data_dir.rglob("*_input.json"))
            
            logger.info(f"Found {len(input_files)} regression input files")
            
            success_count = 0
            
            for input_file in input_files:
                try:
                    self._calculate_single_output(input_file)
                    success_count += 1
                    logger.info(f"✅ {input_file.name} processed successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process {input_file.name}: {e}")
                    continue
            
            logger.info(f"Regression output calculation completed: {success_count}/{len(input_files)} successful")
            
        except Exception as e:
            logger.error(f"Error in calculate_all_regression_outputs: {e}")
            raise
    
    def _calculate_single_output(self, input_file: Path) -> None:
        """단일 회귀 테스트 출력 계산"""
        try:
            # 입력 데이터 로드
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            scenario_name = input_data["scenario_name"]
            logger.info(f"Calculating regression output for: {scenario_name}")
            
            # PEG 데이터를 PegSampleSeries로 변환
            converted_data = self._convert_peg_data(input_data["peg_data"])
            
            # 실제 알고리즘 실행
            response = self._run_algorithm_with_test_data(
                input_data["input_data"],
                input_data["cell_ids"],
                self._convert_time_range(input_data["time_range"]),
                converted_data
            )
            
            # 예상 출력 저장
            output_file = input_file.parent / input_file.name.replace("_input.json", "_expected.json")
            
            # 응답을 JSON serializable 형태로 변환
            output_data = self._serialize_response(response)
            
            # 회귀 테스트 메타데이터 추가
            output_data["regression_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "algorithm_version": response.algorithm_version,
                "input_file": input_file.name,
                "scenario_category": input_file.parent.name,
                "test_purpose": input_data["test_purpose"]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Regression output saved: {output_file.name}")
            
        except Exception as e:
            logger.error(f"Error calculating single output: {e}")
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
    
    def _run_algorithm_with_test_data(self, 
                                    input_data: Dict[str, Any],
                                    cell_ids: List[str],
                                    time_range: Dict[str, datetime],
                                    peg_data: Dict[str, List[PegSampleSeries]]) -> Any:
        """테스트 데이터로 알고리즘 실행"""
        try:
            logger.info("Running Choi algorithm with regression test data")
            
            # 실제 PEGProcessingService 내부 메서드 직접 호출
            # (Mock 데이터 대신 실제 테스트 데이터 사용)
            
            # 1-2단계: 데이터 검증 (테스트 데이터 직접 사용)
            validated_data = peg_data
            
            # 3단계: 필터링 실행
            filtering_result = self.service._run_filtering(validated_data)
            
            # 4-5단계: 집계 및 파생 계산
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
            
            logger.info("Regression algorithm execution completed")
            return response
            
        except Exception as e:
            logger.error(f"Error running algorithm with test data: {e}")
            raise
    
    def _serialize_response(self, response: Any) -> Dict[str, Any]:
        """응답 객체를 JSON serializable 형태로 변환"""
        try:
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            elif hasattr(response, '__dict__'):
                return self._serialize_dict(response.__dict__)
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error serializing response: {e}")
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
                result[key] = [
                    self._serialize_dict(item) if isinstance(item, dict) 
                    else item for item in value
                ]
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        
        return result


def main():
    """메인 실행 함수"""
    try:
        print("🧮 Choi 알고리즘 회귀 테스트 출력 계산")
        print("=" * 50)
        
        calculator = RegressionOutputCalculator()
        calculator.calculate_all_regression_outputs()
        
        print("✅ 모든 회귀 테스트 출력 계산 완료")
        print("📁 결과 저장: tests/regression/data/**/*_expected.json")
        
    except Exception as e:
        print(f"❌ 회귀 테스트 출력 계산 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
