"""
회귀 테스트 픽스처 검증 스크립트

이 스크립트는 회귀 테스트 데이터의 무결성과 일관성을 검증합니다.
JSON 스키마 검증, 데이터 품질 검사, 파일 구조 검증을 수행합니다.

Author: Choi Algorithm Regression Test Team
Created: 2025-09-20
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import jsonschema
import sys

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegressionFixtureValidator:
    """
    회귀 테스트 픽스처 검증기
    
    JSON 스키마 검증, 데이터 품질 검사, 파일 구조 일관성을 검증합니다.
    """
    
    def __init__(self):
        """검증기 초기화"""
        self.data_dir = Path(__file__).parent / "data"
        self.schema_file = self.data_dir / "validation_schema.json"
        
        # JSON 스키마 로드
        with open(self.schema_file, 'r', encoding='utf-8') as f:
            self.validation_schema = json.load(f)
        
        self.validation_errors = []
        self.warnings = []
        
        logger.info("Regression fixture validator initialized")
    
    def validate_all_fixtures(self) -> bool:
        """모든 픽스처 검증"""
        try:
            logger.info("Starting comprehensive fixture validation")
            
            # 1. 디렉토리 구조 검증
            self._validate_directory_structure()
            
            # 2. 개별 픽스처 검증
            self._validate_individual_fixtures()
            
            # 3. 픽스처 간 일관성 검증
            self._validate_cross_fixture_consistency()
            
            # 4. 데이터 품질 검증
            self._validate_data_quality()
            
            # 결과 보고
            self._report_validation_results()
            
            return len(self.validation_errors) == 0
            
        except Exception as e:
            logger.error(f"Validation process failed: {e}")
            return False
    
    def _validate_directory_structure(self) -> None:
        """디렉토리 구조 검증"""
        logger.info("Validating directory structure")
        
        expected_dirs = ["normal_cases", "edge_cases", "abnormal_triggers", "boundary_conditions"]
        
        for dir_name in expected_dirs:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                self.validation_errors.append(f"Missing directory: {dir_name}")
            elif not dir_path.is_dir():
                self.validation_errors.append(f"Not a directory: {dir_name}")
        
        logger.info("Directory structure validation completed")
    
    def _validate_individual_fixtures(self) -> None:
        """개별 픽스처 검증"""
        logger.info("Validating individual fixtures")
        
        input_files = list(self.data_dir.rglob("*_input.json"))
        
        for input_file in input_files:
            try:
                self._validate_single_fixture(input_file)
            except Exception as e:
                self.validation_errors.append(f"Error validating {input_file.name}: {e}")
        
        logger.info(f"Individual fixture validation completed: {len(input_files)} files")
    
    def _validate_single_fixture(self, input_file: Path) -> None:
        """단일 픽스처 검증"""
        # JSON 로드 및 스키마 검증
        with open(input_file, 'r', encoding='utf-8') as f:
            fixture_data = json.load(f)
        
        try:
            jsonschema.validate(fixture_data, self.validation_schema)
        except jsonschema.ValidationError as e:
            self.validation_errors.append(f"{input_file.name}: Schema validation failed - {e.message}")
            return
        
        # 대응하는 예상 출력 파일 확인
        expected_file = input_file.parent / input_file.name.replace("_input.json", "_expected.json")
        if not expected_file.exists():
            self.warnings.append(f"{input_file.name}: Missing expected output file")
        
        # 데이터 일관성 검증
        self._validate_fixture_data_consistency(fixture_data, input_file.name)
    
    def _validate_fixture_data_consistency(self, data: Dict[str, Any], filename: str) -> None:
        """픽스처 데이터 일관성 검증"""
        # cell_ids와 peg_data 키 일치 확인
        cell_ids = set(data["cell_ids"])
        peg_data_keys = set(data["peg_data"].keys())
        
        if cell_ids != peg_data_keys:
            self.validation_errors.append(
                f"{filename}: cell_ids {cell_ids} != peg_data keys {peg_data_keys}"
            )
        
        # PEG 데이터 내부 일관성 확인
        for cell_id, peg_list in data["peg_data"].items():
            for peg_info in peg_list:
                if peg_info["cell_id"] != cell_id:
                    self.validation_errors.append(
                        f"{filename}: PEG cell_id mismatch in {cell_id}"
                    )
                
                # 샘플 길이 검증
                pre_len = len(peg_info["pre_samples"])
                post_len = len(peg_info["post_samples"])
                
                if pre_len == 0 or post_len == 0:
                    self.validation_errors.append(
                        f"{filename}: Empty samples in {cell_id}/{peg_info['peg_name']}"
                    )
                
                if abs(pre_len - post_len) > pre_len * 0.5:  # 50% 이상 차이
                    self.warnings.append(
                        f"{filename}: Large sample count difference in {cell_id}/{peg_info['peg_name']}"
                    )
    
    def _validate_cross_fixture_consistency(self) -> None:
        """픽스처 간 일관성 검증"""
        logger.info("Validating cross-fixture consistency")
        
        # 시나리오 이름 중복 확인
        scenario_names = []
        input_files = list(self.data_dir.rglob("*_input.json"))
        
        for input_file in input_files:
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    scenario_name = data.get("scenario_name")
                    if scenario_name in scenario_names:
                        self.validation_errors.append(f"Duplicate scenario name: {scenario_name}")
                    scenario_names.append(scenario_name)
            except Exception as e:
                self.validation_errors.append(f"Error reading {input_file.name}: {e}")
        
        logger.info("Cross-fixture consistency validation completed")
    
    def _validate_data_quality(self) -> None:
        """데이터 품질 검증"""
        logger.info("Validating data quality")
        
        input_files = list(self.data_dir.rglob("*_input.json"))
        
        for input_file in input_files:
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self._check_data_quality_metrics(data, input_file.name)
                
            except Exception as e:
                self.validation_errors.append(f"Data quality check failed for {input_file.name}: {e}")
        
        logger.info("Data quality validation completed")
    
    def _check_data_quality_metrics(self, data: Dict[str, Any], filename: str) -> None:
        """데이터 품질 지표 검증"""
        for cell_id, peg_list in data["peg_data"].items():
            for peg_info in peg_list:
                pre_samples = [x for x in peg_info["pre_samples"] if x is not None]
                post_samples = [x for x in peg_info["post_samples"] if x is not None]
                
                # ND 비율 검증
                total_pre = len(peg_info["pre_samples"])
                total_post = len(peg_info["post_samples"])
                nd_ratio_pre = (total_pre - len(pre_samples)) / total_pre if total_pre > 0 else 0
                nd_ratio_post = (total_post - len(post_samples)) / total_post if total_post > 0 else 0
                
                if nd_ratio_pre > 0.8 or nd_ratio_post > 0.8:
                    self.warnings.append(
                        f"{filename}: High ND ratio in {cell_id}/{peg_info['peg_name']}"
                    )
                
                # 값의 범위 검증 (음수 값 확인)
                all_values = pre_samples + post_samples
                if any(x < 0 for x in all_values):
                    self.warnings.append(
                        f"{filename}: Negative values in {cell_id}/{peg_info['peg_name']}"
                    )
                
                # 극한값 확인
                if all_values and (max(all_values) > 1e6 or max(all_values) < 1e-6):
                    self.warnings.append(
                        f"{filename}: Extreme values in {cell_id}/{peg_info['peg_name']}"
                    )
    
    def _report_validation_results(self) -> None:
        """검증 결과 보고"""
        print("\n" + "=" * 60)
        print("🔍 회귀 테스트 픽스처 검증 결과")
        print("=" * 60)
        
        if not self.validation_errors and not self.warnings:
            print("✅ 모든 검증 통과! 픽스처 데이터가 완벽합니다.")
        else:
            if self.validation_errors:
                print(f"❌ 검증 오류: {len(self.validation_errors)}개")
                for i, error in enumerate(self.validation_errors, 1):
                    print(f"   {i}. {error}")
            
            if self.warnings:
                print(f"⚠️ 경고: {len(self.warnings)}개")
                for i, warning in enumerate(self.warnings, 1):
                    print(f"   {i}. {warning}")
        
        print("=" * 60)


def main():
    """메인 실행 함수"""
    try:
        print("🔍 Choi 알고리즘 회귀 테스트 픽스처 검증")
        
        validator = RegressionFixtureValidator()
        is_valid = validator.validate_all_fixtures()
        
        if is_valid:
            print("🎉 모든 픽스처 검증 성공!")
            sys.exit(0)
        else:
            print("❌ 픽스처 검증 실패")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 검증 프로세스 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
