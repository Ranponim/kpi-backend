"""
Choi 알고리즘 성능 벤치마크 스위트

이 모듈은 Choi 알고리즘의 성능을 종합적으로 측정하고 분석합니다.
PRD 4.3의 성능 요구사항 (< 5초)을 검증하고 병목 지점을 식별합니다.

Author: Choi Algorithm Performance Team
Created: 2025-09-20
"""

import time
import logging
import statistics
import psutil
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
import json
import cProfile
import pstats
import io

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.peg_processing_service import PEGProcessingService
from app.models.judgement import PegSampleSeries

# 로깅 설정
logging.basicConfig(level=logging.WARNING)  # 벤치마크 시 로그 최소화
logger = logging.getLogger(__name__)


class ChoiPerformanceBenchmark:
    """
    Choi 알고리즘 성능 벤치마크
    
    다양한 데이터 크기와 복잡도로 성능을 측정하고
    PRD 4.3 요구사항 대비 성능을 검증합니다.
    """
    
    def __init__(self):
        """벤치마크 초기화"""
        self.service = PEGProcessingService()
        self.results = []
        self.baseline_results = {}
        
        # 성능 요구사항 (PRD 4.3)
        self.performance_targets = {
            "small_workload": 100,      # 1-2셀: 100ms
            "standard_workload": 5000,  # 10셀: 5초
            "large_workload": 15000     # 50셀: 15초
        }
        
        logger.info("Choi performance benchmark initialized")
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """포괄적 성능 벤치마크 실행"""
        print("🚀 Choi 알고리즘 포괄적 성능 벤치마크")
        print("=" * 60)
        
        try:
            # 1. 기본 성능 벤치마크
            print("1. 기본 성능 벤치마크:")
            basic_results = self._run_basic_performance_tests()
            
            # 2. 확장성 테스트
            print("\n2. 확장성 테스트:")
            scalability_results = self._run_scalability_tests()
            
            # 3. 메모리 사용량 테스트
            print("\n3. 메모리 사용량 테스트:")
            memory_results = self._run_memory_tests()
            
            # 4. 단계별 성능 분석
            print("\n4. 단계별 성능 분석:")
            phase_results = self._run_phase_performance_analysis()
            
            # 5. 성능 회귀 기준선 설정
            print("\n5. 성능 회귀 기준선:")
            baseline_results = self._establish_performance_baseline()
            
            # 종합 결과
            comprehensive_results = {
                "timestamp": datetime.now().isoformat(),
                "basic_performance": basic_results,
                "scalability": scalability_results,
                "memory_usage": memory_results,
                "phase_analysis": phase_results,
                "baseline": baseline_results,
                "targets_met": self._verify_performance_targets(basic_results)
            }
            
            # 결과 저장
            self._save_benchmark_results(comprehensive_results)
            
            # 요약 보고
            self._report_benchmark_summary(comprehensive_results)
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            raise
    
    def _run_basic_performance_tests(self) -> Dict[str, Any]:
        """기본 성능 테스트"""
        test_cases = [
            {"name": "small_workload", "cells": 2, "iterations": 10},
            {"name": "standard_workload", "cells": 10, "iterations": 5}, 
            {"name": "large_workload", "cells": 50, "iterations": 3}
        ]
        
        results = {}
        
        for test_case in test_cases:
            print(f"  🔄 {test_case['name']}: {test_case['cells']}셀 × {test_case['iterations']}회")
            
            times = []
            for i in range(test_case['iterations']):
                # 테스트 데이터 생성
                cell_ids = [f"bench_cell_{j:03d}" for j in range(test_case['cells'])]
                input_data = {"ems_ip": "192.168.200.1"}
                time_range = {"start": datetime.now()}
                
                # 성능 측정
                start_time = time.perf_counter()
                
                response = self.service.process_peg_data(input_data, cell_ids, time_range)
                
                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000  # ms
                
                times.append(execution_time)
                
                # 결과 검증
                assert response.total_cells_analyzed == test_case['cells']
                assert response.total_pegs_analyzed == test_case['cells'] * 3  # 셀당 3개 PEG
            
            # 통계 계산
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            
            target_time = self.performance_targets[test_case['name']]
            meets_target = avg_time < target_time
            
            results[test_case['name']] = {
                "cells": test_case['cells'],
                "iterations": test_case['iterations'],
                "avg_time_ms": round(avg_time, 2),
                "min_time_ms": round(min_time, 2),
                "max_time_ms": round(max_time, 2),
                "std_dev_ms": round(std_dev, 2),
                "target_ms": target_time,
                "meets_target": meets_target,
                "performance_ratio": round(avg_time / target_time, 3)
            }
            
            status = "✅" if meets_target else "❌"
            print(f"    {status} 평균: {avg_time:.2f}ms (목표: {target_time}ms, "
                  f"비율: {avg_time/target_time:.1%})")
        
        return results
    
    def _run_scalability_tests(self) -> Dict[str, Any]:
        """확장성 테스트"""
        cell_counts = [1, 2, 5, 10, 20, 50]
        results = {}
        
        for cell_count in cell_counts:
            print(f"  📈 {cell_count}셀 확장성 테스트")
            
            # 테스트 데이터 생성
            cell_ids = [f"scale_cell_{i:03d}" for i in range(cell_count)]
            input_data = {"ems_ip": "192.168.200.2"}
            time_range = {"start": datetime.now()}
            
            # 3회 측정 후 평균
            times = []
            for _ in range(3):
                start_time = time.perf_counter()
                response = self.service.process_peg_data(input_data, cell_ids, time_range)
                end_time = time.perf_counter()
                
                execution_time = (end_time - start_time) * 1000
                times.append(execution_time)
            
            avg_time = statistics.mean(times)
            
            # 셀당 처리 시간 계산
            time_per_cell = avg_time / cell_count
            
            results[f"{cell_count}_cells"] = {
                "cell_count": cell_count,
                "total_time_ms": round(avg_time, 2),
                "time_per_cell_ms": round(time_per_cell, 2),
                "pegs_analyzed": response.total_pegs_analyzed,
                "linear_scaling": cell_count <= 10 or time_per_cell < 100  # 100ms/셀 이하면 양호
            }
            
            print(f"    ⏱️ {avg_time:.2f}ms (셀당 {time_per_cell:.2f}ms)")
        
        # 선형 확장성 분석
        scaling_analysis = self._analyze_scaling_linearity(results)
        results["scaling_analysis"] = scaling_analysis
        
        return results
    
    def _run_memory_tests(self) -> Dict[str, Any]:
        """메모리 사용량 테스트"""
        print("  🧠 메모리 사용량 분석")
        
        # 메모리 추적 시작
        tracemalloc.start()
        
        # 기본 메모리 상태
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 대용량 데이터로 메모리 테스트
        cell_ids = [f"memory_cell_{i:03d}" for i in range(20)]
        input_data = {"ems_ip": "192.168.200.3"}
        time_range = {"start": datetime.now()}
        
        # 메모리 측정
        start_memory = process.memory_info().rss / 1024 / 1024
        
        response = self.service.process_peg_data(input_data, cell_ids, time_range)
        
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = end_memory - start_memory
        
        # tracemalloc 통계
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 메모리 효율성 계산 (MB per cell)
        memory_per_cell = memory_increase / len(cell_ids)
        
        results = {
            "initial_memory_mb": round(initial_memory, 2),
            "start_memory_mb": round(start_memory, 2),
            "end_memory_mb": round(end_memory, 2),
            "memory_increase_mb": round(memory_increase, 2),
            "memory_per_cell_mb": round(memory_per_cell, 3),
            "tracemalloc_current_mb": round(current / 1024 / 1024, 2),
            "tracemalloc_peak_mb": round(peak / 1024 / 1024, 2),
            "cells_tested": len(cell_ids),
            "memory_efficient": memory_per_cell < 1.0  # 1MB/셀 이하면 효율적
        }
        
        print(f"    📊 메모리 증가: {memory_increase:.2f}MB "
              f"(셀당 {memory_per_cell:.3f}MB)")
        print(f"    🔍 Peak 메모리: {peak/1024/1024:.2f}MB")
        
        return results
    
    def _run_phase_performance_analysis(self) -> Dict[str, Any]:
        """단계별 성능 분석"""
        print("  📊 단계별 성능 분석")
        
        # 표준 테스트 데이터
        cell_ids = [f"phase_cell_{i:03d}" for i in range(10)]
        input_data = {"ems_ip": "192.168.200.4"}
        time_range = {"start": datetime.now()}
        
        # 단계별 시간 측정을 위한 커스텀 실행
        phase_times = {}
        
        # Mock 데이터 생성
        peg_data = self._generate_test_peg_data(cell_ids)
        
        # 1단계: 데이터 조회 (Mock이므로 제외)
        
        # 2단계: 데이터 검증
        start_time = time.perf_counter()
        validated_data = peg_data  # 이미 검증된 데이터
        validation_time = (time.perf_counter() - start_time) * 1000
        
        # 3단계: 필터링
        start_time = time.perf_counter()
        filtering_result = self.service._run_filtering(validated_data)
        filtering_time = (time.perf_counter() - start_time) * 1000
        
        # 4단계: 집계
        start_time = time.perf_counter()
        aggregated_data = self.service._aggregation(validated_data, filtering_result)
        aggregation_time = (time.perf_counter() - start_time) * 1000
        
        # 5단계: 파생 계산
        start_time = time.perf_counter()
        derived_data = self.service._derived_calculation(aggregated_data)
        derivation_time = (time.perf_counter() - start_time) * 1000
        
        # 6단계: 판정
        start_time = time.perf_counter()
        judgement_result = self.service._run_judgement(derived_data, filtering_result)
        judgement_time = (time.perf_counter() - start_time) * 1000
        
        # 7단계: 포맷팅
        start_time = time.perf_counter()
        processing_results = {
            "filtering": filtering_result,
            "aggregation": aggregated_data,
            "derived_calculation": derived_data,
            "judgement": judgement_result
        }
        response = self.service._result_formatting(processing_results, [])
        formatting_time = (time.perf_counter() - start_time) * 1000
        
        total_time = validation_time + filtering_time + aggregation_time + derivation_time + judgement_time + formatting_time
        
        phase_times = {
            "validation": round(validation_time, 3),
            "filtering": round(filtering_time, 3),
            "aggregation": round(aggregation_time, 3),
            "derivation": round(derivation_time, 3),
            "judgement": round(judgement_time, 3),
            "formatting": round(formatting_time, 3),
            "total": round(total_time, 3)
        }
        
        # 각 단계별 비율 계산
        phase_percentages = {
            phase: round((time_ms / total_time) * 100, 1) if total_time > 0 else 0
            for phase, time_ms in phase_times.items() if phase != "total"
        }
        
        results = {
            "cells_analyzed": len(cell_ids),
            "phase_times_ms": phase_times,
            "phase_percentages": phase_percentages,
            "bottleneck_phase": max(phase_percentages.items(), key=lambda x: x[1])[0],
            "meets_target": total_time < self.performance_targets["standard_workload"]
        }
        
        print(f"    ⏱️ 총 시간: {total_time:.3f}ms")
        print(f"    🔥 병목: {results['bottleneck_phase']} ({phase_percentages[results['bottleneck_phase']]}%)")
        
        return results
    
    def _run_scalability_tests(self) -> Dict[str, Any]:
        """확장성 테스트"""
        cell_counts = [1, 2, 5, 10, 20, 50]
        scalability_data = []
        
        for cell_count in cell_counts:
            print(f"  📈 {cell_count}셀 확장성 측정")
            
            # 3회 측정
            times = []
            for _ in range(3):
                cell_ids = [f"scale_cell_{i:03d}" for i in range(cell_count)]
                input_data = {"ems_ip": "192.168.200.5"}
                time_range = {"start": datetime.now()}
                
                start_time = time.perf_counter()
                response = self.service.process_peg_data(input_data, cell_ids, time_range)
                end_time = time.perf_counter()
                
                execution_time = (end_time - start_time) * 1000
                times.append(execution_time)
            
            avg_time = statistics.mean(times)
            time_per_cell = avg_time / cell_count
            
            scalability_data.append({
                "cell_count": cell_count,
                "avg_time_ms": round(avg_time, 2),
                "time_per_cell_ms": round(time_per_cell, 2),
                "pegs_analyzed": response.total_pegs_analyzed
            })
            
            print(f"    ⏱️ {avg_time:.2f}ms (셀당 {time_per_cell:.2f}ms)")
        
        # 선형성 분석
        linearity_score = self._calculate_linearity_score(scalability_data)
        
        return {
            "scalability_data": scalability_data,
            "linearity_score": linearity_score,
            "is_linear": linearity_score > 0.8,
            "max_time_per_cell": max(data["time_per_cell_ms"] for data in scalability_data)
        }
    
    def _run_memory_tests(self) -> Dict[str, Any]:
        """메모리 사용량 테스트"""
        memory_tests = [
            {"name": "small_dataset", "cells": 5, "samples_per_peg": 20},
            {"name": "medium_dataset", "cells": 10, "samples_per_peg": 40},
            {"name": "large_dataset", "cells": 20, "samples_per_peg": 100}
        ]
        
        results = {}
        
        for test in memory_tests:
            print(f"  🧠 {test['name']}: {test['cells']}셀 × {test['samples_per_peg']}샘플")
            
            # 메모리 추적 시작
            tracemalloc.start()
            process = psutil.Process()
            
            start_memory = process.memory_info().rss / 1024 / 1024
            
            # 큰 데이터셋으로 테스트
            large_peg_data = self._generate_large_test_data(test['cells'], test['samples_per_peg'])
            
            # 알고리즘 실행
            filtering_result = self.service._run_filtering(large_peg_data)
            aggregated_data = self.service._aggregation(large_peg_data, filtering_result)
            derived_data = self.service._derived_calculation(aggregated_data)
            judgement_result = self.service._run_judgement(derived_data, filtering_result)
            
            end_memory = process.memory_info().rss / 1024 / 1024
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_increase = end_memory - start_memory
            total_samples = test['cells'] * 3 * test['samples_per_peg'] * 2  # 셀×PEG×샘플×pre/post
            
            results[test['name']] = {
                "cells": test['cells'],
                "samples_per_peg": test['samples_per_peg'],
                "total_samples": total_samples,
                "memory_increase_mb": round(memory_increase, 3),
                "peak_memory_mb": round(peak / 1024 / 1024, 3),
                "memory_per_sample_bytes": round((peak / total_samples), 2) if total_samples > 0 else 0,
                "memory_efficient": memory_increase < 50  # 50MB 이하면 효율적
            }
            
            print(f"    📊 메모리 증가: {memory_increase:.3f}MB, "
                  f"샘플당: {results[test['name']]['memory_per_sample_bytes']:.2f}B")
        
        return results
    
    def _establish_performance_baseline(self) -> Dict[str, Any]:
        """성능 회귀 기준선 설정"""
        print("  📏 성능 기준선 설정 (10셀 표준)")
        
        # 표준 워크로드로 10회 측정
        times = []
        cell_ids = [f"baseline_cell_{i:03d}" for i in range(10)]
        
        for i in range(10):
            input_data = {"ems_ip": f"192.168.200.{10+i}"}
            time_range = {"start": datetime.now()}
            
            start_time = time.perf_counter()
            response = self.service.process_peg_data(input_data, cell_ids, time_range)
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000
            times.append(execution_time)
        
        # 통계 계산
        baseline = {
            "measurements": len(times),
            "mean_ms": round(statistics.mean(times), 3),
            "median_ms": round(statistics.median(times), 3),
            "std_dev_ms": round(statistics.stdev(times), 3),
            "min_ms": round(min(times), 3),
            "max_ms": round(max(times), 3),
            "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 3),
            "p99_ms": round(sorted(times)[int(len(times) * 0.99)], 3),
            "coefficient_of_variation": round(statistics.stdev(times) / statistics.mean(times), 3),
            "stable_performance": statistics.stdev(times) / statistics.mean(times) < 0.1
        }
        
        print(f"    📊 기준선: 평균 {baseline['mean_ms']:.3f}ms ± {baseline['std_dev_ms']:.3f}ms")
        print(f"    📈 P95: {baseline['p95_ms']:.3f}ms, P99: {baseline['p99_ms']:.3f}ms")
        
        return baseline
    
    def _generate_test_peg_data(self, cell_ids: List[str]) -> Dict[str, List[PegSampleSeries]]:
        """테스트용 PEG 데이터 생성"""
        peg_data = {}
        
        for cell_id in cell_ids:
            peg_series = []
            
            # 각 셀당 3개 PEG
            peg_names = ["AirMacDLThruAvg", "AirMacULThruAvg", "ConnNoAvg"]
            
            for peg_name in peg_names:
                # 20개 샘플 생성
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
    
    def _generate_large_test_data(self, cell_count: int, samples_per_peg: int) -> Dict[str, List[PegSampleSeries]]:
        """대용량 테스트 데이터 생성"""
        peg_data = {}
        
        for i in range(cell_count):
            cell_id = f"large_cell_{i:03d}"
            peg_series = []
            
            peg_names = ["AirMacDLThruAvg", "AirMacULThruAvg", "ConnNoAvg"]
            
            for peg_name in peg_names:
                # 지정된 수의 샘플 생성
                pre_samples = [1000.0 + j * 5 for j in range(samples_per_peg)]
                post_samples = [1050.0 + j * 6 for j in range(samples_per_peg)]
                
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
    
    def _analyze_scaling_linearity(self, scalability_results: Dict[str, Any]) -> Dict[str, Any]:
        """확장성 선형성 분석"""
        # 셀 수와 처리 시간의 상관관계 분석
        data_points = []
        for key, result in scalability_results.items():
            if key != "scaling_analysis" and isinstance(result, dict):
                data_points.append((result["cell_count"], result["total_time_ms"]))
        
        if len(data_points) < 2:
            return {"linearity_score": 0.0, "analysis": "Insufficient data"}
        
        # 선형 회귀 계산 (간단한 상관관계)
        x_values = [point[0] for point in data_points]
        y_values = [point[1] for point in data_points]
        
        # 피어슨 상관계수 계산
        if len(x_values) > 1:
            correlation = self._calculate_correlation(x_values, y_values)
        else:
            correlation = 0.0
        
        return {
            "correlation_coefficient": round(correlation, 3),
            "linearity_score": round(abs(correlation), 3),
            "is_linear": abs(correlation) > 0.8,
            "data_points": data_points
        }
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """피어슨 상관계수 계산"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_linearity_score(self, scalability_data: List[Dict[str, Any]]) -> float:
        """선형성 점수 계산"""
        if len(scalability_data) < 3:
            return 0.0
        
        # 셀당 처리 시간의 일관성 확인
        times_per_cell = [data["time_per_cell_ms"] for data in scalability_data]
        
        if not times_per_cell:
            return 0.0
        
        # 변동계수 계산 (낮을수록 선형성 높음)
        mean_time = statistics.mean(times_per_cell)
        std_dev = statistics.stdev(times_per_cell) if len(times_per_cell) > 1 else 0
        
        coefficient_of_variation = std_dev / mean_time if mean_time > 0 else 1.0
        
        # 선형성 점수 (변동계수가 낮을수록 높은 점수)
        linearity_score = max(0.0, 1.0 - coefficient_of_variation)
        
        return linearity_score
    
    def _verify_performance_targets(self, basic_results: Dict[str, Any]) -> Dict[str, bool]:
        """성능 목표 달성 여부 검증"""
        targets_met = {}
        
        for workload_name, result in basic_results.items():
            target_time = self.performance_targets.get(workload_name, float('inf'))
            actual_time = result["avg_time_ms"]
            targets_met[workload_name] = actual_time < target_time
        
        return targets_met
    
    def _save_benchmark_results(self, results: Dict[str, Any]) -> None:
        """벤치마크 결과 저장"""
        output_file = Path(__file__).parent / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📁 벤치마크 결과 저장: {output_file.name}")
    
    def _report_benchmark_summary(self, results: Dict[str, Any]) -> None:
        """벤치마크 요약 보고"""
        print("\n" + "=" * 60)
        print("📊 Choi 알고리즘 성능 벤치마크 요약")
        print("=" * 60)
        
        # 기본 성능 요약
        basic = results["basic_performance"]
        print("🚀 기본 성능:")
        for workload, data in basic.items():
            status = "✅" if data["meets_target"] else "❌"
            print(f"  {status} {workload}: {data['avg_time_ms']:.2f}ms "
                  f"(목표: {data['target_ms']}ms, 비율: {data['performance_ratio']:.1%})")
        
        # 확장성 요약
        scalability = results["scalability"]
        print(f"\n📈 확장성:")
        print(f"  선형성 점수: {scalability['linearity_score']:.3f}")
        print(f"  선형 확장: {'✅' if scalability['is_linear'] else '❌'}")
        print(f"  최대 셀당 시간: {scalability['max_time_per_cell']:.2f}ms")
        
        # 메모리 요약
        memory = results["memory_usage"]
        print(f"\n🧠 메모리 효율성:")
        for test_name, data in memory.items():
            if isinstance(data, dict):
                efficient = "✅" if data["memory_efficient"] else "❌"
                print(f"  {efficient} {test_name}: {data['memory_increase_mb']:.3f}MB "
                      f"(샘플당 {data['memory_per_sample_bytes']:.2f}B)")
        
        # 단계별 성능
        phases = results["phase_analysis"]
        print(f"\n⚡ 단계별 성능:")
        print(f"  총 처리 시간: {phases['phase_times_ms']['total']:.3f}ms")
        print(f"  주요 병목: {phases['bottleneck_phase']} ({phases['phase_percentages'][phases['bottleneck_phase']]}%)")
        
        # 전체 평가
        all_targets_met = all(results["targets_met"].values())
        overall_status = "🎉 우수" if all_targets_met else "⚠️ 개선 필요"
        print(f"\n🏆 전체 평가: {overall_status}")
        
        print("=" * 60)


# =============================================================================
# 직접 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    try:
        benchmark = ChoiPerformanceBenchmark()
        results = benchmark.run_comprehensive_benchmark()
        
        # 성능 목표 달성 여부에 따른 종료 코드
        all_targets_met = all(results["targets_met"].values())
        
        if all_targets_met:
            print("\n🎉 모든 성능 목표 달성!")
            sys.exit(0)
        else:
            print("\n⚠️ 일부 성능 목표 미달성")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 벤치마크 실행 실패: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
