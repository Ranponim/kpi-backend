"""
Statistics 비교 분석을 위한 MongoDB 데이터베이스 유틸리티

이 모듈은 MongoDB에서 효율적으로 시계열 데이터를 조회하고
통계 분석을 위한 집계 쿼리를 제공합니다.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging
import asyncio

# 로거 설정
logger = logging.getLogger(__name__)

class StatisticsDataBase:
    """Statistics 분석을 위한 데이터베이스 접근 클래스"""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        """
        MongoDB 데이터베이스 인스턴스로 초기화
        
        Args:
            database: Motor AsyncIOMotorDatabase 인스턴스
        """
        self.db = database
        self.collection = database.kpi_data  # KPI 데이터 컬렉션
        
    async def ensure_indexes(self) -> None:
        """
        통계 분석 성능을 위한 인덱스 생성
        """
        try:
            logger.info("Statistics 분석용 인덱스 생성 시작")
            
            # 복합 인덱스: timestamp + peg_name + ne + cell_id
            await self.collection.create_index([
                ("timestamp", 1),
                ("peg_name", 1),
                ("ne", 1),
                ("cell_id", 1)
            ], name="idx_stats_analysis")
            
            # 날짜 범위 쿼리용 인덱스
            await self.collection.create_index([
                ("timestamp", 1)
            ], name="idx_timestamp")
            
            # PEG별 쿼리용 인덱스
            await self.collection.create_index([
                ("peg_name", 1),
                ("timestamp", 1)
            ], name="idx_peg_timestamp")
            
            logger.info("인덱스 생성 완료")
            
        except Exception as e:
            logger.error(f"인덱스 생성 실패: {e}")
            raise
    
    async def get_period_data(
        self,
        start_date: datetime,
        end_date: datetime,
        peg_names: List[str],
        ne_filter: Optional[List[str]] = None,
        cell_id_filter: Optional[List[str]] = None,
        include_outliers: bool = True
    ) -> List[Dict[str, Any]]:
        """
        지정된 기간의 KPI 데이터를 조회
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜  
            peg_names: 조회할 PEG 이름 목록
            ne_filter: NE 필터 (선택사항)
            cell_id_filter: Cell ID 필터 (선택사항)
            include_outliers: 이상치 포함 여부
            
        Returns:
            KPI 데이터 리스트
        """
        try:
            logger.info(f"기간 데이터 조회 시작: {start_date} ~ {end_date}")
            logger.info(f"PEG: {peg_names}, NE: {ne_filter}, Cell: {cell_id_filter}")
            
            # 기본 쿼리 조건
            query = {
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                },
                "peg_name": {"$in": peg_names}
            }
            
            # 필터 조건 추가
            if ne_filter:
                query["ne"] = {"$in": ne_filter}
            
            if cell_id_filter:
                query["cell_id"] = {"$in": cell_id_filter}
            
            # 집계 파이프라인 구성
            pipeline = [
                {"$match": query},
                {
                    "$addFields": {
                        "value_numeric": {
                            "$cond": {
                                "if": {"$isNumber": "$value"},
                                "then": "$value",
                                "else": {"$toDouble": "$value"}
                            }
                        }
                    }
                }
            ]
            
            # 이상치 제거 (선택사항)
            if not include_outliers:
                # IQR 방법으로 이상치 제거
                pipeline.extend([
                    {
                        "$group": {
                            "_id": "$peg_name",
                            "values": {"$push": "$value_numeric"},
                            "docs": {"$push": "$$ROOT"}
                        }
                    },
                    {
                        "$addFields": {
                            "sorted_values": {"$sortArray": {"input": "$values", "sortBy": 1}},
                            "count": {"$size": "$values"}
                        }
                    },
                    {
                        "$addFields": {
                            "q1_index": {"$floor": {"$multiply": ["$count", 0.25]}},
                            "q3_index": {"$floor": {"$multiply": ["$count", 0.75]}}
                        }
                    },
                    {
                        "$addFields": {
                            "q1": {"$arrayElemAt": ["$sorted_values", "$q1_index"]},
                            "q3": {"$arrayElemAt": ["$sorted_values", "$q3_index"]}
                        }
                    },
                    {
                        "$addFields": {
                            "iqr": {"$subtract": ["$q3", "$q1"]},
                            "lower_bound": {"$subtract": ["$q1", {"$multiply": [1.5, {"$subtract": ["$q3", "$q1"]}]}]},
                            "upper_bound": {"$add": ["$q3", {"$multiply": [1.5, {"$subtract": ["$q3", "$q1"]}]}]}
                        }
                    },
                    {
                        "$unwind": "$docs"
                    },
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$gte": ["$docs.value_numeric", "$lower_bound"]},
                                    {"$lte": ["$docs.value_numeric", "$upper_bound"]}
                                ]
                            }
                        }
                    },
                    {
                        "$replaceRoot": {"newRoot": "$docs"}
                    }
                ])
            
            # 정렬 추가
            pipeline.append({"$sort": {"timestamp": 1, "peg_name": 1}})
            
            # 쿼리 실행
            cursor = self.collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            
            logger.info(f"조회된 데이터 포인트: {len(results)}개")
            return results
            
        except Exception as e:
            logger.error(f"기간 데이터 조회 실패: {e}")
            raise
    
    async def get_aggregated_statistics(
        self,
        start_date: datetime,
        end_date: datetime,
        peg_names: List[str],
        ne_filter: Optional[List[str]] = None,
        cell_id_filter: Optional[List[str]] = None,
        include_outliers: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        기간별 PEG 통계를 집계하여 반환
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            peg_names: 조회할 PEG 이름 목록
            ne_filter: NE 필터 (선택사항)
            cell_id_filter: Cell ID 필터 (선택사항)
            include_outliers: 이상치 포함 여부
            
        Returns:
            PEG별 통계 딕셔너리
        """
        try:
            logger.info(f"집계 통계 계산 시작: {start_date} ~ {end_date}")
            
            # 기본 쿼리 조건
            query = {
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                },
                "peg_name": {"$in": peg_names}
            }
            
            # 필터 조건 추가
            if ne_filter:
                query["ne"] = {"$in": ne_filter}
            
            if cell_id_filter:
                query["cell_id"] = {"$in": cell_id_filter}
            
            # 집계 파이프라인
            pipeline = [
                {"$match": query},
                {
                    "$addFields": {
                        "value_numeric": {
                            "$cond": {
                                "if": {"$isNumber": "$value"},
                                "then": "$value",
                                "else": {"$toDouble": "$value"}
                            }
                        }
                    }
                },
                {
                    "$group": {
                        "_id": "$peg_name",
                        "count": {"$sum": 1},
                        "sum": {"$sum": "$value_numeric"},
                        "avg": {"$avg": "$value_numeric"},
                        "min": {"$min": "$value_numeric"},
                        "max": {"$max": "$value_numeric"},
                        "std_dev": {"$stdDevPop": "$value_numeric"},
                        "values": {"$push": "$value_numeric"}
                    }
                },
                {
                    "$addFields": {
                        "sorted_values": {"$sortArray": {"input": "$values", "sortBy": 1}},
                        "median_index": {"$floor": {"$divide": [{"$size": "$values"}, 2]}},
                        "q1_index": {"$floor": {"$multiply": [{"$size": "$values"}, 0.25]}},
                        "q3_index": {"$floor": {"$multiply": [{"$size": "$values"}, 0.75]}}
                    }
                },
                {
                    "$addFields": {
                        "median": {"$arrayElemAt": ["$sorted_values", "$median_index"]},
                        "percentile_25": {"$arrayElemAt": ["$sorted_values", "$q1_index"]},
                        "percentile_75": {"$arrayElemAt": ["$sorted_values", "$q3_index"]}
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "count": 1,
                        "mean": "$avg",
                        "std": "$std_dev", 
                        "min": 1,
                        "max": 1,
                        "median": 1,
                        "percentile_25": 1,
                        "percentile_75": 1
                    }
                }
            ]
            
            # 쿼리 실행
            cursor = self.collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            
            # 결과를 딕셔너리로 변환
            stats_dict = {}
            for result in results:
                peg_name = result.pop('_id')
                stats_dict[peg_name] = result
            
            logger.info(f"집계된 PEG 수: {len(stats_dict)}")
            return stats_dict
            
        except Exception as e:
            logger.error(f"집계 통계 계산 실패: {e}")
            raise
    
    async def get_comparison_data(
        self,
        period1: Tuple[datetime, datetime],
        period2: Tuple[datetime, datetime], 
        peg_names: List[str],
        ne_filter: Optional[List[str]] = None,
        cell_id_filter: Optional[List[str]] = None,
        include_outliers: bool = True
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        두 기간의 비교 데이터를 동시에 조회
        
        Args:
            period1: 첫 번째 기간 (start_date, end_date)
            period2: 두 번째 기간 (start_date, end_date)
            peg_names: 조회할 PEG 이름 목록
            ne_filter: NE 필터 (선택사항)
            cell_id_filter: Cell ID 필터 (선택사항)
            include_outliers: 이상치 포함 여부
            
        Returns:
            (period1_stats, period2_stats) 튜플
        """
        try:
            logger.info("두 기간 비교 데이터 조회 시작")
            
            # 두 기간의 데이터를 병렬로 조회
            period1_task = asyncio.create_task(
                self.get_aggregated_statistics(
                    period1[0], period1[1], peg_names, 
                    ne_filter, cell_id_filter, include_outliers
                )
            )
            
            period2_task = asyncio.create_task(
                self.get_aggregated_statistics(
                    period2[0], period2[1], peg_names,
                    ne_filter, cell_id_filter, include_outliers
                )
            )
            
            # 병렬 실행 완료 대기
            period1_stats, period2_stats = await asyncio.gather(
                period1_task, period2_task
            )
            
            logger.info("두 기간 비교 데이터 조회 완료")
            return period1_stats, period2_stats
            
        except Exception as e:
            logger.error(f"비교 데이터 조회 실패: {e}")
            raise
    
    async def validate_data_availability(
        self,
        start_date: datetime,
        end_date: datetime,
        peg_names: List[str],
        ne_filter: Optional[List[str]] = None,
        cell_id_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        지정된 기간과 조건에 대한 데이터 가용성 검증
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            peg_names: 조회할 PEG 이름 목록
            ne_filter: NE 필터 (선택사항)
            cell_id_filter: Cell ID 필터 (선택사항)
            
        Returns:
            데이터 가용성 정보 딕셔너리
        """
        try:
            logger.info(f"데이터 가용성 검증: {start_date} ~ {end_date}")
            
            # 기본 쿼리 조건
            query = {
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                },
                "peg_name": {"$in": peg_names}
            }
            
            # 필터 조건 추가
            if ne_filter:
                query["ne"] = {"$in": ne_filter}
            
            if cell_id_filter:
                query["cell_id"] = {"$in": cell_id_filter}
            
            # 집계 파이프라인
            pipeline = [
                {"$match": query},
                {
                    "$group": {
                        "_id": {
                            "peg_name": "$peg_name",
                            "ne": "$ne",
                            "cell_id": "$cell_id"
                        },
                        "count": {"$sum": 1},
                        "min_timestamp": {"$min": "$timestamp"},
                        "max_timestamp": {"$max": "$timestamp"}
                    }
                },
                {
                    "$group": {
                        "_id": "$_id.peg_name",
                        "total_count": {"$sum": "$count"},
                        "entity_count": {"$sum": 1},
                        "min_timestamp": {"$min": "$min_timestamp"},
                        "max_timestamp": {"$max": "$max_timestamp"}
                    }
                }
            ]
            
            cursor = self.collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            
            # 결과 정리
            availability_info = {
                "period": {
                    "start": start_date,
                    "end": end_date
                },
                "peg_availability": {},
                "total_data_points": 0,
                "available_pegs": [],
                "missing_pegs": []
            }
            
            result_pegs = set()
            for result in results:
                peg_name = result['_id']
                result_pegs.add(peg_name)
                availability_info["peg_availability"][peg_name] = {
                    "data_points": result["total_count"],
                    "entities": result["entity_count"],
                    "data_range": {
                        "start": result["min_timestamp"],
                        "end": result["max_timestamp"]
                    }
                }
                availability_info["total_data_points"] += result["total_count"]
            
            # 사용 가능한/누락된 PEG 목록
            availability_info["available_pegs"] = list(result_pegs)
            availability_info["missing_pegs"] = [
                peg for peg in peg_names if peg not in result_pegs
            ]
            
            logger.info(f"데이터 가용성 검증 완료 - 총 {availability_info['total_data_points']}개 포인트")
            return availability_info
            
        except Exception as e:
            logger.error(f"데이터 가용성 검증 실패: {e}")
            raise

async def create_sample_data(db: AsyncIOMotorDatabase, count: int = 1000) -> None:
    """
    테스트용 샘플 데이터 생성 (개발/테스트 환경용)
    
    Args:
        db: MongoDB 데이터베이스 인스턴스
        count: 생성할 데이터 포인트 수
    """
    import random
    from datetime import timedelta
    
    try:
        logger.info(f"샘플 데이터 {count}개 생성 시작")
        
        collection = db.kpi_data
        base_date = datetime(2025, 8, 1)
        
        # PEG별 기본값 설정
        peg_configs = {
            'availability': {'base': 99.5, 'std': 0.3, 'min': 95.0, 'max': 100.0},
            'rrc': {'base': 98.8, 'std': 0.5, 'min': 90.0, 'max': 100.0},
            'erab': {'base': 99.2, 'std': 0.4, 'min': 90.0, 'max': 100.0},
            'sar': {'base': 95.5, 'std': 2.0, 'min': 80.0, 'max': 100.0},
            'mobility_intra': {'base': 97.8, 'std': 1.2, 'min': 85.0, 'max': 100.0}
        }
        
        ne_list = ['nvgnb#10000', 'nvgnb#20000', 'nvgnb#30000']
        cell_ids = ['2010', '2011', '2012', '2013']
        
        documents = []
        
        for i in range(count):
            # 시간 생성 (최근 14일 내 랜덤)
            timestamp = base_date + timedelta(
                days=random.randint(0, 13),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # PEG 선택
            peg_name = random.choice(list(peg_configs.keys()))
            config = peg_configs[peg_name]
            
            # 값 생성 (정규분포 + 경계값 제한)
            value = random.gauss(config['base'], config['std'])
            value = max(config['min'], min(config['max'], value))
            
            # 문서 생성
            doc = {
                'timestamp': timestamp,
                'peg_name': peg_name,
                'value': round(value, 4),
                'ne': random.choice(ne_list),
                'cell_id': random.choice(cell_ids),
                'unit': '%' if peg_name in ['availability', 'rrc', 'erab'] else 'score'
            }
            
            documents.append(doc)
        
        # 배치 삽입
        await collection.insert_many(documents)
        
        logger.info(f"샘플 데이터 생성 완료: {count}개")
        
    except Exception as e:
        logger.error(f"샘플 데이터 생성 실패: {e}")
        raise

if __name__ == "__main__":
    # 테스트 코드
    import asyncio
    from motor.motor_asyncio import AsyncIOMotorClient
    
    async def test_statistics_db():
        """StatisticsDataBase 클래스 테스트"""
        client = AsyncIOMotorClient("mongodb://mongo:27017")
        db = client.test_statistics
        
        stats_db = StatisticsDataBase(db)
        
        # 인덱스 생성
        await stats_db.ensure_indexes()
        
        # 샘플 데이터 생성
        await create_sample_data(db, 500)
        
        # 데이터 가용성 검증
        availability = await stats_db.validate_data_availability(
            datetime(2025, 8, 1),
            datetime(2025, 8, 7),
            ['availability', 'rrc', 'erab']
        )
        
        print(f"✅ 데이터 가용성: {availability}")
        
        # 클린업
        await client.close()
    
    if __name__ == "__main__":
        asyncio.run(test_statistics_db())

