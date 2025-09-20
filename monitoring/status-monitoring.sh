#!/bin/bash

# 마할라노비스 분석 모니터링 시스템 상태 확인 스크립트

echo "🔍 마할라노비스 분석 모니터링 시스템 상태 확인"

# 모니터링 디렉토리로 이동
cd "$(dirname "$0")"

echo ""
echo "🐳 Docker 컨테이너 상태:"
docker-compose -f docker-compose.monitoring.yml ps

echo ""
echo "🌐 서비스 상태:"

# Prometheus 상태 확인
echo -n "Prometheus: "
if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo "✅ 실행 중 (http://localhost:9090)"
else
    echo "❌ 중지됨"
fi

# Grafana 상태 확인
echo -n "Grafana: "
if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✅ 실행 중 (http://localhost:3000)"
else
    echo "❌ 중지됨"
fi

# Node Exporter 상태 확인
echo -n "Node Exporter: "
if curl -s http://localhost:9100/metrics > /dev/null 2>&1; then
    echo "✅ 실행 중 (http://localhost:9100)"
else
    echo "❌ 중지됨"
fi

# cAdvisor 상태 확인
echo -n "cAdvisor: "
if curl -s http://localhost:8080/metrics > /dev/null 2>&1; then
    echo "✅ 실행 중 (http://localhost:8080)"
else
    echo "❌ 중지됨"
fi

# FastAPI 메트릭 확인
echo -n "FastAPI Metrics: "
if curl -s http://localhost:8000/metrics > /dev/null 2>&1; then
    echo "✅ 실행 중 (http://localhost:8000/metrics)"
else
    echo "❌ 중지됨"
fi

echo ""
echo "📊 메트릭 샘플 확인:"

# Prometheus 메트릭 수 확인
echo -n "Prometheus 메트릭 수: "
METRICS_COUNT=$(curl -s http://localhost:9090/api/v1/query?query=count%28up%29 | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
echo "$METRICS_COUNT"

# FastAPI 메트릭 샘플
echo "FastAPI 메트릭 샘플:"
curl -s http://localhost:8000/metrics | head -10

echo ""
echo "💡 명령어:"
echo "   시작: ./start-monitoring.sh"
echo "   중지: ./stop-monitoring.sh"
echo "   상태: ./status-monitoring.sh"


