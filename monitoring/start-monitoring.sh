#!/bin/bash

# 마할라노비스 분석 모니터링 시스템 시작 스크립트

set -e

echo "🚀 마할라노비스 분석 모니터링 시스템 시작 중..."

# 모니터링 디렉토리로 이동
cd "$(dirname "$0")"

# Docker Compose를 사용하여 모니터링 스택 시작
echo "📊 Prometheus, Grafana, Node Exporter, cAdvisor 시작..."
docker-compose -f docker-compose.monitoring.yml up -d

# 서비스 시작 대기
echo "⏳ 서비스 시작 대기 중..."
sleep 30

# 서비스 상태 확인
echo "🔍 서비스 상태 확인..."

# Prometheus 상태 확인
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus: http://localhost:9090"
else
    echo "❌ Prometheus 시작 실패"
fi

# Grafana 상태 확인
if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✅ Grafana: http://localhost:3000 (admin/admin123)"
else
    echo "❌ Grafana 시작 실패"
fi

# Node Exporter 상태 확인
if curl -s http://localhost:9100/metrics > /dev/null; then
    echo "✅ Node Exporter: http://localhost:9100"
else
    echo "❌ Node Exporter 시작 실패"
fi

# cAdvisor 상태 확인
if curl -s http://localhost:8080/metrics > /dev/null; then
    echo "✅ cAdvisor: http://localhost:8080"
else
    echo "❌ cAdvisor 시작 실패"
fi

echo ""
echo "🎯 모니터링 시스템이 성공적으로 시작되었습니다!"
echo ""
echo "📊 대시보드 접근:"
echo "   - Prometheus: http://localhost:9090"
echo "   - Grafana: http://localhost:3000 (admin/admin123)"
echo "   - API 메트릭: http://localhost:8000/metrics"
echo ""
echo "📈 마할라노비스 분석 대시보드:"
echo "   1. Grafana에 로그인 (admin/admin123)"
echo "   2. 좌측 메뉴에서 'Dashboards' 클릭"
echo "   3. 'Mahalanobis Analysis API Dashboard' 선택"
echo ""
echo "🛑 모니터링 시스템 중지:"
echo "   docker-compose -f docker-compose.monitoring.yml down"


