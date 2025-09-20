#!/bin/bash

# 마할라노비스 분석 모니터링 시스템 중지 스크립트

set -e

echo "🛑 마할라노비스 분석 모니터링 시스템 중지 중..."

# 모니터링 디렉토리로 이동
cd "$(dirname "$0")"

# Docker Compose를 사용하여 모니터링 스택 중지
echo "📊 Prometheus, Grafana, Node Exporter, cAdvisor 중지..."
docker-compose -f docker-compose.monitoring.yml down

echo "✅ 모니터링 시스템이 성공적으로 중지되었습니다!"

echo ""
echo "💡 다음에 다시 시작하려면:"
echo "   ./start-monitoring.sh"


