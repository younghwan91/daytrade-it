#!/usr/bin/env bash
# 뉴스→신호 데몬 실행기 -- scripts/news_signal_poller.py 를 크래시 재시작 루프로
# 감싼다. scalp-it 의 pair_detect_daily.sh 와 같은 관례(경로는 스크립트 위치에서
# 유도, cd 실패만 명시적으로 죽게, 나머지는 재시작 관용)를 따른다.
#
# 이 데몬은 **신호만 gptquant DB 에 적재한다 -- 주문은 내지 않는다.** 실주문은
# execute_trade MCP 툴 호출이 별도로 담당하고, 거기 하드캡(1주)·가격밴드·
# 킬스위치·중복진입 방지가 그대로 걸려 있다. 이 데몬이 죽거나 오작동해도
# 최악의 결과는 "신호가 안 쌓인다"이지 "잘못된 주문이 나간다"가 아니다.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

export $(grep -E '^KIWOOM_APP_(KEY|SECRET)=' ../quant-airflow/.env | xargs)
export $(grep -E '^ANTHROPIC_API_KEY=' ../quant-airflow/.env | xargs)
export GPTQ_DATABASE_URL=$(grep '^GPTQ_DATABASE_URL=' .env | cut -d= -f2-)
export GPTQ_QUANT_AIRFLOW_DB_HOST=192.168.45.9
export GPTQ_QUANT_AIRFLOW_DB_PORT=5433
export GPTQ_QUANT_AIRFLOW_DB_DBNAME=kr_quant
export GPTQ_QUANT_AIRFLOW_DB_USER=kr_quant
export GPTQ_QUANT_AIRFLOW_DB_PASSWORD=$(grep -E '^KR_QUANT_DB=' ../kr-quant/.env | sed -E 's#.*://[^:]+:([^@]+)@.*#\1#')
export GPTQ_KIWOOM_IS_MOCK=false

END=$(date -d "15:20" +%s)
while [ "$(date +%s)" -lt "$END" ]; do
  REMAIN=$(( (END - $(date +%s)) / 60 ))
  [ "$REMAIN" -lt 1 ] && break
  echo "=== [$(date '+%F %T')] 뉴스 신호 데몬 시작 (남은 ${REMAIN}분)"
  /home/young/.local/bin/uv run python scripts/news_signal_poller.py --stop-at 15:20
  echo "=== [$(date '+%F %T')] 데몬 종료 -- $((REMAIN > 1 ? 5 : 0))초 후 재시작 시도"
  sleep 5
done
echo "=== [$(date '+%F %T')] 15:20 도달, 데몬 완전 종료"
