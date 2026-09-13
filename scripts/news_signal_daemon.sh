#!/usr/bin/env bash
# 뉴스→신호 데몬 실행기 -- scripts/news_signal_poller.py 를 크래시 재시작 루프로
# 감싼다. scalp-it 의 pair_detect_daily.sh 와 같은 관례(경로는 스크립트 위치에서
# 유도, cd 실패만 명시적으로 죽게, 나머지는 재시작 관용)를 따른다.
#
# ⚠️ 2026-09-14 부터 **실주문 1주**가 나간다(--execute --confirm-real, 사용자 지시).
#    되돌리려면 아래 실행줄에서 --confirm-real 을 지운다 -- 그 즉시 dry-run(원장만 기록).
#    --execute 까지 지우면 신호만 적재한다.
# 안전장치(application/trading/auto_trader.py, interfaces/mcp/tools.py::execute_trade):
#   진입 1주·지정가(매도1호가)·±3% 가격밴드·종목당 하루 1회·일일 10회·동시 2종목·
#   1주 명목 40만원 이하·진입 09:00:30~14:30·미체결 5분 뒤 취소·원래 보유 종목 매수 금지.
#   청산 15:00 부터 봇이 산 1주만 지정가로 매도 -- 1분마다 재시도, 매수1호가에서 0.5%씩 낮춰
#   최대 2% 아래까지. 뉴스 폴링은 15:20 에 멈추지만 청산 관리는 동시호가 끝(15:30)까지 돈다.
#   15:30 에 남은 보유는 원장에 overnight_positions 로 기록된다.
#   킬스위치 data/live_STOP: 신규 신호·진입만 멈추고 청산은 계속한다.
#   원장 data/auto_trader/YYYY-MM-DD.jsonl -- 재시작해도 이어받는다.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

export $(grep -E '^KIWOOM_APP_(KEY|SECRET)=' ../quant-airflow/.env | xargs)
export $(grep -E '^ANTHROPIC_API_KEY=' ../quant-airflow/.env | xargs)
export GPTQ_DATABASE_URL=$(grep '^GPTQ_DATABASE_URL=' .env | cut -d= -f2-)
# KR_QUANT_DB (postgresql://user:pass@host:port/db) lives in quant-airflow/.env
# since 2026-09-13, when kr-quant (now swing-it) was removed from this host --
# the old ../kr-quant/.env path left the daemon with no DB password. All five
# fields come from that one DSN so a future move only changes one line.
KR_QUANT_DB=$(grep -E '^KR_QUANT_DB=' ../quant-airflow/.env | head -1 | cut -d= -f2- | tr -d "\"'")
if [ -z "$KR_QUANT_DB" ]; then
  echo "=== [$(date '+%F %T')] KR_QUANT_DB 가 ../quant-airflow/.env 에 없다 -- 데몬 시작 안 함" >&2
  exit 1
fi
_dsn=${KR_QUANT_DB#*://}
export GPTQ_QUANT_AIRFLOW_DB_USER=${_dsn%%:*}
_rest=${_dsn#*:}
export GPTQ_QUANT_AIRFLOW_DB_PASSWORD=${_rest%@*}
_hostpart=${_rest##*@}
export GPTQ_QUANT_AIRFLOW_DB_HOST=${_hostpart%%:*}
_portdb=${_hostpart#*:}
export GPTQ_QUANT_AIRFLOW_DB_PORT=${_portdb%%/*}
export GPTQ_QUANT_AIRFLOW_DB_DBNAME=${_portdb#*/}
unset _dsn _rest _hostpart _portdb
export GPTQ_KIWOOM_IS_MOCK=false

# 재시작 루프도 15:30 까지 -- 청산 창에서 크래시해도 다시 떠야 한다.
END=$(date -d "15:30" +%s)
while [ "$(date +%s)" -lt "$END" ]; do
  REMAIN=$(( (END - $(date +%s)) / 60 ))
  [ "$REMAIN" -lt 1 ] && break
  echo "=== [$(date '+%F %T')] 뉴스 신호 데몬 시작 (남은 ${REMAIN}분)"
  # ⚠️ --confirm-real 켜짐 → 진짜 주문이 나간다.
  /home/young/.local/bin/uv run python scripts/news_signal_poller.py --stop-at 15:20 --exits-until 15:30 --execute --confirm-real
  echo "=== [$(date '+%F %T')] 데몬 종료 -- $((REMAIN > 1 ? 5 : 0))초 후 재시작 시도"
  sleep 5
done
echo "=== [$(date '+%F %T')] 15:30 도달, 데몬 완전 종료"
