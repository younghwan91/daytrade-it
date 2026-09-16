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

# 2026-09-15: 08:55 크론이 데몬을 두 번 띄워 같은 종목을 2주씩 샀다(기아·이마트).
# 한 번에 하나만 돈다 -- 락을 못 잡으면 두 번째 실행은 아무것도 안 하고 끝난다.
mkdir -p data
exec 9>data/news_signal_daemon.lock
if ! flock -n 9; then
  echo "=== [$(date '+%F %T')] 이미 실행 중인 데몬이 있다 -- 이 실행은 종료" >&2
  exit 0
fi

export $(grep -E '^KIWOOM_APP_(KEY|SECRET)=' ../quant-airflow/.env | xargs)
export $(grep -E '^ANTHROPIC_API_KEY=' ../quant-airflow/.env | xargs)
export GPTQ_DATABASE_URL=$(grep '^GPTQ_DATABASE_URL=' .env | cut -d= -f2-)
# KR_QUANT_DB (postgresql://user:pass@host:port/db) lives in quant-airflow/.env  # pragma: allowlist secret
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

# 판정 전 사전필터: 시황 묶음·표·관련 종목 11개 초과·지수 해설 기사를 버린다.
# 97,721 건 측정 기준 쌍의 29.0% 를 줄이고 매수 후보는 0.373% 만 잃는다.
# 끄려면 이 줄을 0 으로. 한도만 바꾸려면 DAYTRADE_NEWS_PREFILTER_MAX_CODES /
# DAYTRADE_NEWS_PREFILTER_INDEX_MIN_CODES.
export DAYTRADE_NEWS_PREFILTER=1

# 대시보드 확인 주기 180초 -> 20초. 평균 탐지 지연이 90초에서 10초로 줄어든다.
#
# 주의 -- 이 값은 **측정된 우위에 근거한 것이 아니다.** 처음에는 "먼저 온
# 기사부터 사는 현행 순서가 +62bp" 를 신선도 효과로 읽고 줄였으나, 분해해 보니
# 그것은 지연이 아니라 **장중 시각** 효과였다(09:00-09:01 +166bp, 10:00 이후
# -63bp). 그리고 발행 -> 우리 결정까지의 중앙값 21.8 분 중 폴링 주기가 차지하는
# 몫은 최대 3 분(13%)이고, 나머지 ~18 분은 토스 쪽에 있어 우리가 못 줄인다.
# 탐지 지연 자체는 수익과 상관이 +0.051 로 사실상 무관했고, 지연 5 분 이하로
# 들어가 본 적이 한 번도 없다.
#
# 그래도 되돌리지 않는 이유: 해롭지 않고(안전 한도의 1/20), 우리가 통제할 수
# 있는 유일한 구간을 줄이며, 토스가 빨라지거나 다른 소스를 붙이면 그때
# 효력이 생긴다. **성과 근거로 인용하지 말 것.**
# 안전: 08:00~15:20 에 약 1,320 요청/일 (20 초당 1 회). A안 순회 설계가 안전
# 하다고 본 1 초당 1 회보다 20 배 여유가 있고, 최초 조사에서도 몇 초 간격에서
# 속도 제한이 관측되지 않았다. 180 초는 측정된 한계가 아니라 보수적 기본값이었다.
# 연속 실패 3 회면 기존 회로 차단기가 DART 전용으로 내려간다 -- 그대로 둔다.
# 되돌리려면 이 줄을 지운다(기본값 180 초).
export GPTQ_KRX_NEWS_TOSS_POLL_INTERVAL_SECONDS=20

# 재시작 루프도 15:30 까지 -- 청산 창에서 크래시해도 다시 떠야 한다.
END=$(date -d "15:30" +%s)
while [ "$(date +%s)" -lt "$END" ]; do
  REMAIN=$(( (END - $(date +%s)) / 60 ))
  [ "$REMAIN" -lt 1 ] && break
  echo "=== [$(date '+%F %T')] 뉴스 신호 데몬 시작 (남은 ${REMAIN}분)"
  # ⚠️ --confirm-real 켜짐 → 진짜 주문이 나간다.
  # 2026-09-17 부터 종이매매(--paper)로 돌린다. 실주문은 나가지 않는다.
  #
  # 이유: 실매매를 `data/live_STOP` 으로 멈춘 상태인데(뉴스 선정의 초과수익이
  # 매칭 대조군 대비 +1.6bp(se 3.4)인 반면 왕복 비용이 36.7bp), 그 스위치는
  # 뉴스 폴링까지 멈춰서 판정 기록이 하루도 안 쌓인다. 그런데 지금 이 프로젝트의
  # 병목이 바로 그 기록이다 -- 신뢰할 수 있는 아카이브 날짜가 09-14, 09-15
  # 이틀뿐이고, 나머지 5 일은 데몬이 본 것의 ~25% 만 담은 재구성 데이터다.
  # 세션의 유일했던 채택 후보도 그 재구성일이 만들어 낸 것이라 철회됐다.
  #
  # 종이매매는 같은 사전필터·같은 프롬프트로 같은 쌍을 판정하고
  # `data/news_judgments_paper/` 에 기록하되, 주문은 PaperBroker 로 간다.
  # 즉 **멈춘 날을 데이터 없는 날로 버리지 않는다.** 하루에 아카이브 하루씩.
  #
  # 실매매 재개: --paper 를 빼고 --confirm-real 을 되살린 뒤(둘은 상호배타)
  # data/live_STOP 을 지운다. 우위가 확인되기 전에는 재개하지 않는다.
  /home/young/.local/bin/uv run python scripts/news_signal_poller.py --stop-at 15:20 --exits-until 15:30 --execute --paper
  echo "=== [$(date '+%F %T')] 데몬 종료 -- $((REMAIN > 1 ? 5 : 0))초 후 재시작 시도"
  sleep 5
done
echo "=== [$(date '+%F %T')] 15:30 도달, 데몬 완전 종료"
