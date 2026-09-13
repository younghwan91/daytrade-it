#!/usr/bin/env bash
# 개장 전 점검 -- daytrade-it 의 실주문 전제를 읽기 전용으로 확인한다.
#
# daytrade-it 은 scalp-it 과 달리 무인 cron 데몬이 아니라, 사람/에이전트가
# execute_trade MCP 툴을 직접 호출하는 구조다. 그래서 이 점검은 "런처가 죽는가"
# 대신 "그 호출이 안전망을 갖춘 채로 실행 가능한가"를 본다 -- 자격증명,
# kiwoom-client 계약(부호 처리), execute_trade 의 하드캡/가격밴드/킬스위치
# 안전장치, 전체 테스트 베이스라인.
#
# 읽기 전용이다 -- 주문을 내지도, 계좌를 바꾸지도 않는다(get_account/get_quote
# 등 조회 호출은 정상적으로 나간다 -- 계좌 조회는 scalp-it 의 launcher 크론
# 점검도 동일하게 허용한다).
#
#   사용: scripts/preopen_check.sh
set -uo pipefail

FAIL=0
WARN=0
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; WARN=$((WARN + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAIL=$((FAIL + 1)); }
sec()  { printf '\n%s\n' "$*"; }

# 크론 PATH 에는 ~/.local/bin 이 없다 -- uv 가 거기 있다.
export PATH="$HOME/.local/bin:$PATH"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || { echo "레포 경로 유도 실패"; exit 1; }
cd "$REPO" || { echo "cd 실패: $REPO"; exit 1; }

echo "개장 전 점검(daytrade-it) -- $(date '+%F %T %A') · $(hostname) · $REPO"

# 크론은 빈 환경으로 돈다 -- news_signal_daemon.sh 와 같은 출처에서 키를 읽는다.
# 이미 설정돼 있으면(손으로 돌릴 때) 덮어쓰지 않는다.
if [ -z "${GPTQ_KIWOOM_APP_KEY:-}" ] && [ -f ../quant-airflow/.env ]; then
    # xargs 로 읽는다(데몬과 같은 방식) -- .env 값이 따옴표로 싸여 있어서 cut 으로
    # 읽으면 따옴표까지 키에 붙어 키움 인증이 8001 로 실패한다.
    export $(grep -E '^KIWOOM_APP_(KEY|SECRET)=' ../quant-airflow/.env | xargs)
    export GPTQ_KIWOOM_APP_KEY="$KIWOOM_APP_KEY" GPTQ_KIWOOM_APP_SECRET="$KIWOOM_APP_SECRET"
    export GPTQ_KIWOOM_IS_MOCK="${GPTQ_KIWOOM_IS_MOCK:-false}"
fi

# ───────────────────────────────────────────────────── 1. 자격증명·설정
sec "1) 자격증명과 설정"
if [ -n "${GPTQ_KIWOOM_APP_KEY:-}" ] && [ -n "${GPTQ_KIWOOM_APP_SECRET:-}" ]; then
    ok "GPTQ_KIWOOM_APP_KEY/SECRET 있음"
else
    bad "GPTQ_KIWOOM_APP_KEY/SECRET 없음 -- KiwoomBroker 생성 불가"
fi

if [ "${GPTQ_KIWOOM_IS_MOCK:-true}" = "false" ]; then
    warn "GPTQ_KIWOOM_IS_MOCK=false -- 실계좌로 향한다(의도된 것인지 재확인)"
else
    ok "GPTQ_KIWOOM_IS_MOCK 이 false 로 명시적 설정되지 않음(기본 true) -- 실계좌 연결 전 반드시 명시 설정 확인"
fi

if [ -f data/live_STOP ]; then
    bad "킬스위치 있음(data/live_STOP) -- execute_trade 가 모든 주문을 거부한다"
else
    ok "킬스위치 없음(data/live_STOP)"
fi

# ─────────────────────────────────────────────── 2. 실행 환경(venv·의존성)
sec "2) 실행 환경"
if command -v uv >/dev/null 2>&1; then
    ok "uv 실행 가능: $(command -v uv)"
else
    bad "uv 를 찾을 수 없음"
fi

if uv run python -c "import daytrade_it, kiwoom_client" >/dev/null 2>&1; then
    ok "daytrade_it/kiwoom_client import 정상"
else
    bad "daytrade_it 또는 kiwoom_client import 실패 -- venv/의존성 확인 필요"
fi

# ────────────────────────────────────── 3. execute_trade 안전장치 (정적 확인)
sec "3) execute_trade 안전장치 (코드 확인, 호출 없음)"
SAFETY_CHECK=$(uv run python - <<'PY' 2>&1
from daytrade_it.interfaces.mcp import tools

assert tools._LIVE_TRADE_QUANTITY == 1, "quantity cap is not 1"
assert tools._LIVE_TRADE_PRICE_BAND_PCT > 0, "price band not set"
assert str(tools._KILL_SWITCH_PATH) == "data/live_STOP", "kill switch path drifted"
print("OK")
PY
)
if [ "$SAFETY_CHECK" = "OK" ]; then
    ok "수량 하드캡=1 · 가격밴드 설정됨 · 킬스위치 경로 일치"
else
    bad "execute_trade 안전장치 코드 확인 실패: $SAFETY_CHECK"
fi

# ───────────────────────────────────────────────────────── 4. 테스트 베이스라인
sec "4) 테스트 베이스라인"
# 알려진 기존 실패(e2e 워크플로 테스트 31건, DB/API 서버 의존). 이보다 늘면 회귀,
# 같거나 줄면 통과 -- 누가 기존 실패를 고쳐도 점검이 거짓 FAIL 을 내지 않게 한다.
KNOWN_FAILURES=31
TEST_OUT=$(uv run pytest tests -q --no-cov 2>&1 | tail -1)
FAILED_COUNT=$(echo "$TEST_OUT" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo 0)
# 요약줄이 아예 없으면(pytest 가 안 돌았으면) "실패 0건"이 아니라 점검 불가다.
if ! echo "$TEST_OUT" | grep -qE '[0-9]+ passed'; then
    bad "pytest 요약을 읽지 못함 -- 테스트가 실행되지 않았다: ${TEST_OUT}"
elif [ "$FAILED_COUNT" -gt "$KNOWN_FAILURES" ]; then
    bad "테스트 실패 ${FAILED_COUNT}건 > 알려진 베이스라인 ${KNOWN_FAILURES}건 -- 새 회귀 의심"
elif [ "$FAILED_COUNT" -gt 0 ]; then
    warn "테스트 실패 ${FAILED_COUNT}건 (베이스라인 ${KNOWN_FAILURES}건 이하, 기존 e2e 의존성) -- 새 회귀 없음"
else
    ok "전체 테스트 통과"
fi

# ────────────────────────────────────────────────────── 5. 실계좌 조회(읽기 전용)
sec "5) 실계좌 조회 (읽기 전용 -- 주문 없음)"
if [ -n "${GPTQ_KIWOOM_APP_KEY:-}" ] && [ -n "${GPTQ_KIWOOM_APP_SECRET:-}" ]; then
    LIVE_CHECK=$(uv run python - <<'PY' 2>&1
import os
from kiwoom_client import KiwoomAPI
from daytrade_it.infrastructure.brokers.kiwoom_broker import KiwoomBroker

api = KiwoomAPI(
    app_key=os.environ["GPTQ_KIWOOM_APP_KEY"],
    app_secret=os.environ["GPTQ_KIWOOM_APP_SECRET"],
    is_mock=os.environ.get("GPTQ_KIWOOM_IS_MOCK", "true").lower() != "false",
)
broker = KiwoomBroker(api=api)
assert broker.connect(), "connect failed"
account = broker.get_account()
assert account.cash >= 0, "negative cash -- sign-stripping regression"
quote = broker.get_quote("005930")
if quote.bid is not None:
    assert quote.bid >= 0, "negative bid -- sign-stripping regression"
if quote.ask is not None:
    assert quote.ask >= 0, "negative ask -- sign-stripping regression"
positions = broker.get_positions()
broker.disconnect()
print(f"OK positions={len(positions)}")
PY
)
    if echo "$LIVE_CHECK" | grep -q "^OK"; then
        ok "connect·get_account·get_quote·get_positions 정상, 부호 회귀 없음 ($LIVE_CHECK)"
    else
        bad "실계좌 조회 실패: $LIVE_CHECK"
    fi
else
    warn "자격증명 없어 5)를 건너뜀"
fi

# ─────────────────────────────────────────── 6. 데몬이 읽는 DB·키 (데몬과 같은 출처)
sec "6) 데몬 전제 -- DB·Anthropic 키 (news_signal_daemon.sh 와 같은 출처)"
# 2026-09-13: kr-quant 가 이 호스트에서 사라지며 데몬이 옛 ../kr-quant/.env 를 읽어
# DB 비밀번호 없이 뜨는 상태였다. 이 점검은 그 경로를 안 봐서 못 잡았다.
if grep -qE '^ANTHROPIC_API_KEY=.+' ../quant-airflow/.env 2>/dev/null; then
    ok "ANTHROPIC_API_KEY 있음(../quant-airflow/.env)"
else
    bad "ANTHROPIC_API_KEY 가 ../quant-airflow/.env 에 없다 -- 뉴스 판정 불가"
fi
DB_CHECK=$(
    KR_QUANT_DB=$(grep -E '^KR_QUANT_DB=' ../quant-airflow/.env 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\"'") \
    GPTQ_DATABASE_URL=$(grep '^GPTQ_DATABASE_URL=' .env 2>/dev/null | cut -d= -f2-) \
    uv run python - <<'PY' 2>&1
import os
import psycopg2
kr = os.environ.get("KR_QUANT_DB")
app = os.environ.get("GPTQ_DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
if not kr:
    raise SystemExit("KR_QUANT_DB 가 ../quant-airflow/.env 에 없다")
if not app:
    raise SystemExit("GPTQ_DATABASE_URL 이 .env 에 없다")
with psycopg2.connect(kr, connect_timeout=5) as c, c.cursor() as cur:
    cur.execute("SELECT max(date) FROM daily_bars_adjusted WHERE code = '005930'")
    last_bar = cur.fetchone()[0]
with psycopg2.connect(app, connect_timeout=5) as c, c.cursor() as cur:
    cur.execute("SELECT pg_is_in_recovery(), count(*) FROM trading_signals")
    standby, _ = cur.fetchone()
if standby:
    raise SystemExit("gptquant DB 가 스탠바이다 -- 신호 저장 불가")
print(f"OK kr_quant 최근 일봉 {last_bar} · gptquant 쓰기 가능")
PY
)
if echo "$DB_CHECK" | grep -q "^OK"; then
    ok "${DB_CHECK#OK }"
else
    bad "DB 점검 실패: $(echo "$DB_CHECK" | tail -1)"
fi

echo
echo "──────────────────────────────────────────"
if [ "$FAIL" -eq 0 ]; then
    echo "FAIL 0 · WARN $WARN -- execute_trade 실주문 전제는 충족됐다"
else
    echo "FAIL $FAIL · WARN $WARN -- 위 FAIL 항목을 해결하기 전에는 실주문 금지"
fi
exit "$([ "$FAIL" -eq 0 ] && echo 0 || echo 1)"
