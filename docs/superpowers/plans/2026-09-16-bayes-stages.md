# 베이즈 최적화 1~4단계 (선정·매수 타이밍·손절 → 수익실현) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 종목 선정·매수 타이밍·손절(핵심)과 수익실현 규칙을 운영 코드 설정값으로 구현하고, 0단계 리플레이 엔진으로 베이즈 탐색·판정해 채택 설정을 낸다.

**Architecture:** 모든 후보 규칙은 운영 코드(`AutoTrader`/`TradingLimits`, `news_entry`/`SelectionConfig`)의 설정값이고 기본값은 현행 동작과 같다. 운영 폴러와 리플레이는 같은 정책 파일 로더(`config/news_policy.json`, 없으면 기본값)를 쓴다. 최적화 스크립트는 simnode에서 거래일 하나씩 빼는 검증(LODO)으로 Optuna TPE 탐색을 병렬 실행하고, 판정 스크립트는 numpy 격자 사후분포(Student-t)로 채택 조건을 계산한다.

**Tech Stack:** Python 3.13, uv, pytest, optuna(이미 의존성), numpy, multiprocessing, 0단계 리플레이 엔진.

**Spec:** `docs/superpowers/specs/2026-09-15-bayes-pipeline-optimization-design.md` (2단계 = 선정+타이밍+손절, 3단계 = 수익실현, 후보 구현 원칙)

## Global Constraints

- 백테스트 = 실매매: 후보는 운영 코드 설정값으로만 구현한다. 리플레이 전용 규칙 구현 금지. 기본값(정책 파일 없음)에서 운영 동작은 현행과 완전히 같아야 하고, 0단계 동일성 게이트(09-14·09-15 PASS, other 0, errors 0)와 기준선(7일 합계 +39,915원, 날짜별 동일)이 그대로 재현돼야 한다.
- 매매 시점에 볼 수 없는 정보 금지: 규칙 판정은 완성된 1분봉(`ts < 현재 분`)과 그 시점 호가만 쓴다. 1분 간격(`exit_check_every`)으로 판정한다.
- 손절·수익실현 매도는 기존 `_sell`(매수호가 지정가, 재시도 할인) 경로를 쓴다.
- 운영 반영(정책 파일을 운영 체크아웃에 두는 것)은 이 계획 범위 밖이다. 이 계획은 채택 후보 정책 파일과 판정 문서까지 만든다.
- 평가·최적화 스크립트는 `require_backtest_host()` (simnode). 단위 테스트는 네트워크·DB 없이.
- simnode: `young@192.168.45.9`, 워크트리 `~/git/daytrade-it-bayes`, DB 환경변수 `export $(grep -E "^KR_QUANT_DB=" ../quant-airflow/.env | xargs)`, DART `--dart data/eval/dart_trader_20260915.db`.
- 기준선 7일 리플레이는 simnode에서 약 2.6초(2026-09-15 측정).
- 비용: 매도세 0.20% + 수수료 편도 0.015% (`replay_baseline.trade_pnl`).
- 커밋 메시지 끝: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

## File Structure

| 파일 | 책임 |
|---|---|
| `src/daytrade_it/infrastructure/brokers/sim_broker.py` (수정) | 가격제한 기준 `limit_base`(KRX 종가) 분리 (R18) |
| `scripts/eval/replay_parity.py` (수정) | `check_session_hours` 죽은 코드 수정 (R19) |
| `src/daytrade_it/application/trading/auto_trader.py` (수정) | 손절·수익실현 규칙, 매수 타이밍(지연·눌림·VWAP 회복·지정가 틱) |
| `src/daytrade_it/application/signals/news_entry.py` (수정) | `SelectionConfig`: 점수 하한, subject, 제외 목록, 움직임 상·하한, 세션 |
| `src/daytrade_it/application/trading/policy.py` (신규) | `NewsPolicy(limits, selection)`, JSON 로더/저장 |
| `scripts/news_signal_poller.py` (수정) | 정책 파일 로드 → `AutoTrader(limits=…)`, `build_entry_candidate(selection=…)`, `rank_batch(key=…)` |
| `src/daytrade_it/application/replay/day_replay.py` (수정) | `DayConfig.policy: NewsPolicy` 사용 |
| `scripts/eval/replay_common.py` (신규) | 일자 데이터 준비 캐시(피클), 정책으로 하루 리플레이 → 일일 순손익 |
| `scripts/eval/optimize_policy.py` (신규) | 탐색 공간, LODO × TPE 병렬 탐색, 결과 JSON |
| `scripts/eval/bayes_judge.py` (신규) | 격자 사후분포, 채택 조건 판정 |
| `scripts/eval/build_replay_events.py` (수정) | `--features v4` (아카이브 v4_shadow 판정으로 이벤트) |
| `docs/research/2026-09-16-bayes-pipeline-eval.md` (수정) | 1~4단계 결과 |
| `config/news_policy.candidate.json` (신규, 산출물) | 채택 후보 정책 (운영 미적용) |

---

### Task 1: 브랜치 최신화 + 보류 항목 R18·R19

**Files:**
- Modify: `src/daytrade_it/infrastructure/brokers/sim_broker.py`, `scripts/eval/replay_parity.py`, `src/daytrade_it/application/replay/prev_close.py` (필요 시), 관련 테스트

**Interfaces:**
- Produces: `SimBroker(bars, prev_close, clock, cash=Decimal("1100000"), limit_base: dict[str, Decimal] | None = None)` — 가격제한(상·하한가 잠김) 판정 기준. `None`이면 `prev_close` 사용(기존 동작).

- [ ] **Step 1: main 반영**

```bash
cd /home/young/git/daytrade-it-bayes
git fetch origin
git rebase origin/main
uv sync
uv run pytest tests/unit -q
```
Expected: 충돌 없이 rebase(main의 `534d899` flock, `6a5cffc` krx-news-client 0.4). 충돌 시 main 쪽 변경을 살리고 브랜치 변경을 다시 적용. 전체 단위 테스트 PASS. `TOSS_NEWS_NAMESPACE` import 경로가 바뀌었으면 `build_replay_events.py`/테스트를 맞춘다.

- [ ] **Step 2: R19 실패 테스트** — `tests/unit/scripts/test_replay_parity.py`에 `load_session_bars`가 08:00 이전 봉이 섞인 입력에서 `check_session_hours`를 실제로 호출해 예외를 내는지 확인하는 테스트(DB 함수는 monkeypatch한 커서로 대체). 현재 `return out` 뒤에 붙어 있어 FAIL.

- [ ] **Step 3: R19 수정** — `replay_parity.py`의 `load_session_bars`에서 첫 `return out` 삭제, `check_session_hours(out)` 후 `return out`. 테스트 PASS.

- [ ] **Step 4: R18 실패 테스트** — `tests/unit/infrastructure/test_sim_broker.py`:

```python
def test_limit_lock_uses_limit_base_not_prev_close() -> None:
    # NXT close 2900 but KRX close 2705: upper limit is ~3515 (2705 * 1.3), not 3770.
    bars = {
        "000009": [_bar("09:00", 3515, 3515, 3515, 3515), _bar("09:01", 3515, 3515, 3515, 3515)]
    }
    c = _Clock()
    b = SimBroker(bars, {"000009": Decimal(2900)}, c, limit_base={"000009": Decimal(2705)})
    assert b.get_quote("000009").ask == 0
    b2 = SimBroker(bars, {"000009": Decimal(2900)}, c)
    assert b2.get_quote("000009").ask == 3515
```

- [ ] **Step 5: R18 구현** — `SimBroker.__init__`에 `limit_base` 추가, `_open_locked`(및 완성봉 잠김 판정)의 기준을 `self._limit_base.get(t) or self._prev_close.get(t)`로. `replay_parity.prepare_day`가 고정 파일의 `krx_close`를 `limit_base`로 넘긴다. 테스트 PASS.

- [ ] **Step 6: simnode 재확인과 커밋**

```bash
git add -A src scripts tests && git commit -m "fix(replay): price limits from the KRX close; run the session-hours guard (R18, R19)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push --force-with-lease origin feat/bayes-pipeline-opt
ssh young@192.168.45.9 'cd ~/git/daytrade-it-bayes && git fetch origin && git reset --hard origin/feat/bayes-pipeline-opt && ~/.local/bin/uv sync -q && export $(grep -E "^KR_QUANT_DB=" ../quant-airflow/.env | xargs) && ~/.local/bin/uv run python scripts/eval/replay_parity.py --days 2026-09-14,2026-09-15 --dart data/eval/dart_trader_20260915.db | tail -20 && ~/.local/bin/uv run python scripts/eval/replay_baseline.py --dart data/eval/dart_trader_20260915.db | tail -3'
```
Expected: 게이트 두 날짜 PASS(other 0, errors 0), 기준선 합계 +39,915원 날짜별 동일. 달라지면 원인을 보고서에 적고 멈춘다(`--force-with-lease` push는 이 피처 브랜치에 한함; simnode 워크트리에 커밋 안 된 변경이 있으면 reset 전에 멈추고 보고).

---

### Task 2: 손절·수익실현 규칙 (운영 `AutoTrader`)

**Files:**
- Modify: `src/daytrade_it/application/trading/auto_trader.py`
- Test: `tests/unit/application/trading/test_auto_trader_exit_rules.py` (신규)

**Interfaces:**
- Produces (`TradingLimits` 신규 필드, 전부 기본 비활성):
  - 손절: `stop_loss_pct: Decimal | None = None`, `stop_vwap_gap: Decimal | None = None`, `stop_time: dt.timedelta | None = None`, `stop_time_min_gain: Decimal = Decimal("0")`, `stop_orb: dt.timedelta | None = None`
  - 수익실현: `take_profit_pct: Decimal | None = None`, `trail_after_gain: Decimal | None = None`, `trail_gap: Decimal = Decimal("0.01")`
  - 순수 함수 `exit_rule_hit(bars: list[MinuteBar], entry: Decimal, entered_at: dt.datetime, now: dt.datetime, lim: TradingLimits) -> str | None` — 발동한 규칙 이름(`"stop_loss" | "stop_vwap" | "stop_time" | "stop_orb" | "take_profit" | "trailing" | "vwap_break"`) 또는 None

규칙(모두 `ts < 현재 분`인 09:00 이후 완성봉만, `c` = 마지막 완성봉 종가, `vwap` = 그 봉들의 거래량가중 종가, `peak` = 진입 분 이후 완성봉 고가 최대):
1. `stop_loss`: `c <= entry × (1 − stop_loss_pct)`
2. `stop_vwap`: `c < entry` 이고 `c < vwap × (1 − stop_vwap_gap)`
3. `stop_time`: `now − entered_at >= stop_time` 이고 `c < entry × (1 + stop_time_min_gain)`
4. `stop_orb`: 09:00부터 `stop_orb` 동안의 완성봉 저가 최소를 `orb_low`로 두고, 그 구간이 끝난 뒤 `c < orb_low`
5. `take_profit`: `c >= entry × (1 + take_profit_pct)`
6. `trailing`: `peak >= entry × (1 + trail_after_gain)` 이고 `c <= peak × (1 − trail_gap)`
7. `vwap_break`: 기존 `vwap_exit_hit` 규칙(기존 필드 그대로)
검사 순서는 위 번호 순(손절 먼저). `entered_at` = 원장 `entry_submitted.ts`, `entry` = 원장 `limit`(기존 VWAP 규칙과 같은 기준).

- [ ] **Step 1: 실패 테스트 작성** — 합성 분봉으로 각 규칙 1개씩 발동/비발동 경계를 검사하는 순수 함수 테스트 7개 + 기본 `TradingLimits()`에서 `exit_rule_hit`가 기존 `vwap_exit_hit`와 같은 결과만 내는 테스트 1개. 예:

```python
def test_stop_loss_fires_at_threshold_on_completed_bar_only() -> None:
    lim = TradingLimits(stop_loss_pct=Decimal("0.02"), vwap_exit_gap=None)
    bars = [_bar("09:00", 100, 100), _bar("09:01", 98, 98)]
    entered = _t("09:00:30")
    assert (
        exit_rule_hit(bars, Decimal(100), entered, _t("09:01:59"), lim) is None
    )  # 09:01 not complete
    assert exit_rule_hit(bars, Decimal(100), entered, _t("09:02:00"), lim) == "stop_loss"
```
(`_bar(hm, close, low_or_high...)` 헬퍼는 테스트 파일 안에 정의: open=high=low=close, volume 100.)

- [ ] **Step 2: 실패 확인** — `uv run pytest tests/unit/application/trading/test_auto_trader_exit_rules.py -v` → ImportError.

- [ ] **Step 3: 구현**
  - `TradingLimits`에 위 필드 추가(각 필드 위에 한 줄 주석: 무엇이고 기본 None이면 꺼짐).
  - `exit_rule_hit` 순수 함수 추가(위 규칙 그대로).
  - `_early_exits`: 첫 줄의 `if lim.vwap_exit_gap is None ...: return` 을 "어떤 청산 규칙도 켜져 있지 않으면 return"으로 바꾼다(`_any_exit_rule(lim)`). 기존 VWAP 판정 호출을 `exit_rule_hit`로 바꾸고, 발동 시 `exit_signal` 기록의 `reason`에 규칙 이름을 넣는다. 추가매수 판정 분기는 그대로 둔다(발동하지 않았을 때만).
  - `entered_at`는 `self._state.entries[ticker]["ts"]`.

- [ ] **Step 4: 테스트 통과** — 신규 테스트 + `uv run pytest tests/unit -q` 전체.

- [ ] **Step 5: 커밋**

```bash
git add src/daytrade_it/application/trading/auto_trader.py tests/unit/application/trading/test_auto_trader_exit_rules.py
git commit -m "feat(trading): stop-loss and profit-taking exit rules as TradingLimits settings (off by default)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: 매수 타이밍 규칙 (운영 `AutoTrader`)

**Files:**
- Modify: `src/daytrade_it/application/trading/auto_trader.py`
- Test: `tests/unit/application/trading/test_auto_trader_entry_timing.py` (신규)

**Interfaces:**
- Produces (`TradingLimits` 신규 필드, 기본 비활성):
  - `entry_delay: dt.timedelta | None = None` — 신호 후 이 시간이 지난 뒤 진입 시도
  - `entry_pullback: Decimal | None = None` — 신호 시점 가격 대비 이만큼 내려온 완성봉 종가가 나오면 진입
  - `entry_vwap_reclaim: bool = False` — 완성봉 종가가 VWAP 아래였다가 다시 위로 올라온 봉에서 진입
  - `entry_max_wait: dt.timedelta = dt.timedelta(minutes=30)` — 조건을 기다리는 최대 시간(넘으면 포기, `entry_expired` 기록)
  - `entry_limit_ticks: int = 0` — 지정가 = 매도호가 − 틱 × 이 값(0이면 현행)
- 순수 함수 `entry_ready(bars, ref_price, signaled_at, now, lim) -> bool`

동작:
- 타이밍 규칙이 하나도 켜져 있지 않으면 `enter()`는 현행과 완전히 같다.
- 켜져 있으면 `enter()`는 기존의 즉시 거부 검사(BUY, 진입 창, 이미 진입, 일일 한도, 사용자 보유)만 한 뒤 `entry_deferred`를 원장에 기록(ticker, signal_id, ref_price = 그 시점 `get_quote().last_price`, signaled_at)하고 신호를 메모리(`_SignalLookup`)에 넣고 끝낸다.
- `manage()`는 매 틱 대기 중인 신호마다 `entry_ready`를 검사(분봉 조회는 종목당 `exit_check_every` 간격)하고, 참이면 기존 진입 경로(`_enter_now(signal)` — 현재 `enter()`의 동시 보유·호가·명목·현금·`execute_trade` 부분을 분리한 함수)를 부른다. 이때 추격 금지를 다시 적용한다: `quote.ask`가 신호 메타데이터의 `prev_close` 대비 `entry_filter.MAX_PRE_ENTRY_MOVE`를 넘으면 `entry_expired(reason="extended_at_entry")`.
- `entry_limit_ticks > 0`이면 지정가 = `ask − tick_size(ask) × entry_limit_ticks` (`krx_quant_core.market.ticks.tick_size`).
- 재시작 시 원장의 `entry_deferred` 중 메모리에 신호가 없는 것은 `entry_expired(reason="restart")`로 닫는다.

`entry_ready` 규칙(완성봉만, `c` 마지막 종가):
1. `entry_delay`: `now − signaled_at >= entry_delay`
2. `entry_pullback`: `c <= ref_price × (1 − entry_pullback)`
3. `entry_vwap_reclaim`: 신호 이후 완성봉 중 종가 < VWAP인 봉이 있었고, 마지막 완성봉 종가 > VWAP
켜진 규칙을 모두(AND) 만족하면 참.

- [ ] **Step 1: 실패 테스트** — 순수 함수 `entry_ready` 규칙별 경계 테스트 3개; `AutoTrader`를 SimBroker(0단계, 합성 분봉)로 구동해 (a) 기본값에서 `enter()` 즉시 `entry_submitted`, (b) `entry_delay=5분`이면 `entry_deferred` 후 5분 뒤 틱에서 `entry_submitted`, (c) `entry_max_wait` 초과 시 `entry_expired`, (d) 진입 시점 추격 금지 초과 시 `entry_expired(extended_at_entry)`, (e) `entry_limit_ticks=1`이면 지정가가 한 틱 아래. 시계는 `time_machine`. `execute_trade`의 DART·킬스위치 경로는 0단계 `day_replay` 테스트처럼 tmp 경로로 치환(테스트 안에서 monkeypatch).

- [ ] **Step 2: 실패 확인** → ImportError / AssertionError.

- [ ] **Step 3: 구현** (위 동작 그대로; `enter()` 본문을 `_admit(signal)`(즉시 거부 검사) + `_enter_now(signal, limit)`로 나누고 기존 로그·원장 이벤트 이름과 순서는 유지).

- [ ] **Step 4: 테스트 통과** — 신규 + `tests/unit` 전체 + `tests/unit/scripts/test_news_signal_poller.py`.

- [ ] **Step 5: 커밋**

```bash
git commit -am "feat(trading): entry timing (delay, pullback, VWAP reclaim, limit ticks) as TradingLimits settings (off by default)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: 종목 선정 설정 + 정책 파일 (운영 폴러까지 배선)

**Files:**
- Modify: `src/daytrade_it/application/signals/news_entry.py`, `src/daytrade_it/application/trading/auto_trader.py` (`rank_batch` key), `scripts/news_signal_poller.py`
- Create: `src/daytrade_it/application/trading/policy.py`
- Test: `tests/unit/application/signals/test_news_entry.py` (추가), `tests/unit/application/trading/test_policy.py` (신규)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class SelectionConfig: min_score: float = SIGNAL_THRESHOLD; subjects: tuple[str, ...] = ("main", "partial"); exclude_persistence: tuple[str, ...] = (); exclude_material: tuple[str, ...] = (); min_pre_move: Decimal | None = None; max_pre_move: Decimal | None = None; sessions: tuple[str, ...] = ("premarket", "intraday"); rank_key: str = "surprise"`
    - `max_pre_move=None`이면 `entry_filter.MAX_PRE_ENTRY_MOVE` 사용(현행).
    - 세션: `published_at < 09:00 KST` → premarket, 그 외 intraday.
    - `rank_key` ∈ `surprise | persistence | score`: `surprise`는 현행 `rank_batch`와 같은 정렬.
  - `build_entry_candidate(..., selection: SelectionConfig = SelectionConfig())` — 기본값에서 결과 동일. 점수·subject·제외 목록·세션은 LLM 판정 직후, 움직임 하한은 `pre_entry_move`로 검사. 거부 시 `logger.info("news_entry: not selected", ticker, reason)`.
  - `rank_batch(items, key: str = "surprise")` — 기본 동작 동일. `items`에 persistence를 넘길 수 있도록 튜플을 `(signal, surprise, score)` 그대로 두고 persistence는 `signal.metadata` (Task 4에서 `build_entry_candidate`가 `metadata["persistence"]` 추가)에서 읽는다.
  - `policy.py`: `@dataclass(frozen=True) class NewsPolicy: limits: TradingLimits = TradingLimits(); selection: SelectionConfig = SelectionConfig()`; `load_policy(path: Path) -> NewsPolicy` (파일 없으면 기본값, 알 수 없는 키는 ValueError); `dump_policy(policy, path)`. Decimal·timedelta는 문자열/초로 직렬화.
  - 폴러: `POLICY_PATH = Path("config/news_policy.json")`; `run()`에서 `policy = load_policy(POLICY_PATH)`, 시작 로그에 정책 내용 출력, `AutoTrader(..., limits=policy.limits)`, `build_entry_candidate(..., selection=policy.selection)`, `enter_batch`가 `rank_batch(pending, key=policy.selection.rank_key)`.

- [ ] **Step 1: 실패 테스트** — (a) `SelectionConfig()` 기본에서 기존 `test_news_entry` 결과 동일, (b) `min_score=0.6`이면 score 0.53 거부, (c) `min_pre_move=-0.02`이면 −3% 종목 거부, (d) `sessions=("premarket",)`이면 09:30 기사 거부, (e) `rank_batch(key="score")` 정렬, (f) `load_policy` 없는 파일 → 기본값, 왕복 `dump→load` 동일, 알 수 없는 키 → ValueError, (g) 폴러 모듈의 `POLICY_PATH` 부재 시 `AutoTrader`에 기본 `TradingLimits()`가 넘어가는지(폴러 `run` 대신 정책 적용 헬퍼 `_policy_parts(policy)`를 만들어 테스트).

- [ ] **Step 2: 실패 확인**, **Step 3: 구현**, **Step 4: `tests/unit` 전체 PASS**

- [ ] **Step 5: 커밋**

```bash
git add -A src scripts tests && git commit -m "feat(signals): SelectionConfig and a policy file shared by the live poller and replay (defaults = live)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 리플레이를 정책으로 구동 + 기본값 동일성 재확인

**Files:**
- Modify: `src/daytrade_it/application/replay/day_replay.py`, `scripts/eval/replay_parity.py`, `scripts/eval/replay_baseline.py`
- Create: `scripts/eval/replay_common.py`
- Test: `tests/unit/application/replay/test_day_replay.py` (추가), `tests/unit/scripts/test_replay_common.py` (신규)

**Interfaces:**
- Produces:
  - `DayConfig(policy: NewsPolicy = NewsPolicy(), manage_every=30s, exits_until=15:30, enter_on_decide=False)` — `limits`·`max_pre_entry_move` 필드는 `policy`로 대체. `MAX_PRE_ENTRY_MOVE` 치환 값은 `policy.selection.max_pre_move or Decimal("0.03")`. `LIVE_CONFIGS`(09-14 cap 2 등)는 `NewsPolicy(limits=..., selection=SelectionConfig(max_pre_move=Decimal("0.02")))`로 옮긴다.
  - `_run`은 `build_entry_candidate(..., selection=policy.selection)`, `rank_batch(..., key=policy.selection.rank_key)`.
  - `replay_common.py`: `@dataclass class DayData: day, events, bars, frozen, stale_before`; `load_day(day, cache_dir=Path("data/eval/cache/days")) -> DayData` (처음엔 `prepare_day`로 만들고 피클 캐시, 이후 캐시 사용); `day_net(data: DayData, policy: NewsPolicy, stocks, dart: Path, work_root: Path) -> dict` → `{"day", "net", "trades", "losers", "worst_trade", "errors", "leftovers"}` (매 호출 새 `SimBroker`, `trade_pnl`로 계산, errors > 0이면 `net`은 그대로 두고 `errors` 기록).

- [ ] **Step 1: 실패 테스트** — `day_net`가 합성 DayData(0단계 `test_day_replay` 합성 분봉 재사용)에서 기본 정책과 `stop_loss_pct` 정책의 결과가 다르고, 같은 정책 두 번 호출 결과가 같음(결정성).
- [ ] **Step 2~4:** 구현, `tests/unit` 전체 PASS.
- [ ] **Step 5: simnode 동일성 재확인** (Task 1 Step 6 명령과 동일). Expected: 게이트 PASS 두 날짜, 기준선 +39,915 날짜별 동일. 달라지면 멈추고 보고.
- [ ] **Step 6: 커밋·push**

```bash
git add -A src scripts tests && git commit -m "feat(replay): drive replay by NewsPolicy; cached day data and day_net for optimization

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push origin feat/bayes-pipeline-opt
```

---

### Task 6: 베이즈 판정기

**Files:**
- Create: `scripts/eval/bayes_judge.py`
- Test: `tests/unit/scripts/test_bayes_judge.py`

**Interfaces:**
- Produces:
  - `posterior_mu(d: list[float], prior_sd: float, nu: float = 4.0, grid: int = 401) -> tuple[np.ndarray, np.ndarray]` — μ 격자와 주변 사후확률(σ는 로그 균등 격자로 적분). 우도 Student-t(ν, μ, σ), 사전 μ ~ Normal(0, prior_sd).
  - `judge(candidate: dict[str, dict], baseline: dict[str, dict], prior_sd: float = 5000.0) -> dict` — 입력은 날짜별 `day_net` 결과. 반환 `{"p_mu_gt_0", "mu_mean", "days_ge_0", "n_days", "worst_day_candidate", "worst_day_baseline", "loser_ratio_candidate", "loser_ratio_baseline", "adopt": bool, "reasons": [...]}`.
  - 채택 조건(스펙 1~3): `p_mu_gt_0 >= 0.90`, `worst_day_candidate >= worst_day_baseline` 이고 `loser_ratio_candidate <= loser_ratio_baseline`, `days_ge_0 >= ceil(5/7 × n_days)`. 스펙 4(전진 5일)는 `"forward_pending": true`로 표시.

- [ ] **Step 1: 실패 테스트** — (a) d가 전부 +3000이면 `p_mu_gt_0 > 0.95`, (b) 전부 0이면 0.4~0.6, (c) 전부 −3000이면 < 0.05, (d) `judge`에서 최악일이 나빠지면 `adopt False`와 이유 문자열 포함, (e) 격자 사후 합이 1.
- [ ] **Step 2~4:** 구현(numpy만), 테스트 PASS.
- [ ] **Step 5: 커밋**

```bash
git add scripts/eval/bayes_judge.py tests/unit/scripts/test_bayes_judge.py
git commit -m "feat(eval): grid-posterior Bayesian judge with the spec's adoption rules

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: LODO × TPE 최적화 스크립트

**Files:**
- Create: `scripts/eval/optimize_policy.py`
- Test: `tests/unit/scripts/test_optimize_policy.py`

**Interfaces:**
- Consumes: `load_day`, `day_net` (Task 5), `NewsPolicy`, `SelectionConfig`, `TradingLimits`, `dump_policy` (Task 4), `judge` (Task 6)
- Produces:
  - `SPACES: dict[str, Callable[[optuna.Trial, NewsPolicy], NewsPolicy]]` — `"stage2"`, `"stage3"`.
  - `stage2` 탐색 파라미터(선정+타이밍+손절, 운영 기준값을 시도 0으로 enqueue):
    - `min_score` float [0.15, 0.6]; `subjects` categorical {"main,partial", "main"}; `min_pre_move` categorical {None, −0.05, −0.03, −0.02, −0.01}; `max_pre_move` categorical {0.01, 0.02, 0.03, 0.05}; `sessions` categorical {"premarket,intraday", "premarket"}; `rank_key` categorical {surprise, persistence, score}
    - `entry_delay_min` categorical {0, 3, 5, 10, 20}; `entry_pullback` categorical {None, 0.005, 0.01, 0.02}; `entry_vwap_reclaim` bool; `entry_limit_ticks` {0, 1}
    - `stop_loss_pct` {None, 0.01, 0.015, 0.02, 0.03}; `stop_vwap_gap` {None, 0.005, 0.01}; `stop_time_min` {None, 30, 60, 120}; `stop_orb_min` {None, 5, 15, 30}
  - `stage3` (stage2 결과 정책 위에서): `take_profit_pct` {None, 0.02, 0.03, 0.05, 0.08}; `trail_after_gain` {None, 0.01, 0.02, 0.03}; `trail_gap` {0.005, 0.01, 0.02}; `vwap_exit_gap` {None, 0.005, 0.01, 0.02}; `vwap_exit_min_gain` {0.01, 0.02, 0.03}; `add_on_min_gain` {None, 0.01, 0.02}; `flatten_at` {"14:30", "15:00", "15:15"}
  - `objective(train_days)` = 학습일 순손익 합 − 0.5 × max(0, −최악일 순손익) (손실 방지 가중; errors > 0 인 시도는 `-1e12`)
  - CLI: `--stage stage2|stage3 --trials 300 --jobs 14 --base-policy PATH --out data/eval/opt/<stage>/` → 폴드마다 `fold-<day>.json`(최적 파라미터, 학습 목적값, 빠진 날 결과), `oof.json`(빠진 날 결과 7개), `full.json`(전체 7일로 한 번 더 탐색한 최종 정책), `judge.json`(oof vs 기준선), `policy.json`(full 정책 dump)
  - 병렬: 폴드 7개 × 각 폴드 optuna study(TPESampler(seed=폴드번호), `n_trials`)를 `multiprocessing.Pool(jobs)`로; 각 워커는 `load_day` 캐시에서 7일 데이터를 한 번 읽는다. 작업 디렉터리는 워커별 `--work-root/<pid>`.

- [ ] **Step 1: 실패 테스트** — (a) `SPACES["stage2"]`에 `optuna.trial.FixedTrial`로 운영 기준값을 넣으면 `NewsPolicy()`와 같은 정책, (b) `objective`가 최악일 손실에 벌점을 주는지(합성 day 결과), (c) 합성 `day_net` 대체 함수로 2폴드 × 5시도 미니 실행이 파일 5종을 쓰는지.
- [ ] **Step 2~4:** 구현, 테스트 PASS.
- [ ] **Step 5: 커밋·push**

```bash
git add scripts/eval/optimize_policy.py tests/unit/scripts/test_optimize_policy.py
git commit -m "feat(eval): LODO x TPE policy optimizer for selection, timing, stop-loss and profit-taking

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push origin feat/bayes-pipeline-opt
```

---

### Task 8: 1단계 프롬프트 비교 (v2 vs v4, 추가 비용 0)

**Files:**
- Modify: `scripts/eval/build_replay_events.py`, `docs/research/2026-09-16-bayes-pipeline-eval.md`

**Interfaces:**
- Produces: `--features v2|v4` (기본 v2). `v4`면 아카이브 행의 `v4_shadow`로 features를 채우고(없으면 그 쌍 제외, 제외 수 출력) 출력 디렉터리 `data/eval/replay_events_v4/`. kr_quant 날짜는 v4 판정이 없으므로 v4 모드에서 건너뛴다.

- [ ] **Step 1:** 옵션 구현 + 단위 테스트(아카이브 행 2개, v4_shadow 하나 None → 이벤트 1개).
- [ ] **Step 2:** trader에서 `uv run python scripts/eval/build_replay_events.py --features v4` → simnode로 rsync(`data/eval/replay_events_v4/`).
- [ ] **Step 3:** simnode에서 `replay_baseline.py --events data/eval/replay_events_v4` (옵션 추가) 로 09-14·09-15 운영 정책 리플레이, v2 결과와 날짜별 순손익·거래·손실 거래 비교.
- [ ] **Step 4:** 결과 문서 "1단계: 프롬프트" 절 — 표본이 2일뿐이라 판정 불가 여부를 명시하고, `bayes_judge.judge`로 계산한 값을 그대로 적는다. 채택 규칙: 2단계는 v2로 진행(표본 2일로는 스펙 채택 조건 3을 만족할 수 없음 — 결과 문서에 그대로 기록).
- [ ] **Step 5:** 커밋·push.

---

### Task 9: 2단계 실행 — 선정 + 타이밍 + 손절

**Files:**
- Modify: `docs/research/2026-09-16-bayes-pipeline-eval.md`
- Create: `config/news_policy.candidate.json` (2단계 채택 시), `data/eval/opt/stage2/*` (git 제외)

- [ ] **Step 1: simnode 실행**

```bash
ssh young@192.168.45.9 'cd ~/git/daytrade-it-bayes && git pull -q && ~/.local/bin/uv sync -q && export $(grep -E "^KR_QUANT_DB=" ../quant-airflow/.env | xargs) && nohup ~/.local/bin/uv run python scripts/eval/optimize_policy.py --stage stage2 --trials 300 --jobs 14 --dart data/eval/dart_trader_20260915.db --out data/eval/opt/stage2 > data/eval/opt/stage2.log 2>&1 &'
```
완료 대기는 `ssh ... 'tail -3 data/eval/opt/stage2.log'`를 5분 간격으로 확인(예상 10~30분).

- [ ] **Step 2: 결과 확인** — `judge.json`, `oof.json`, 폴드별 파라미터 수렴(폴드 7개 중 같은 값이 나온 파라미터 목록).
- [ ] **Step 3: 결과 문서 "2단계" 절** — 기준선 대비 날짜별 빠진 날 순손익 표, 사후 P(μ>0), 최악일·손실 비율 비교, 채택 여부와 이유, 폴드 간 수렴한 파라미터, full 정책 내용.
- [ ] **Step 4:** 채택(스펙 조건 1~3 충족)이면 `full.json` 정책을 `config/news_policy.candidate.json`으로 커밋. 탈락이면 운영 기본값 유지를 문서에 적고, 3단계는 운영 기본 정책 위에서 진행.
- [ ] **Step 5:** 커밋·push.

---

### Task 10: 3단계 실행 — 수익실현 + 4단계 합동 확인

**Files:**
- Modify: `docs/research/2026-09-16-bayes-pipeline-eval.md`, `config/news_policy.candidate.json`

- [ ] **Step 1:** simnode에서 `optimize_policy.py --stage stage3 --base-policy <Task 9 채택 정책 또는 기본> --trials 300 --jobs 14 --out data/eval/opt/stage3` (Task 9 Step 1과 같은 방식).
- [ ] **Step 2: 4단계 합동 확인** — 2·3단계 최종 정책으로 7일 리플레이(`replay_baseline.py --policy config/news_policy.candidate.json`, 옵션 추가)와 LODO oof 결과를 `bayes_judge.judge`로 기준선과 비교. 동일성 게이트 두 날짜를 이 정책으로도 돌려 errors 0 확인(차이는 정책 차이라 비교 대상 아님 — errors와 데이터 실패만 확인).
- [ ] **Step 3: 결과 문서 "3단계·4단계" 절**과 결론: 최종 후보 정책, 스펙 조건 1~3 판정, 조건 4(전진 5거래일) 대기, 운영 반영 방법(`config/news_policy.json`으로 복사하면 폴러가 읽음 — 사용자 결정).
- [ ] **Step 4:** 커밋·push.
