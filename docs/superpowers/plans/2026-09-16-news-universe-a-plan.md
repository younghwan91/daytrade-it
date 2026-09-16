# A안 구현 계획 — 같은 뉴스 풀(대시보드 ∪ 종목별 순회)

> For agentic workers: 태스크마다 별도 브랜치·워크트리에서 작업하고, 아래 인터페이스 계약을 그대로 지킨다. TDD. 커밋·push를 자주 한다.

**Spec:** `docs/superpowers/specs/2026-09-16-news-universe-a-plan-design.md`
**Base branch:** `feat/bayes-pipeline-opt` (리플레이 엔진·정책·밴딧 포함)

## Global Constraints

- 운영에서 가능한 것만, 못 볼 건 학습하지 않는다(백테스트 = 실매매).
- Toss 비공식 API 보호: 요청 간격 ≥ 1초(프로세스당), 429/403 3회 연속이면 정지. 평일 08:20~15:45에는 **백필·검증용 호출 금지**(운영 순회 dry-run은 사용자 승인 후 별도 IP에서만).
- 실주문 경로(`AutoTrader` 주문, `execute_trade`)는 이 계획에서 바꾸지 않는다. 순회 풀 주문 활성화는 계획 밖(채택 후 사용자 결정).
- 백테스트·대량 계산은 simnode(`require_backtest_host()`). kr_quant는 읽기만(분봉 백필은 기존 `scalp-intraday` 도구로 추가만).
- LLM: Haiku 4.5(`claude-haiku-4-5`), 과거 판정은 Batch API, 예산 상한은 CLI 인자로 강제.
- 커밋 트레일러: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

## 인터페이스 계약

### C1. 백필 파일 (A1, 진행 중)
`data/toss_universe/{code}.jsonl`, 행 = Toss 종목별 API body 원소 + `"code"`. 필드: `id, title, summary, createdAt (KST naive ISO), source, newsType, ...`.

### C2. `src/daytrade_it/application/news/universe.py` (A2)
- `UNIVERSE_SIZE = 300`, `SWEEP_INTERVAL = dt.timedelta(seconds=1)`
- `universe_size(override: int | None = None) -> int` — **순회 유니버스 크기의 단일 진실 원천.** `override`(CLI 플래그/정책 필드) > 환경변수 `DAYTRADE_NEWS_UNIVERSE_SIZE` > `UNIVERSE_SIZE`. 정수가 아니거나 0 이하인 환경변수 값은 무시하고 기본값으로 떨어진다(오타가 운영 데몬을 죽이지 못하게 — `mention_prefilter.fanout_max()` 와 같은 계약).
- `select_universe(daily_values: dict[str, float], size: int | None = None) -> list[str]` — 전일 거래대금 내림차순, 동률은 코드 오름차순. 결정적. `size=None` 이면 호출 시점에 `universe_size()` 로 해석하므로, 상수를 직접 읽지 않는 호출자에게도 환경변수가 그대로 전달된다.
- 이 하나의 값을 읽는 곳: 라이브 순회(`scripts/news_sweep_shadow.py --size`), C3 이벤트 빌더(`scripts/eval/build_universe_events.py --universe-size`), 분석 스크립트(`scripts/eval/universe_readiness.py --universe-size`, `scripts/eval/universe_coverage.py --universe-size`). **상수를 인라인하지 말 것** — 라이브와 백테스트가 한 숫자에서 갈라지는 순간 판정 캐시와 이벤트 풀이 어긋난다.

#### N = 300 을 고른 이유 — 비용이 아니라 **순회 지연**
근거: `.superpowers/sdd/2026-09-16-bayes-stages/prefilter-report.md` **Addendum C** (97,721 건의 판정된 쌍에 대한 rank-bucket 분석, 331,308 쌍 전부가 랭크를 해석).

08:00~15:20 사이 초당 1요청 = 26,400 요청/일, 랩 시간 = `N / (26,400 − articles_per_day)`:

| N | 랩 시간 | BUY 커버리지 | 남은 6개월 판정 비용(채택 사전필터 적용) |
|---|---|---|---|
| 200 | 3.44분 | 66.4% | $96.21 |
| **300** | **5.19분** | **79.5%** | **$111.26** |
| 400 | 6.94분 | 87.9% | $120.77 |
| 600 | 10.45분 | 100% | $133.23 |

- **지연이 진짜 상금이다.** 600→300 은 랩을 10.45분 → 5.19분으로 줄인다. 순회로 발견되는 기사의 기대 탐지 지연은 랩의 절반이므로 평균 판정 도달 시간이 ~5.2분 → ~2.6분, 최악이 ~10.5분 → ~5.2분. 장중 뉴스에서는 첫 임펄스에 들어가느냐 지나간 뒤에 들어가느냐의 차이.
- **포기하는 20.5% 는 한계 가치가 가장 낮은 구간이다.** 랭크 301-600 의 랭크 슬롯 하나는 6개월에 BUY 10-13건, 남기는 구간은 22-80건. 모든 랭크 슬롯이 똑같이 랩 1초를 먹으므로 잘라야 할 기준은 "지연 1초당 BUY" 이고, 그 무릎이 200~300 사이에 있다. N=200 은 지연이 더 좋지만 BUY 의 3분의 1을 버리고, 201-300 구간은 실제 거래대금이 받쳐 주는 구간 중 BUY 비율이 가장 높다(21.4%).
- **비용 $111.26 은 부수적 이득이지 논거가 아니다** (600 대비 $21.97 절감). 비용은 약한 레버다 — 뉴스 물량이 머리에 몰려 있어서 유니버스의 3분의 2를 잘라도 청구서는 28%만 준다.
- **재판정은 필요 없다.** 캐시는 `news_id|code` 로 키가 잡혀 있고 기존 97,721건 중 83,588건이 이미 N=300 안에 있다.
- **주의:** N=300 산출물은 기존 N=600 산출물(기준선·옵티마이저·밴딧)과 **다른 풀**이다. 직접 비교 금지. 실험마다 사전필터 스위치와 똑같이 N 을 못 박을 것.
- `sweep_slot(code: str, universe: list[str], day: dt.date, start: dt.time = dt.time(8, 0), interval=SWEEP_INTERVAL) -> Callable[[dt.datetime], dt.datetime]` 대신 순수 함수 `next_visit(code, universe, day, after: dt.datetime, start=dt.time(8,0), interval=SWEEP_INTERVAL) -> dt.datetime | None` — 순회는 `start`부터 universe 순서대로 interval마다 한 종목, 끝나면 처음부터 반복. `after` 이후 그 종목의 첫 방문 시각(KST aware). 15:20 이후면 None.
- `live_article_id(news_id: str) -> str` — 운영 키 uuid 문자열.
- `related_codes(rows_by_id: dict[str, set[str]]) -> dict[str, tuple[str, ...]]` — 기사 ID → 정렬된 관련 종목.

### C3. `scripts/eval/build_universe_events.py` (A3)
- 입력: C1 파일, kr_quant 일봉(전일 거래대금), 판정 캐시 `data/eval/cache/universe_v2.jsonl`(`{"key", "v2"}`).
- 날짜 d의 이벤트: 전일 거래대금으로 `select_universe`; 기사 (id, createdAt) 중 발행 시각이 전 거래일 15:20 ~ d 15:20이고 관련 종목 ∩ universe 가 있는 쌍; `decide_at = next_visit(code, universe, d, max(createdAt, 08:00)) + judge_lag`(판정 지연 중앙값, 기본 운영 아카이브 측정값); `batch_at = decide_at + 1초`; features = 캐시 v2(없으면 이벤트 제외, 제외 수 출력).
- 출력: `data/eval/replay_events_universe/{d}.jsonl` (0단계 `ReplayEvent` 형식, `source="universe"`).

### C4. `scripts/eval/judge_batch.py` (A3)
- `--pairs FILE(jsonl: key, code, name, market, sector, title, body, category)` `--budget-usd N` `--sample N --stratify month,sector,session --seed S`
- 운영 `anthropic_client`의 프롬프트 빌더와 `ArticleFeatures` 파싱을 그대로 써서 Batch API 요청 생성·제출·폴링·결과 저장(C3 캐시 형식). 요청 전 토큰 수 추정으로 예산 초과면 중단.

### C5. `scripts/news_sweep_shadow.py` (A5)
- 운영과 같은 판정·신호 생성 경로(`process_pair` 계열)를 쓰되 **주문 없음**. universe 순회(C2), 새 기사만 상세 API로 `companyCodes` 확인, 판정 결과를 `data/news_judgments_universe/{day}.jsonl`에 운영 아카이브와 같은 형식 + `"pool": "universe"|"dashboard"`로 기록. 대시보드 데몬과 같은 기사면 같은 키.
- 차단 감지 시 순회 정지·로그. 실행은 사용자 승인 후.

## 태스크

| # | 내용 | 산출 | 의존 |
|---|---|---|---|
| A1 | 6개월 백필 + 그룹핑 검증 (진행 중) | C1, 검증 보고 | — |
| A2 | universe·순회 시각·키·관련 종목 순수 라이브러리 | C2 + 테스트 | — |
| A3 | 유니버스 이벤트 빌더 + Batch 판정기 | C3, C4 + 테스트 | C2 |
| A4 | 유니버스 분봉 커버리지 점검(6개월) + 누락 백필 계획 | 보고서, 누락 목록 | — |
| A5 | 순회 섀도 러너(주문 없음) | C5 + 테스트 | C2 |
| A6 | 표본 판정 실행(예산 상한) → 이벤트 생성 → 6개월 기준선 리플레이 | 결과 문서 | A1, A3, A4 |
| A7 | 6개월 규칙 최적화(시간순 폴드) + 밴딧 재검증 + 판정 | 결과 문서, 후보 정책 | A6 |
