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
- `UNIVERSE_SIZE = 600`, `SWEEP_INTERVAL = dt.timedelta(seconds=1)`
- `select_universe(daily_values: dict[str, float], size: int = UNIVERSE_SIZE) -> list[str]` — 전일 거래대금 내림차순, 동률은 코드 오름차순. 결정적.
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
