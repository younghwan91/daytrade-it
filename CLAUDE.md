# daytrade-it

토스·DART 뉴스를 LLM 으로 판정해 1주 지정가로 사고 15:00 에 청산하는 데이트레이딩 시스템과, 그 우위를 재는
연구 레포. 2026-09-17 부터 실매매 진입 정지(`data/live_STOP`), 데몬은 `--paper` 로 판정·종이 장부만 쌓는다.
연구 결론("왕복 비용 36.7bp 를 넘는 우위 없음")과 닫힌 가설 목록은 `docs/research/README.md` — 연구 질문이면 먼저 읽는다.

## 런타임 체크아웃 — 여기서 작업하지 않는다

- IMPORTANT: `~/git/daytrade-it` 는 크론이 도는 런타임이다. 08:45 `scripts/preopen_check.sh`, 08:55
  `scripts/news_signal_daemon.sh`, 15:50 `scripts/eval/forward_gate.py`, 18:30 `kqc nightly` 가 이 디렉터리의
  코드와 `data/`(장부·판정·`live_STOP`)를 쓴다. 이 디렉터리는 main 에 두고, 브랜치를 바꾸거나 커밋하지 않는다.
  작업은 워크트리에서: `git worktree add ../daytrade-it-<주제> -b <브랜치> origin/main` (훅이 checkout/switch 를 막는다).
- 워크트리는 `.venv` 를 따로 만든다(`uv sync --extra dev`). main 의 `.venv` 를 심링크하면 editable install 이
  워크트리 `src` 를 가리켜 런타임 데몬이 워크트리 코드로 돈다.
- `~/git/daytrade-it-rlnews`(3c409b6, detached)도 런타임이다. 16:30 크론 `scripts/eval/pipeline/collect_toss_daily.sh`
  가 거기서 돈다(이 스크립트는 main 에 없다). 워크트리 정리 때 지우거나 체크아웃하지 않는다.
- 크론은 `deploy/crontab.simnode` 를 고쳐 커밋한다. `crontab -e` 로 고친 건 07:50 cron-install 이 origin/main 기준으로
  덮는다(위 rlnews 16:30 줄만 레포 밖 비관리 줄).
- 장중에 live poller(`--paper` 없이)를 손으로 띄우지 않는다. `data/auto_trader/poller.lock` 을 잡아 크론 데몬이 하루 종일
  쉰다(청산 포함). `--paper` 실행도 판정마다 데몬과 같은 Anthropic 크레딧을 쓴다.
- `pkill -f <패턴>` 금지 — simnode 의 다른 에이전트와 ssh 셸까지 죽인다(크론의 15:40 pkill 만 예외).

## 실매매 안전

- 재개 = `data/live_STOP` 삭제 + 데몬 실행줄 `--paper` → `--confirm-real`. `data/eval/forward_gate/latest.md` 가
  `promote=True` 이고 사용자가 명시 승인하기 전에는 제안도 하지 않는다. `live_STOP` 삭제는 훅이 막는다.
- 키움은 scalp-it 과 같은 계좌·현금·appkey·토큰이다. 토큰 revoke/logout 을 부르지 않는다(scalp-it 실주문이 끊긴다, aaeedcb).
  키움 필드·값을 고칠 땐 `krx_quant_core.execution.kiwoom_spec` 과 scalp-it 실주문 어댑터와 먼저 대조한다.
- `risk_gate.py`·`cost_model.py`·`tick_size.py`·`dart_disclosure_db.py`·`shared/runtime_host.py` 는 krx-quant-core 의
  얇은 재수출이다. 로직은 core 에서 고치고, 여기선 `pyproject.toml` 의 `krx-quant-core==` 핀만 올린다.

## 명령 (워크트리 루트, 2026-10-12 확인)

```bash
uv sync --extra dev
uv run pytest tests/unit tests/integration --no-cov -q   # 약 10초, 전부 통과
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy src            # application·infrastructure·interfaces 는 pyproject 에서 ignore_errors
```

- `tests/e2e/` 는 떠 있는 API 서버가 있어야 한다. 없으면 31건이 ConnectError 로 실패하니 판정에 쓰지 않는다
  (README 의 "40 pre-existing failures" 는 낡은 설명).
- pre-commit 은 전역 git 훅이 `uvx pre-commit` 으로 돌린다. detect-secrets 가 DSN 모양 주석을 잡으면 그 줄 끝에
  `# pragma: allowlist secret` 을 붙인다(30846f7).
- 백테스트·리플레이는 simnode 에서만 돈다(`krx_quant_core.runtime.host` 가드).
- DB: 시세는 `KR_QUANT_DB`(`../quant-airflow/.env`), 신호는 `GPTQ_DATABASE_URL`(레포 `.env`, gptquant DB).
  simnode 에 `psql` 이 없다 → `uv run python` + psycopg2.
- numpy 를 여러 프로세스로 돌릴 땐 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

## 데이터 함정

- `daily_bars` 는 2026-09-14 부터 NXT 거래 종목의 종가·고가·저가·거래량에 NXT(~19:59) 체결이 섞였다
  (005930 09-17: 일봉 종가 256,000 vs 정규장 분봉 마지막 252,500). 정규장 값은 `minute_bars` 09:00~15:30 으로 다시
  만든다(swing-it `src/swing_it/krx_session.py`). `docs/research/2026-10-05-cross-repo-synthesis.md` 끝의
  "정정 (2026-10-08)" 절("일봉은 정상")은 틀렸다 — 인용하지 않는다.
- `minute_bars.ts` 는 시간대 없는 KST 다. `at time zone` 을 걸면 비교가 어긋난다(이 실수로 위 오염을 오진이라 판정했다).
- `daily_bars.trade_value` 는 백만원 단위다(f09f4c8, volume-entry amend1 — 원으로 읽어 문턱이 꺼진 실행이 무효 처리됐다).
- 비용은 리터럴로 쓰지 않는다. `scripts/eval/costs.py`(core `round_trip_cost` 위임)를 쓴다. 예전 `COST = 0.00215` 는
  스프레드가 빠져 15bp 이상 낙관적이었다.
- 분봉 수집은 scalp-it `cli_intraday --universe-top 1400` 이 한다. 소형주는 분봉이 없어 봉 종가로 폴백된다.
- 데몬은 KRX 휴장일에도 돌아 종이 체결을 남긴다. 게이트·리플레이는 분봉 없는 날을 뺀다.

## 연구 절차 (이 레포 관례)

1. 사전등록 `docs/research/frozen/<날짜>-<주제>-prereg.json` 을 결과 전에 커밋한다: `research(prereg): … — 결과 전 고정`.
2. 측정기 `scripts/eval/<주제>.py` + `tests/unit/scripts/test_<주제>.py` 커밋: `feat(eval): … (결과 전 커밋)`.
3. 실행 → `docs/research/<날짜>-<주제>.md` 와 `docs/research/README.md` 색인에 한 줄을 추가한다.
4. `research/<주제>` 브랜치를 main 에 `--no-ff` 로 병합한다: `Merge research/<주제> — 연구 스크립트·문서만(새 파일), 실매매 코드 무변경`.

- frozen 파일은 고치지 않는다(설정 deny + 훅). 구현 버그 정정은 새 파일 `<원본>-amendN.json` 에
  `amends`·`bug`·`invalid_run_seen`(버그 상태 결과를 봤는지)·`verdict_rule` 을 적는다.
- 홀드아웃·`--confirm` 채점은 사용자 승인 없이 돌리지 않는다. 죽은 작업은 정지 지시가 있었는지 확인하기 전엔
  다시 띄우지 않는다(2026-09-22 재실행이 예약된 홀드아웃을 소진했다).
- 결과를 믿기 전에: 결과와 같은 기간의 양으로 만든 표본·순위·필터, 체결 여부로 고른 부분집합이 있는지 본다
  (대시보드 소속 +233bp 등 이 레포에서 4번 반복된 거짓 우위). 스윕 풀 결과는 실매매 풀에서 6번 뒤집혔다 —
  채택 전엔 실매매 코드를 그대로 돌리는 리플레이(`application/replay`, `scripts/eval/replay_parity.py`)로 확인한다.

## 문서

- `docs/research/README.md` — 연구 색인·전진 게이트·데이터 공백·운영 상태. 연구 질문이면 항상 먼저.
- `docs/research/frozen/` — 사전등록 원본. 판정은 여기 적힌 규칙으로만.
- `README.md` "What's real vs. still a stub" — 컴포넌트가 실제 배선됐는지 볼 때.
- `docs/adr/`, `docs/mcp/README.md` — 구조 결정, MCP 도구.
- `PLAN.md` (85KB) — gpt-quant-v2 → KRX 이전 8단계 기록. 옛 결정의 이유를 찾을 때만.

커밋 메시지는 한국어 conventional(`feat(news): …`, `fix(eval): …`, `docs(research): …`, `research(prereg): …`).
