# daytrade-it 가이드

[← README로 돌아가기](../README.md)

## 목차

- [아키텍처](#아키텍처)
- [환경 설정](#환경-설정)
- [테스트](#테스트)

## 아키텍처

```
src/daytrade_it/
├── domain/          # 엔티티, 값 객체, 서비스 (RiskGate, RiskManager, 비용, 호가)
├── application/     # 커맨드/쿼리/핸들러, 시그널, 트레이딩, 리플레이, 뉴스, ML
├── infrastructure/  # 외부 연동 (Toss, DART, Anthropic, quant-airflow DB), 브로커, 영속성
├── interfaces/      # api (FastAPI), cli (Typer), mcp, websocket
└── shared/          # 설정, 로깅, 에러
```

모든 주문은 `KiwoomBroker.place_order()`에 도달하기 전 `RiskManager.validate_trade()`를 통과해야 한다. 매수 시 리스크 게이트는 최근 7일간 해당 종목의 중대 공시를 확인하며, 데이터 파일이 없거나 오래된 경우 통과시키지 않고 진입을 거부한다.

유래: 미국 주식 시스템이었던 `gpt-quant-v2`(삭제됨)를 한국 주식 시스템으로 재개발했다.

## 환경 설정

설정은 `GPTQ_` 접두사가 붙은 환경 변수에서 읽으며, 전체 목록은 `.env.example`에 있다(`shared/config/settings.py`에서 정확한 필드명 확인). 키움 앱 키/시크릿, DART 키, Anthropic 키, DB DSN 등 인증 정보는 환경 변수로만 주입하며 값이 채워진 `.env`는 절대 커밋하지 않는다. `GPTQ_BROKER_TYPE`은 `paper` 또는 `kiwoom`이며 키움 모의투자 플래그는 기본 활성화되어 있다.

## 테스트

```bash
uv run pytest tests/unit tests/integration --no-cov -q
uv run ruff check src tests
```

`tests/integration/`의 일부 테스트는 `quant-airflow` DB 연결이 필요하며, DB가 없으면 자동으로 건너뛴다. `tests/e2e/`는 실행 중인 API 서버(`E2E_BASE_URL`, 기본 `http://localhost:8000`)가 필요하다.
