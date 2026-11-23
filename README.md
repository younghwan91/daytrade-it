<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/brand/wordmark-dark.svg">
  <img alt="daytrade-it" src="docs/images/brand/wordmark-light.svg" height="52">
</picture>

# daytrade-it

**코스피·코스닥 뉴스 데이트레이딩 시스템과, 그 전략에 실제 우위(edge)가 있는지 측정한 리서치 기록입니다.**

토스 뉴스와 DART 공시를 읽어 LLM(Anthropic Claude Haiku)으로 호재·악재를 판정하고, 키움 브로커 어댑터와 공시 기반 리스크 게이트로 주문을 검증합니다.

[![CI](https://github.com/younghwan91/daytrade-it/actions/workflows/ci.yml/badge.svg)](https://github.com/younghwan91/daytrade-it/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[가이드](docs/guide.md) · [리서치](docs/research/README.md) · [MCP 도구](docs/mcp/README.md) · [English](README.en.md)

</div>

> **상태 — 실매매 정지.** 2026-09-17 이후 실시간 진입이 중단되었습니다(킬 스위치 `data/live_STOP`).
> 현재 데몬은 판정 기록과 페이퍼 원장만 남깁니다.
> [`docs/research/`](docs/research/README.md)의 리서치 결과, 이 방식의 뉴스 매매는 **왕복 거래비용(약 36.7 bp)을 넘는 우위가 없음**이 확인되었습니다.
> 종료된 가설, 사전 등록(pre-registration), 포워드 게이트가 해당 문서에 정리되어 있습니다. 이 저장소는 수익 전략이 아닌 리서치 기록 및 레퍼런스 구현으로 참고하세요.

## 빠른 시작

Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker가 필요합니다.

```bash
git clone https://github.com/younghwan91/daytrade-it.git
cd daytrade-it

uv sync --extra dev
cp .env.example .env        # 본인 DB DSN 및 API 키 입력

docker compose -f docker/docker-compose.yml up -d   # TimescaleDB / Redis
uv run alembic upgrade head
uv run uvicorn daytrade_it.interfaces.api.main:app --reload   # /docs 에서 Swagger UI 확인
```

```bash
uv run gptq analyze 005930          # 6자리 KRX 종목코드
uv run gptq signals --min-confidence 0.7
uv run gptq trade --paper
```

## 구성 요소

| 영역 | 역할 |
|---|---|
| **리스크 게이트** (`domain/services/risk_gate.py`) | DART 공시(상장폐지, 매매거래정지, 감사의견거절, 횡령 등)를 분류해 시그널과 무관하게 진입을 차단하거나 강제 청산합니다. 뉴스 감성만으로는 진입 차단만 가능하며 강제 청산은 불가합니다. |
| **뉴스 판정** (`infrastructure/external/`) | 토스 및 DART 폴러와 프롬프트 캐싱을 적용한 Anthropic 뉴스 판정 클라이언트입니다. |
| **시그널 및 매매** (`application/`, `scripts/news_signal_poller.py`) | 판정을 진입·청산으로 변환합니다(1주 지정가 주문, 가격 밴드, 일일/종목별 상한, 페이퍼 모드). |
| **브로커** (`infrastructure/brokers/`) | 키움 지정가 주문 및 페이퍼/시뮬레이션 브로커입니다. 시장가 주문 코드는 미검증 상태라 거부합니다. |
| **백테스트 / 리플레이** | 한국 시장 비용 모델로 과거 봉 데이터 위에서 라이브 매매 코드를 재현합니다(지정 호스트에서만 실행 — `krx-quant-core` 가드). |
| **리서치** (`scripts/eval/`, `docs/research/`) | 측정 스크립트, 사전 등록(`docs/research/frozen/`), 판정 문서입니다. |
| **인터페이스** | `gptq` CLI, FastAPI 앱, AI 에이전트용 MCP 서버(`docs/mcp/README.md`)입니다. |

시장 공통 로직(호가 단위, 비용 모델, 공시 리스크 게이트, 키움 상수)은 [`krx-quant-core`](https://github.com/younghwan91/krx-quant-core)에 있으며, 이 레포의 `risk_gate.py`, `cost_model.py`, `tick_size.py`는 얇은 재수출(re-export) 래퍼입니다.

## 문서

| 문서 | 내용 |
|---|---|
| [프로젝트 가이드](docs/guide.md) | 아키텍처, 환경 설정, 테스트 실행 방법 |
| [`docs/research/README.md`](docs/research/README.md) | 리서치 인덱스, 포워드 게이트, 종료된 가설 |
| [`docs/adr/`](docs/adr/README.md) | 아키텍처 결정 기록(ADR) |
| [`docs/mcp/README.md`](docs/mcp/README.md) | MCP 도구 안내 |
| [`PLAN.md`](PLAN.md) | 초기 마이그레이션 계획 (이력용; 최신 상태와 다를 수 있음) |

## 형제 레포지토리

| 레포 | 역할 |
|---|---|
| [`krx-news-client`](https://github.com/younghwan91/krx-news-client) | 토스·DART 뉴스 수집 클라이언트 라이브러리 |
| [`kiwoom-client`](https://github.com/younghwan91/kiwoom-client) | 키움증권 REST API 클라이언트 |
| [`krx-quant-core`](https://github.com/younghwan91/krx-quant-core) | KRX 시장 규칙, 비용 모델, 리스크 게이트, 실행 상수 |
| [`quant-airflow`](https://github.com/younghwan91/quant-airflow) | TimescaleDB 배치 적재 파이프라인 (여기서 읽기 전용 조회) |
| [`swing-it`](https://github.com/younghwan91/swing-it) | 코스피·코스닥 스윙 리서치 (실매매 없음) |

## 라이선스

MIT — [LICENSE](LICENSE) 참고.
