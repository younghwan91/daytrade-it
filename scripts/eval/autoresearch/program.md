# autoresearch — 가드 붙인 장중 뉴스 매매 탐색 (사람이 고치는 파일)

[karpathy/autoresearch](https://github.com/karpathy/autoresearch) 의 루프를 이 레포 가드레일 안에 넣었다. 숫자 파라미터는
[dietmarwo/autoresearch-trading](https://github.com/dietmarwo/autoresearch-trading) 처럼 평가기가 폴드마다 고르고, 너는 **구조**만 바꾼다.
규칙·합격 바·잠금은 사전등록 `docs/research/frozen/2026-10-24-autoresearch-prereg.json`(정정이 있으면 `-amendN.json`)이다.
시작 전에 이 파일, 사전등록, `prepare.py`(읽기만), `train.py` 를 읽는다. 레포 `CLAUDE.md` 의 운영 체크아웃 규칙도 지킨다.

## 파일
- `scripts/eval/autoresearch/prepare.py` — 고정 평가기. 캐시(탐색 절반), 워크포워드(학습 3개월 → 시험 1개월, 11폴드),
  파라미터 점 48개 → 폴드마다 고원 선택, 분봉 체결, 비용, 점수, keep 판정. **고치지 않는다**(고치면 평가가 거부된다).
- `scripts/eval/autoresearch/train.py` — 네가 고치는 유일한 파일. `get_strategy()` 계약과 V 의 열은 파일 머리말에 있다.
- `scripts/eval/autoresearch/results.tsv` — 실험 기록(git 추적 안 함). 머리줄:
  `commit	score	folds_pos	oos_trades	trades_per_day	top1pct_removed	alpha_bp	net_bp	payoff	status	description`

## 실험 한 번
1. `train.py` 를 아이디어 하나로 고친다 → `git commit -m "exp: <한 줄>"`
2. 작업 트리 루트에서 `uv run python scripts/eval/autoresearch/prepare.py > run.log 2>&1` (20분 넘으면 죽이고 crash)
3. `grep -A30 "^---" run.log` — 비었으면 `tail -n 50 run.log`. 쉬운 오류(허용 목록 위반·모양 오류)면 고쳐 다시, 아니면 crash 로 적고 넘어간다.
4. `results.tsv` 에 한 줄 덧붙인다(모든 시도 — discard·crash 포함. 지우지 않는다. 평가기 원장은 따로 자동 기록된다).
5. **keep 은 평가기가 정한다:** `beats_best: True` 면 `uv run python scripts/eval/autoresearch/prepare.py --accept` 후 keep.
   아니면 `git reset --hard HEAD~1` (discard). 너는 keep 조건을 바꾸거나 해석하지 않는다.
6. 다시 1.

## 점수가 뜻하는 것
거래마다 알파 = 순수익(비용 차감) − 같은 진입·청산 시각 사이 그날 유니버스 절반 동일가중 수익. score = 11 시험 폴드
거래당 알파 평균(bp)의 평균 − 0.5 × 표준편차. 장중 하락 흐름과 시각 선택은 점수에 0 이다 — **무엇을 사나(선정)와 청산 메뉴**만 점수가 된다.
비용이 거래당 37~45bp 라 0 을 넘으려면 그만큼의 선정 알파가 필요하다. 지금까지 이 레포의 매치드 널 선정 알파는 +1.6bp(se 3.4)였다.
**음수에서 시작하는 것이 정상이다.** `net_bp`(실제 순수익, 흐름 포함)는 보고 열이다.

## 하지 말 것(어기면 결과 무효)
- `prepare.py`·사전등록·데이터·의존성 변경, 캐시·원장·`~/.cache/daytrade-it/` 파일 읽기·쓰기, `--final`·`--build`·`--manifest` 실행.
- 2026-10-13 이후 데이터, 운영 체크아웃(`~/git/daytrade-it`)·크론·데몬·`data/live_STOP` 건드리기, LLM API 호출(크레딧).
- 잠금 절반 종목 접근(코드 해시 짝홀을 추론해 피하거나 겨냥하지 않는다). 행 순서·값으로 날짜나 종목을 알아내려는 시도.
- 미래 정보: 봉 m 의 신호는 봉 m 종가까지의 값만. 그날 전체에 대한 집계(그날 고가·거래량 합·마지막 봉)는 미래다. 평가기가 절단 재계산으로 잡는다.
- 숫자 맞추기: 상수를 손으로 바꿔 점수를 올리지 않는다 — 숫자는 `params` 범위로 내고 평가기에 맡긴다. 범위를 한 점으로 좁히는 것도 손 맞추기다.
- 청산은 메뉴만(`close`·`vwap`·`ma`·`struct` + 하드 손절 ≥ 1.5% + 15:00). 신호로 익절·추적·시간 손절을 흉내 내지 않는다.
- 거래를 줄여 거래당 숫자를 올리는 방향(체결 < 30 인 폴드가 2개 초과 또는 전체 < 600 이면 void).

## 사용자 매매 원칙(설계 공간)
- 승률이 아니라 손익비·기대값. 승률이 높은데 손익비 < 1 이면 승자를 일찍 자르는 청산부터 의심한다. 좁은 손절로 손익비를 만들지 않는다.
- 진입·선정이 먼저다(무엇을 언제 사나). 손실은 짧게, 수익은 15:00 까지. 비용은 정가(평가기가 이미 그렇게 잰다).

## 이미 닫힌 것(같은 것을 파라미터만 바꿔 다시 내지 않는다 — `docs/research/README.md`)
- 익절·추적·시간 청산: 15:00 보다 나쁘다(큰 승자를 자른다). 순차 RL 청산·신경망 청산·주춤/본절: 전부 고정 손절보다 나쁘다.
- 거래량 5배 양봉 진입 + −2%/+4%: 거래당 −55bp. 차트 손익비 2:1 자리(손절 = 당일 저가): 승률 13.5%, −58bp.
- 지연 단축: 발행 시각에 사도 ≤ 7bp. 장 시작 일괄 매수: 125일로 보면 가장 나쁜 시각대. 불타기: −82bp.
- LLM 필드·상호작용·HMM 레짐 조건화: 선정 알파 −2bp(se 1.2). `[특징주]` 제목: 20일 −250bp(피할 신호, 장중은 미확인).
그러니 다른 **구조**를 시도하라: 어떤 뉴스 맥락(장전/장중, 기사 몰림, 침묵 뒤 첫 기사, 이미 움직인 정도), 진입 시점의
가격·거래량 모양(되돌림 뒤, VWAP 위/아래, 장초 범위 돌파 실패), 순위(무엇을 먼저 사나), 청산 메뉴 선택.

## 멈추지 않는다
루프가 시작되면 사람에게 묻지 않는다. 실험 사이에 말하지 않는다. 10 실험마다 한 번만:
```
=== 진행 (N 실험) ===
기준선 X → 최고 Y (top1 제거 Z) · keep K / discard D / crash C
배운 것: <한 줄>
```
사람이 끊을 때까지 계속한다. 졸업 심사(`--final`)는 사람이 정한 때 메인 세션이 한다 — 너는 하지 않는다.
