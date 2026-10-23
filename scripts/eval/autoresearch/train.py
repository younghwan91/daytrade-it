"""autoresearch 실험 파일 — 에이전트가 고치는 유일한 파일(program.md). 실행은 평가기:
``uv run python scripts/eval/autoresearch/prepare.py``.

기준선 = 운영 데몬의 판정 없는 뼈대: 오늘 새 기사가 보인 봉에서(장전 기사는 09:00 봉) 추격 금지(전일 정규장 종가 대비
θ 이상 오른 종목 제외) → 다음 봉 시가 진입, 큰 종목 먼저, 슬롯 5·하루 10건, 하드 손절, 15:00 정리.
숫자 파라미터는 ``params`` 범위만 정하고 값은 평가기가 폴드마다 학습 구간에서 고른다 — 손으로 맞추지 않는다.

계약(prepare.load_strategy — AST 허용 목록: import 는 numpy·pandas·math·functools·itertools·collections 만, 모듈 속성으로
하위 모듈에 내려가기(np.random·np.linalg·np.fft 말고)·밑줄로 시작하는 속성·파일·OS·print·getattr·연도·종목코드 리터럴 금지):
  get_strategy() → {"params": {이름: ("int"|"float", lo, hi) | ("choice", (값, ...))},
                    "signals": f(V, x) → {"cand": bool[D,N,M], "score": float[D,N,M]},
                    "cfg": f(x) → {max_pos 1~10, max_entries 1~30, stop 0.015~0.08,
                                   exit_kind "close"|"vwap"|"ma"|"struct", exit_g 0~0.01, exit_w 5~60, struct_b 0.002~0.02}}
  V (읽기 전용, 날짜·종목코드 없음. D = 날짜, N = 그날 유니버스 절반 행(전일 거래대금 순, 날마다 다른 종목), M = 380 분봉 09:00~15:19):
    분봉 [D,N,M]: O·H·L·C(빈 봉은 직전 거래가), Vol(빈 봉 0), vwap(09:00 부터), na·nf·nx·nm(그 봉에 새로 보인 기사 수:
                  전체·[특징주]·[단독]·주가 움직임 단어. 기사는 createdAt + 5분에 보인다)
    날짜·행 [D,N]: uni(유효 행), prev_close(직전 세션 분봉 정규장 마지막 종가 — 분봉과 같은 가격 기준), ret1·ret5·ret20(수정 일봉),
                  vol20(직전 20일 일수익 표준편차), prev_range, tv_prev(log 전일 거래대금), tv_rank(0 = 가장 큼), kosdaq,
                  pre_all·pre_feat·pre_move(전일 15:00~오늘 09:00 에 보인 기사 수), hist5(직전 5거래일 기사 수)
    날짜·분 [D,M]: mkt(그날 유니버스 절반 동일가중, 시가 = 1).  날짜 [D]: mkt_prev1·mkt_prev5(직전 1·5 세션 mkt 장중 수익)
  cand[d, n, m] = 봉 m 종가까지의 정보로 낸 매수 신호 → 평가기가 m+1 시가에 산다(m ≤ 329). score 가 클수록 빈 슬롯에 먼저 산다.
  날짜끼리 독립: 날짜 d 의 신호는 날짜 d 의 배열(분봉·[D,N]·[D] 값)만으로 만든다 — 날짜 축을 가로지르는 계산(다른 날 값·날짜
  인덱스)은 평가기의 날짜 순열 검사에서 void 다. 봉 m 의 신호는 봉 m 종가까지의 값만(절단 재계산으로 검사 — 다르면 void).
  점수(알파)는 같은 날 같은 특성 칸(코스닥 × 규모 × 변동성) 종목들의 같은 시간창 수익을 뺀 값이다.
  신호는 결정론이어야 한다(난수 동점 처리 금지 — 순열 검사가 void 로 본다). 날짜 횡단 통계(전 기간 분위수·정규화)도 void.
"""

from __future__ import annotations

import numpy as np


def signals(V: dict, x: dict) -> dict:
    m = np.arange(V["C"].shape[2])
    new = (V["na"] > 0) & (m >= 4)[None, None, :]  # 장중 기사만, 장초(진입 < 09:05) 제외
    with np.errstate(invalid="ignore", divide="ignore"):
        up = V["C"] / V["prev_close"][:, :, None] - 1.0
    above = V["C"] > V["vwap"]  # 기사 봉 종가가 VWAP 위(그날 산 사람들이 이익인 쪽)
    day_up = V["C"] > V["O"][:, :, :1]  # 오늘 시가보다 위(갭 뒤 장중으로도 오르는 중)
    dip = V["C"] < np.maximum.accumulate(V["C"], axis=2)  # 오늘 최고 종가 아래(신고 종가 추격 아님)
    cs = np.cumsum(V["C"], axis=2)
    lag = np.concatenate([np.zeros(cs.shape[:2] + (10,)), cs[:, :, :-10]], axis=2)
    ma10 = (cs - lag) / np.minimum(m + 1, 10)[None, None, :]  # 직전 10분 종가 평균(봉 m 포함)
    pull = ~(V["C"] > ma10)  # 짧은 평균 아래 — 상승 추세(VWAP 위) 안의 눌림
    cand = new & above & day_up & dip & pull & (up < x["theta"]) & V["uni"][:, :, None]
    score = np.broadcast_to(V["tv_rank"][:, :, None], cand.shape)
    return {"cand": cand, "score": score}


def cfg(x: dict) -> dict:
    return {"max_pos": x["slots"], "max_entries": 30, "stop": x["stop"], "exit_kind": "close"}


def get_strategy() -> dict:
    return {
        "params": {
            "theta": ("float", 0.01, 0.10),
            "stop": ("float", 0.015, 0.08),
            "slots": ("int", 6, 10),
        },
        "signals": signals,
        "cfg": cfg,
    }
