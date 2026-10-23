"""autoresearch 실험 파일 — 에이전트가 고치는 유일한 파일(program.md). 실행은 평가기:
``uv run python scripts/eval/autoresearch/prepare.py``.

기준선 = 운영 데몬의 판정 없는 뼈대: 오늘 새 기사가 보인 봉에서(장전 기사는 09:00 봉) 추격 금지(전일 정규장 종가 대비
θ 이상 오른 종목 제외) → 다음 봉 시가 진입, 큰 종목 먼저, 슬롯 5·하루 10건, 하드 손절, 15:00 정리.
숫자 파라미터는 ``params`` 범위만 정하고 값은 평가기가 폴드마다 학습 구간에서 고른다 — 손으로 맞추지 않는다.

계약(prepare.load_strategy — AST 허용 목록: numpy·pandas·math·functools·itertools·collections 만, 파일·OS·print·getattr·
더블 언더스코어·연도·종목코드 리터럴 금지):
  get_strategy() → {"params": {이름: ("int"|"float", lo, hi) | ("choice", (값, ...))},
                    "signals": f(V, x) → {"cand": bool[D,N,M], "score": float[D,N,M]},
                    "cfg": f(x) → {max_pos 1~10, max_entries 1~30, stop 0.015~0.08,
                                   exit_kind "close"|"vwap"|"ma"|"struct", exit_g 0~0.01, exit_w 5~60, struct_b 0.002~0.02}}
  V (읽기 전용, 날짜·종목코드 없음. D = 날짜, N = 그날 유니버스 절반 행(전일 거래대금 순, 날마다 다른 종목), M = 380 분봉 09:00~15:19):
    분봉 [D,N,M]: O·H·L·C(빈 봉은 직전 거래가), Vol(빈 봉 0), vwap(09:00 부터), na·nf·nx·nm(그 봉에 새로 보인 기사 수:
                  전체·[특징주]·[단독]·주가 움직임 단어. 기사는 createdAt + 5분에 보인다)
    날짜·행 [D,N]: uni(유효 행), prev_close(전일 정규장 종가), ret1·ret5·ret20, prev_range, tv_prev(log 전일 거래대금),
                  tv_rank(0 = 가장 큼), kosdaq, pre_all·pre_feat·pre_move(전일 15:00~오늘 09:00 에 보인 기사 수), hist5(직전 5거래일 기사 수)
    날짜·분 [D,M]: mkt(그날 유니버스 절반 동일가중, 시가 = 1)
  cand[d, n, m] = 봉 m 종가까지의 정보로 낸 매수 신호 → 평가기가 m+1 시가에 산다(m ≤ 329). score 가 클수록 빈 슬롯에 먼저 산다.
  신호는 봉 m 종가까지의 값만 쓴다(평가기가 (날짜, 분) 절단 재계산으로 검사 — 다르면 void).
"""

from __future__ import annotations

import numpy as np


def signals(V: dict, x: dict) -> dict:
    new = V["na"] > 0
    pre = (V["pre_all"] > 0)[:, :, None] & (np.arange(V["C"].shape[2]) == 0)[None, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        up = V["C"] / V["prev_close"][:, :, None] - 1.0
    cand = (new | pre) & (up < x["theta"]) & V["uni"][:, :, None]
    score = np.broadcast_to(-V["tv_rank"][:, :, None], cand.shape)
    return {"cand": cand, "score": score}


def cfg(x: dict) -> dict:
    return {"max_pos": 5, "max_entries": 10, "stop": x["stop"], "exit_kind": "close"}


def get_strategy() -> dict:
    return {
        "params": {"theta": ("float", 0.01, 0.10), "stop": ("float", 0.015, 0.08)},
        "signals": signals,
        "cfg": cfg,
    }
