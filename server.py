# server.py
import io
import base64
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Response

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ishihara import (
    generate_ishihara_plate,
    calculate_threshold_and_confidence,
    interpret_threshold,
)

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

# CORS (개발용으로 전체 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 나중에 실제 도메인으로 제한하면 좋음
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 요청/응답 스키마
# -------------------------

class GeneratePlateRequest(BaseModel):
    number: int              # 이시하라 판에 넣을 숫자
    axis: str                # 'protan' | 'deutan' | 'tritan'
    deltaE: float            # 난이도
    size: int = 600
    seed: Optional[int] = None  # 재현성 위해 seed 고정용 (없으면 서버에서 랜덤)


class GeneratePlateResponse(BaseModel):
    image_base64: str        # PNG base64
    number: int
    axis: str
    deltaE: float
    seed: int                # 실제 사용된 seed (나중에 재생성용)


class TrialItem(BaseModel):
    deltaE: float
    correct: bool


class ThresholdRequest(BaseModel):
    axis: str                # 'protan' | 'deutan' | 'tritan'
    history: List[TrialItem]


class ThresholdResponse(BaseModel):
    threshold: Optional[float]
    confidence: float
    level: str
    description: str


# -------------------------
# 유틸: RNG 생성
# -------------------------

def make_rng(seed: Optional[int] = None):
    if seed is None:
        # int32 최대값까지 (넘파이 RandomState가 허용하는 범위)
        seed = int(np.random.randint(0, 2**31 - 1))
    else:
        # 혹시나 너무 큰 값 들어와도 안전하게 잘라주기
        seed = int(seed) & 0x7fffffff  # 0 ~ 2**31-1

    rng = np.random.default_rng(seed)
    return rng, seed

# -------------------------
# 1) 이시하라 판 생성 API
# -------------------------

@app.post("/generate_plate", response_model=GeneratePlateResponse)
def generate_plate(req: GeneratePlateRequest):
    try:
        rng, used_seed = make_rng(req.seed)
        img = generate_ishihara_plate(
            number=req.number,
            axis=req.axis,
            deltaE=req.deltaE,
            rng=rng,
            size=req.size,
        )

        # PNG -> base64
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("ascii")

        return GeneratePlateResponse(
            image_base64=img_b64,
            number=req.number,
            axis=req.axis,
            deltaE=req.deltaE,
            seed=used_seed,
        )

    except Exception as e:
        logger.error(f"❌ generate_plate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate plate")


# -------------------------
# 2) Threshold 계산 API
# -------------------------

@app.post("/compute_threshold", response_model=ThresholdResponse)
def compute_threshold(req: ThresholdRequest):
    try:
        if not req.history:
            return ThresholdResponse(
                threshold=None,
                confidence=0.0,
                level="측정 불가",
                description="데이터가 없습니다.",
            )

        # [(deltaE, correct), ...] 형태로 변환
        history_tuples = [(t.deltaE, t.correct) for t in req.history]

        threshold, confidence = calculate_threshold_and_confidence(history_tuples)
        level, desc = interpret_threshold(threshold, req.axis)

        return ThresholdResponse(
            threshold=threshold,
            confidence=confidence,
            level=level,
            description=desc,
        )

    except Exception as e:
        logger.error(f"❌ compute_threshold error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to compute threshold")
    
@app.get("/preview_plate")
def preview_plate(
    number: int,
    axis: str,
    deltaE: float,
    size: int = 600,
    seed: int | None = None,
):
    """
    브라우저/Swagger에서 바로 PNG 이미지로 볼 수 있는 엔드포인트
    예: /preview_plate?number=12&axis=deutan&deltaE=35
    """
    try:
        rng, _ = make_rng(seed)
        img = generate_ishihara_plate(
            number=number,
            axis=axis,
            deltaE=deltaE,
            rng=rng,
            size=size,
        )

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        return Response(content=png_bytes, media_type="image/png")

    except Exception as e:
        logger.error(f"❌ preview_plate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate preview plate")

# -------------------------
# 헬스체크
# -------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Ishihara FastAPI server running"}
