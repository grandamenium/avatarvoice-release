"""Vercel serverless function entry point for AvatarVoice API.

Lightweight version optimized for Vercel's size limits.
"""

import os
import httpx
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# Settings from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VIBEVOICE_URL = os.getenv("VIBEVOICE_URL")


# Models
class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, str]


class AnalysisResult(BaseModel):
    estimated_age: int
    age_range: List[int]
    gender: str
    race: str
    emotion: str
    confidence: float


class VoiceActor(BaseModel):
    actor_id: str
    name: str
    gender: str
    race: str
    age: int
    emotions: List[str] = ["neutral", "happy", "sad", "angry", "fearful", "disgusted"]


class VoiceMatch(BaseModel):
    actor_id: str
    actor_name: str
    gender: str
    race: str
    age: int
    match_score: float
    sample_url: Optional[str] = None


class GenerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    actor_id: str
    emotion: str = "neutral"
    cfg_scale: float = Field(default=2.0, ge=0.1, le=10.0)
    inference_steps: int = Field(default=32, ge=1, le=100)


class GenerationResponse(BaseModel):
    status: str
    audio_url: Optional[str] = None
    message: Optional[str] = None
    params: Dict[str, Any]


# Sample voice database (in production, this would come from a database)
VOICE_DATABASE = [
    {"actor_id": "1001_DFA", "name": "Actor 1001", "gender": "Female", "race": "African American", "age": 28},
    {"actor_id": "1002_IWL", "name": "Actor 1002", "gender": "Male", "race": "Caucasian", "age": 35},
    {"actor_id": "1003_IOM", "name": "Actor 1003", "gender": "Male", "race": "Caucasian", "age": 45},
    {"actor_id": "1004_MTI", "name": "Actor 1004", "gender": "Female", "race": "Asian", "age": 32},
    {"actor_id": "1005_TSI", "name": "Actor 1005", "gender": "Male", "race": "African American", "age": 40},
    {"actor_id": "1006_TIE", "name": "Actor 1006", "gender": "Female", "race": "Caucasian", "age": 25},
    {"actor_id": "1007_DFA", "name": "Actor 1007", "gender": "Male", "race": "Hispanic", "age": 38},
    {"actor_id": "1008_IWL", "name": "Actor 1008", "gender": "Female", "race": "Asian", "age": 29},
    {"actor_id": "1009_IOM", "name": "Actor 1009", "gender": "Male", "race": "African American", "age": 50},
    {"actor_id": "1010_MTI", "name": "Actor 1010", "gender": "Female", "race": "Caucasian", "age": 42},
]


# Create FastAPI app
app = FastAPI(
    title="AvatarVoice API",
    version="1.0.0",
    description="API for avatar voice matching and TTS generation",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to AvatarVoice API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /api/v1/analyze",
            "voices": "GET /api/v1/voices",
            "match": "POST /api/v1/match",
            "generate": "POST /api/v1/generate"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services={
            "api": "running",
            "gemini": "configured" if GEMINI_API_KEY else "not_configured",
            "vibevoice": "configured" if VIBEVOICE_URL else "not_configured"
        }
    )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check_v1():
    """V1 Health check endpoint."""
    return await health_check()


@app.get("/api/v1/voices", response_model=Dict[str, Any])
async def list_voices(
    gender: Optional[str] = Query(None, description="Filter by gender (Male/Female)"),
    race: Optional[str] = Query(None, description="Filter by race"),
    age_min: Optional[int] = Query(None, ge=18, le=100, description="Minimum age"),
    age_max: Optional[int] = Query(None, ge=18, le=100, description="Maximum age"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """List available voice actors with optional filters."""
    filtered = VOICE_DATABASE.copy()

    if gender:
        filtered = [v for v in filtered if v["gender"].lower() == gender.lower()]
    if race:
        filtered = [v for v in filtered if v["race"].lower() == race.lower()]
    if age_min:
        filtered = [v for v in filtered if v["age"] >= age_min]
    if age_max:
        filtered = [v for v in filtered if v["age"] <= age_max]

    total = len(filtered)
    voices = filtered[offset:offset + limit]

    return {
        "voices": [
            VoiceActor(**v).model_dump() for v in voices
        ],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.post("/api/v1/analyze", response_model=AnalysisResult)
async def analyze_image(
    image: Optional[UploadFile] = File(None, description="Image file to analyze"),
    image_url: Optional[str] = Form(None, description="URL of image to analyze")
):
    """Analyze an avatar image to extract demographics using Gemini Vision API."""
    if not image and not image_url:
        raise HTTPException(status_code=400, detail="Either image file or image_url is required")

    if not GEMINI_API_KEY:
        # Return mock data if Gemini not configured
        return AnalysisResult(
            estimated_age=30,
            age_range=[25, 35],
            gender="male",
            race="caucasian",
            emotion="neutral",
            confidence=0.85
        )

    # In production, this would call the Gemini Vision API
    # For now, return mock data
    return AnalysisResult(
        estimated_age=30,
        age_range=[25, 35],
        gender="male",
        race="caucasian",
        emotion="neutral",
        confidence=0.85
    )


@app.post("/api/v1/match", response_model=Dict[str, Any])
async def match_voices(
    gender: str = Form(..., description="Target gender"),
    race: str = Form(..., description="Target race/ethnicity"),
    age: int = Form(..., ge=18, le=100, description="Target age"),
    emotion: Optional[str] = Form(None, description="Target emotion"),
    limit: int = Form(5, ge=1, le=20, description="Maximum matches to return")
):
    """Find voice actors matching the given demographics."""
    matches = []

    for voice in VOICE_DATABASE:
        # Calculate match score based on demographics
        score = 0.0

        # Gender match (40% weight)
        if voice["gender"].lower() == gender.lower():
            score += 0.4

        # Race match (30% weight)
        if voice["race"].lower() == race.lower():
            score += 0.3

        # Age proximity (30% weight)
        age_diff = abs(voice["age"] - age)
        if age_diff <= 5:
            score += 0.3
        elif age_diff <= 10:
            score += 0.2
        elif age_diff <= 15:
            score += 0.1

        if score > 0:
            matches.append(VoiceMatch(
                actor_id=voice["actor_id"],
                actor_name=voice["name"],
                gender=voice["gender"],
                race=voice["race"],
                age=voice["age"],
                match_score=round(score, 2)
            ))

    # Sort by match score descending
    matches.sort(key=lambda x: x.match_score, reverse=True)
    matches = matches[:limit]

    return {
        "matches": [m.model_dump() for m in matches],
        "query": {
            "gender": gender,
            "race": race,
            "age": age,
            "emotion": emotion
        }
    }


@app.post("/api/v1/generate", response_model=GenerationResponse)
async def generate_speech(request: GenerationRequest):
    """Generate speech using VibeVoice TTS with the specified voice actor."""
    if not VIBEVOICE_URL:
        return GenerationResponse(
            status="error",
            message="VibeVoice TTS endpoint not configured. Set VIBEVOICE_URL environment variable.",
            params={
                "text": request.text,
                "actor_id": request.actor_id,
                "emotion": request.emotion,
                "cfg_scale": request.cfg_scale,
                "inference_steps": request.inference_steps
            }
        )

    # In production, this would call the VibeVoice TTS API
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                VIBEVOICE_URL,
                json={
                    "text": request.text,
                    "speaker_id": request.actor_id,
                    "emotion": request.emotion,
                    "cfg_scale": request.cfg_scale,
                    "inference_steps": request.inference_steps
                }
            )

            if response.status_code == 200:
                result = response.json()
                return GenerationResponse(
                    status="success",
                    audio_url=result.get("audio_url"),
                    params={
                        "text": request.text,
                        "actor_id": request.actor_id,
                        "emotion": request.emotion,
                        "cfg_scale": request.cfg_scale,
                        "inference_steps": request.inference_steps
                    }
                )
            else:
                return GenerationResponse(
                    status="error",
                    message=f"TTS API returned status {response.status_code}",
                    params={
                        "text": request.text,
                        "actor_id": request.actor_id,
                        "emotion": request.emotion,
                        "cfg_scale": request.cfg_scale,
                        "inference_steps": request.inference_steps
                    }
                )
    except Exception as e:
        return GenerationResponse(
            status="error",
            message=str(e),
            params={
                "text": request.text,
                "actor_id": request.actor_id,
                "emotion": request.emotion,
                "cfg_scale": request.cfg_scale,
                "inference_steps": request.inference_steps
            }
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Handler for Vercel
handler = app
