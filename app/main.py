from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
from dotenv import load_dotenv
load_dotenv()
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.routes import router
from app.core.index import build_or_load_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading catalog index…")
    build_or_load_index()
    yield
    logger.info("Shutting down.")

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({(time.time()-t0)*1000:.0f}ms)")
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(status_code=200, content={"reply": "I encountered an error. Please try again.", "recommendations": [], "end_of_conversation": False})

app.include_router(router)