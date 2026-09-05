"""Main FastAPI Application for IP-SAKTI Sahayak (Milestone 6).

Coordinates between the React Frontend and the LangGraph Orchestrator,
providing REST endpoints for grounded chat, citation inspection, document catalogs,
and system health.
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.api.routes import chat, health, sources
from src.config import API_HOST, API_PORT

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ip_sakti_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and pre-warms the LangGraph coordinator at application startup."""
    logger.info("Initializing IP-SAKTI LangGraph Coordinator...")
    try:
        from src.graph.graph import IPSAKTILangGraphCoordinator
        app.state.coordinator = IPSAKTILangGraphCoordinator()
        logger.info("IP-SAKTI LangGraph Coordinator pre-warmed and ready.")
    except Exception as e:
        logger.error(f"Failed to initialize LangGraph Coordinator at startup: {e}", exc_info=True)
        app.state.coordinator = None
        
    yield
    
    logger.info("Shutting down IP-SAKTI Sahayak API server...")


app = FastAPI(
    title="IP-SAKTI Sahayak API",
    description="Intelligent Multilingual Legal & Ayurvedic IP Research Assistant (SIH)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [
    origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()
] if allowed_origins_env else [
    frontend_url,
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routes
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(sources.router)


@app.get("/")
async def root():
    """Root metadata endpoint."""
    return {
        "service": "IP-SAKTI Sahayak API",
        "description": "AI Assistant for Ayurveda • IP • Traditional Knowledge",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=True)
