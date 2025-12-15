"""FastAPI application for BrandClave Aggregator."""

import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create FastAPI app
app = FastAPI(
    title="BrandClave Aggregator API",
    description="API for hospitality demand intelligence, trend signals, and hotelier moves",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.2.0"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "BrandClave Aggregator API",
        "version": "0.3.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "social_pulse": "/api/social-pulse",
            "hotelier_bets": "/api/hotelier-bets",
        },
    }


# Import and include routers
from api.routes.social_pulse import router as social_pulse_router
from api.routes.hotelier_bets import router as hotelier_bets_router

app.include_router(social_pulse_router, prefix="/api", tags=["Social Pulse"])
app.include_router(hotelier_bets_router, prefix="/api", tags=["Hotelier Bets"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
