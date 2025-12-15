"""FastAPI application for BrandClave Aggregator."""

import logging
import os
import sys
from contextlib import asynccontextmanager
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

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    logger.info("Starting BrandClave Aggregator API...")

    # Initialize scheduler if enabled
    if os.getenv("SCHEDULER_ENABLED", "true").lower() == "true":
        try:
            from scheduler.scheduler import init_scheduler
            scheduler = init_scheduler(auto_register=True)
            if scheduler.is_available:
                scheduler.start()
                logger.info("Scheduler started successfully")
        except Exception as e:
            logger.warning(f"Could not start scheduler: {e}")

    yield

    # Shutdown
    logger.info("Shutting down BrandClave Aggregator API...")

    # Stop scheduler
    try:
        from scheduler.scheduler import get_scheduler
        scheduler = get_scheduler()
        if scheduler.is_running:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")
    except Exception:
        pass


# Create FastAPI app
app = FastAPI(
    title="BrandClave Aggregator API",
    description="API for hospitality demand intelligence, trend signals, and hotelier moves",
    version="0.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
    return {"status": "healthy", "version": "0.5.0"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "BrandClave Aggregator API",
        "version": "0.5.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "social_pulse": "/api/social-pulse",
            "hotelier_bets": "/api/hotelier-bets",
            "demand_scan": "/api/demand-scan",
            "scheduler": "/api/scheduler/status",
            "monitoring": "/api/monitoring/dashboard",
        },
    }


# Import and include routers
from api.routes.social_pulse import router as social_pulse_router
from api.routes.hotelier_bets import router as hotelier_bets_router
from api.routes.demand_scan import router as demand_scan_router
from api.routes.scheduler import router as scheduler_router
from api.routes.monitoring import router as monitoring_router

app.include_router(social_pulse_router, prefix="/api", tags=["Social Pulse"])
app.include_router(hotelier_bets_router, prefix="/api", tags=["Hotelier Bets"])
app.include_router(demand_scan_router, prefix="/api", tags=["Demand Scan"])
app.include_router(scheduler_router, prefix="/api", tags=["Scheduler"])
app.include_router(monitoring_router, prefix="/api", tags=["Monitoring"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
