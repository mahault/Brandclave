"""The public Sentient section: /sentient, a page of its own, with recorded simulation runs.

The page plays back runs recorded by `python -m sentient.rich.record_site`; nothing here runs
the model, so the web service needs none of the model's dependencies.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

router = APIRouter()
SITE = Path(__file__).resolve().parents[2] / "web" / "sentient"
DATA_FILES = {"spa_evening", "lounge_ac", "lounge_distress", "lobby_rush", "results"}


@router.get("/sentient", include_in_schema=False)
async def sentient_page():
    return FileResponse(SITE / "index.html", media_type="text/html")


@router.get("/sentient/", include_in_schema=False)
async def sentient_slash():
    return RedirectResponse(url="/sentient", status_code=308)


@router.get("/sentient/data/{name}.json", include_in_schema=False)
async def sentient_data(name: str):
    if name not in DATA_FILES or not (SITE / "data" / f"{name}.json").exists():
        raise HTTPException(404)
    return FileResponse(SITE / "data" / f"{name}.json", media_type="application/json",
                        headers={"Cache-Control": "public, max-age=300"})
