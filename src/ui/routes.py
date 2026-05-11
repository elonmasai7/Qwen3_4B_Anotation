from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "active": "dashboard", "title": "Dashboard"},
    )


@router.get("/annotation", response_class=HTMLResponse)
async def annotation_page(request: Request):
    return templates.TemplateResponse(
        "annotation.html",
        {"request": request, "active": "annotation", "title": "Annotation"},
    )


@router.get("/datasets", response_class=HTMLResponse)
async def datasets_page(request: Request):
    return templates.TemplateResponse(
        "datasets.html",
        {"request": request, "active": "datasets", "title": "Datasets"},
    )


@router.get("/experiments", response_class=HTMLResponse)
async def experiments_page(request: Request):
    return templates.TemplateResponse(
        "experiments.html",
        {"request": request, "active": "experiments", "title": "Experiments"},
    )


@router.get("/prompts", response_class=HTMLResponse)
async def prompts_page(request: Request):
    return templates.TemplateResponse(
        "prompts.html",
        {"request": request, "active": "prompts", "title": "Prompts"},
    )


@router.get("/leaderboard", response_class=HTMLResponse)
async def leaderboard_page(request: Request):
    return templates.TemplateResponse(
        "leaderboard.html",
        {"request": request, "active": "leaderboard", "title": "Leaderboard"},
    )
