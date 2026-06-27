from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter(tags=["ops"])

OPS_CONSOLE_PATH = Path(__file__).resolve().parents[1] / "web" / "ops_console.html"
"""运营验证台 HTML 文件路径。"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
@router.get("/ops", response_class=HTMLResponse, include_in_schema=False)
async def ops_console() -> HTMLResponse:
    """返回本地运营验证台页面。

    一个函数可以挂多个路由装饰器，所以 `/` 和 `/ops` 都能打开同一个页面。
    """
    return HTMLResponse(OPS_CONSOLE_PATH.read_text(encoding="utf-8"))
