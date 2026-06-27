from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析服务启动参数。"""
    parser = argparse.ArgumentParser(description="Start the FastAPI application server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-reload", action="store_true")
    return parser.parse_args()


def main() -> None:
    """使用当前 Python 解释器启动开发服务器。

    这样可以避免误用全局安装的 uvicorn，确保依赖来自当前虚拟环境。
    """

    args = parse_args()
    os.chdir(ROOT)
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "customer_service_app.main:app",
        "--app-dir",
        "src",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if not args.no_reload:
        command.extend(["--reload", "--reload-dir", "src"])
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
