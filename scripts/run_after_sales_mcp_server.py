from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_SRC = ROOT / "mcp_services" / "after_sales_server" / "src"
sys.path.insert(0, str(SERVER_SRC))

from after_sales_mcp.app import main, mcp  # noqa: E402


if __name__ == "__main__":
    if "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        main()
