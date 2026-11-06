"""``python -m memory_rl.web [--port 8000]``: serve the demo API and built UI."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Noesis demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("memory_rl.web.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
