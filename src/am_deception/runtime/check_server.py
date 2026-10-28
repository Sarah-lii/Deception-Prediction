#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import urllib.request


def main() -> None:
    port = sys.argv[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
        payload = json.loads(response.read().decode())
    if not payload.get("ok"):
        raise SystemExit(1)
    print(json.dumps({"event": "server_health", **payload}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
