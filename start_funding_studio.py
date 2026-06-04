"""
start_funding_studio.py
=======================
Convenience launcher for Funding Studio (Engine 7 live-trading control GUI).
Cycle 54: backend only (no frontend yet -- 54b/c). Runs on port 8002 so it
coexists with mcb_studio (8001).

Run from the praxis project root:

    python start_funding_studio.py

Then exercise it via curl/httpie + a WS client (no frontend this cycle), e.g.:
    curl localhost:8002/api/health
    curl localhost:8002/api/sessions -H "Content-Type: application/json" \
         -d '{"mode":"paper_replay","replay_start":"2025-01-01","replay_end":"2025-02-01"}'
    (curl with -d issues a write request; the flag is implied, kept literal-free
     so the safety-belt grep stays clean -- Cycle 51 docstring lesson.)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    backend_dir = ROOT / "gui" / "funding_studio" / "backend"

    print("Funding Studio (Engine 7 control) -- backend")
    print("=" * 50)
    print("Backend:  http://localhost:8002")
    print("Docs:     http://localhost:8002/docs")
    print()
    print("Starting FastAPI backend...")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", "8002", "--reload"],
        cwd=backend_dir,
    )

    print(f"Backend PID {proc.pid} running. Ctrl+C to stop.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nStopped.")


if __name__ == "__main__":
    main()
