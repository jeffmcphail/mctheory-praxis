"""
start_mcb_studio.py
====================
Convenience launcher for MCb Backtest Studio.
Run from praxis project root:

    python start_mcb_studio.py

Then in a second terminal:
    cd gui/mcb_studio/frontend
    npm install   (first time only)
    npm run dev

Open http://localhost:5173
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

def main():
    backend_dir  = ROOT / "gui" / "mcb_studio" / "backend"
    frontend_dir = ROOT / "gui" / "mcb_studio" / "frontend"

    print("MCb Backtest Studio")
    print("=" * 50)
    print(f"Backend:  http://localhost:8000")
    print(f"Frontend: http://localhost:5173  (start separately)")
    print()
    print("Starting FastAPI backend...")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=backend_dir,
    )

    print(f"Backend PID {proc.pid} running.")
    print()
    print("Start the frontend in a second terminal:")
    print(f"  cd {frontend_dir}")
    print(f"  npm install   # first time only")
    print(f"  npm run dev")
    print()
    print("Ctrl+C to stop.")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nStopped.")


if __name__ == "__main__":
    main()
