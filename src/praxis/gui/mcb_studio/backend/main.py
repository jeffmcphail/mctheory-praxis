"""
gui/mcb_studio/backend/main.py
================================
FastAPI + WebSocket server for MCb Backtest Studio.

Imports:
  - praxis.indicators.market_cipher_b  (src/praxis/indicators/)
  - engines.backtest_engine             (engines/)
  - engines.mcb_strategies             (engines/mcb_strategies/)

Run from praxis project root:
    uvicorn gui.mcb_studio.backend.main:app --host 0.0.0.0 --port 8000 --reload

Or use the convenience launcher:
    python start_mcb_studio.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so engines/ is importable
_ROOT = Path(__file__).resolve().parents[3]   # gui/mcb_studio/backend/ -> root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import asyncio
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_feed import fetch_ohlcv, SUPPORTED_SYMBOLS, SUPPORTED_INTERVALS
from praxis.indicators.market_cipher_b import MarketCipherB
from engines.backtest_engine import run_backtest_streaming
from engines.mcb_strategies import get_strategy, list_strategies

app = FastAPI(title="MCb Backtest Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, dict] = {}


class BacktestRequest(BaseModel):
    symbol: str = "BTC/USDT"
    interval: str = "4h"
    start: str = "2024-01-01"
    end: str = "2025-01-01"
    strategy: str = "anchor_trigger"
    params: dict = {}
    replay_speed_ms: int = 30


@app.get("/api/strategies")
def api_strategies():
    return list_strategies()

@app.get("/api/symbols")
def api_symbols():
    return SUPPORTED_SYMBOLS

@app.get("/api/intervals")
def api_intervals():
    return list(SUPPORTED_INTERVALS.keys())

@app.post("/api/backtest/start")
def api_backtest_start(req: BacktestRequest):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"request": req.model_dump(), "status": "pending"}
    return {"session_id": session_id}


@app.websocket("/ws/{session_id}")
async def ws_backtest(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in sessions:
        await _send(websocket, {"type": "error", "message": "Invalid session ID"})
        await websocket.close()
        return

    req = BacktestRequest(**sessions[session_id]["request"])

    try:
        await _send(websocket, {
            "type": "status",
            "message": f"Fetching {req.symbol} {req.interval} data…",
        })

        df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: fetch_ohlcv(req.symbol, req.interval, req.start, req.end)
        )

        await _send(websocket, {
            "type": "status",
            "message": f"Loaded {len(df)} bars. Calculating MCb indicators…",
        })

        mcb = MarketCipherB()
        df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: mcb.calculate(df)
        )

        await _send(websocket, {
            "type": "status",
            "message": f"Running {req.strategy} strategy…",
            "total_bars": len(df),
        })

        strategy = get_strategy(req.strategy, req.params)

        async for frame in run_backtest_streaming(df, strategy, req.replay_speed_ms):
            await _send(websocket, frame)

        await _send(websocket, {"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await _send(websocket, {"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        sessions.pop(session_id, None)


async def _send(ws: WebSocket, data: dict):
    try:
        await ws.send_json(data)
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True,
                reload_dirs=[str(_ROOT / "engines"), str(_ROOT / "src")])
