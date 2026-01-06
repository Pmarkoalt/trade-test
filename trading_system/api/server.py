"""
Simple FastAPI wrapper for n8n integration.

Run with: uvicorn trading_system.api.server:app --host 0.0.0.0 --port 8000
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="Trading System API",
    description="API wrapper for n8n workflow integration",
    version="1.0.0",
)

# Track running jobs
_running_jobs: dict = {}


class BacktestRequest(BaseModel):
    config_path: str = "/app/custom_configs/backtest_config_production.yaml"
    period: str = "train"  # train, validation, holdout


class BacktestResponse(BaseModel):
    status: str
    run_id: str
    message: str


def run_backtest_job(config_path: str, period: str, job_id: str):
    """Background task to run backtest."""
    try:
        _running_jobs[job_id]["status"] = "running"

        # Import here to avoid circular imports
        from trading_system.integration.runner import run_backtest

        results = run_backtest(config_path, period)

        _running_jobs[job_id]["status"] = "completed"
        _running_jobs[job_id]["results"] = results
        _running_jobs[job_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        _running_jobs[job_id]["status"] = "failed"
        _running_jobs[job_id]["error"] = str(e)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/backtest", response_model=BacktestResponse)
async def start_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Start a backtest run.

    Returns immediately with a job_id. Poll /job/{job_id} for status.
    """
    job_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    _running_jobs[job_id] = {
        "status": "queued",
        "config_path": request.config_path,
        "period": request.period,
        "started_at": datetime.now().isoformat(),
    }

    background_tasks.add_task(
        run_backtest_job,
        request.config_path,
        request.period,
        job_id
    )

    return BacktestResponse(
        status="queued",
        run_id=job_id,
        message=f"Backtest queued for period '{request.period}'"
    )


@app.post("/backtest/sync")
async def run_backtest_sync(request: BacktestRequest):
    """
    Run backtest synchronously (blocks until complete).

    Use this for shorter runs or when you need immediate results.
    """
    try:
        from trading_system.integration.runner import run_backtest

        results = run_backtest(request.config_path, request.period)

        return {
            "status": "completed",
            "period": request.period,
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a running or completed job."""
    if job_id not in _running_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return _running_jobs[job_id]


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return {
        "jobs": [
            {"job_id": k, "status": v["status"], "started_at": v.get("started_at")}
            for k, v in _running_jobs.items()
        ]
    }


@app.get("/results")
async def list_results(limit: int = 10):
    """List recent result directories."""
    results_dir = Path("/app/results")

    if not results_dir.exists():
        return {"results": []}

    # Get directories sorted by modification time
    dirs = sorted(
        results_dir.iterdir(),
        key=lambda x: x.stat().st_mtime if x.is_dir() else 0,
        reverse=True
    )

    results = []
    for d in dirs[:limit]:
        if d.is_dir() and d.name.startswith("run_"):
            results.append({
                "run_id": d.name,
                "path": str(d),
                "periods": [p.name for p in d.iterdir() if p.is_dir()],
            })

    return {"results": results}


@app.get("/results/{run_id}/{period}/metrics")
async def get_metrics(run_id: str, period: str):
    """Get metrics from a specific run."""
    report_path = Path(f"/app/results/{run_id}/{period}/monthly_report.json")

    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for {run_id}/{period}")

    with open(report_path) as f:
        data = json.load(f)

    return data


@app.get("/results/{run_id}/{period}/equity_curve")
async def get_equity_curve(run_id: str, period: str, limit: Optional[int] = None):
    """Get equity curve data from a specific run."""
    import pandas as pd

    csv_path = Path(f"/app/results/{run_id}/{period}/equity_curve.csv")

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Equity curve not found for {run_id}/{period}")

    df = pd.read_csv(csv_path)

    if limit:
        df = df.tail(limit)

    return df.to_dict(orient="records")


@app.get("/results/{run_id}/{period}/trades")
async def get_trades(run_id: str, period: str, limit: Optional[int] = None):
    """Get trade log from a specific run."""
    import pandas as pd

    csv_path = Path(f"/app/results/{run_id}/{period}/trade_log.csv")

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Trade log not found for {run_id}/{period}")

    df = pd.read_csv(csv_path)

    if limit:
        df = df.tail(limit)

    return df.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
