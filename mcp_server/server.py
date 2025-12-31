"""MCP Server for Trading System."""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import subprocess  # noqa: S404 - subprocess needed for running trading system CLI
import json
import os
from pathlib import Path

app = FastAPI(
    title="Trading System MCP Server",
    version="1.0.0",
    description="MCP Server for Trading System - Enables Claude integration"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path("/app")
security = HTTPBearer(auto_error=False)


class BacktestRequest(BaseModel):
    config_path: str
    period: str = "train"  # train, validation, holdout


class ValidateRequest(BaseModel):
    config_path: str


class HealthResponse(BaseModel):
    status: str
    version: str


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """Verify authentication token if required."""
    api_token = os.getenv("MCP_API_TOKEN")
    if api_token:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if credentials.credentials != api_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return credentials.credentials if credentials else None


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Trading System MCP Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "backtest": "/backtest",
            "validate": "/validate",
            "configs": "/configs",
            "results": "/results/{run_id}",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/backtest", tags=["Backtest"])
async def run_backtest(request: BacktestRequest, token: Optional[str] = Depends(verify_token)):
    """Run a backtest.
    
    Args:
        request: Backtest request with config path and period
        token: Authentication token (if required)
        
    Returns:
        Backtest results and output
    """
    config_full_path = BASE_DIR / request.config_path.lstrip("/")
    
    if not config_full_path.exists():
        raise HTTPException(status_code=404, detail=f"Config file not found: {request.config_path}")
    
    if request.period not in ["train", "validation", "holdout"]:
        raise HTTPException(status_code=400, detail=f"Invalid period: {request.period}. Must be train, validation, or holdout")
    
    try:
        result = subprocess.run(  # noqa: S603 - subprocess needed for CLI invocation
            [
                "python", "-m", "trading_system", "backtest",
                "--config", str(config_full_path),
                "--period", request.period
            ],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(BASE_DIR)
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Backtest failed: {result.stderr}"
            )
        
        return {
            "status": "success",
            "period": request.period,
            "config_path": request.config_path,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Backtest timed out after 1 hour")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")


@app.post("/validate", tags=["Validation"])
async def run_validation(request: ValidateRequest, token: Optional[str] = Depends(verify_token)):
    """Run validation suite.
    
    Args:
        request: Validation request with config path
        token: Authentication token (if required)
        
    Returns:
        Validation results and output
    """
    config_full_path = BASE_DIR / request.config_path.lstrip("/")
    
    if not config_full_path.exists():
        raise HTTPException(status_code=404, detail=f"Config file not found: {request.config_path}")
    
    try:
        result = subprocess.run(  # noqa: S603 - subprocess needed for CLI invocation
            [
                "python", "-m", "trading_system", "validate",
                "--config", str(config_full_path)
            ],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(BASE_DIR)
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Validation failed: {result.stderr}"
            )
        
        return {
            "status": "success",
            "config_path": request.config_path,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Validation timed out after 1 hour")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


@app.get("/configs", tags=["Config"])
async def list_configs(token: Optional[str] = Depends(verify_token)):
    """List available configuration files.
    
    Args:
        token: Authentication token (if required)
        
    Returns:
        List of available configuration files
    """
    config_dirs = [
        BASE_DIR / "configs",
        BASE_DIR / "EXAMPLE_CONFIGS",
        BASE_DIR / "tests" / "fixtures" / "configs"
    ]
    
    configs = []
    for config_dir in config_dirs:
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                configs.append({
                    "name": config_file.name,
                    "path": str(config_file.relative_to(BASE_DIR))
                })
    
    return {"configs": configs, "count": len(configs)}


@app.get("/results/{run_id}", tags=["Results"])
async def get_results(
    run_id: str,
    period: Optional[str] = None,
    token: Optional[str] = Depends(verify_token)
):
    """Get backtest results.
    
    Args:
        run_id: Backtest run ID
        period: Optional period filter (train, validation, holdout)
        token: Authentication token (if required)
        
    Returns:
        Backtest results for the specified run
    """
    results_dir = BASE_DIR / "results" / run_id
    
    if not results_dir.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for run_id: {run_id}")
    
    if period:
        if period not in ["train", "validation", "holdout"]:
            raise HTTPException(status_code=400, detail=f"Invalid period: {period}")
        results_dir = results_dir / period
        
        if not results_dir.exists():
            raise HTTPException(status_code=404, detail=f"Period not found: {period}")
    else:
        # Return all periods
        periods = {}
        for p in ["train", "validation", "holdout"]:
            period_dir = BASE_DIR / "results" / run_id / p
            if period_dir.exists():
                periods[p] = {}
                for result_file in period_dir.glob("*.json"):
                    try:
                        with open(result_file, "r") as f:
                            periods[p][result_file.stem] = json.load(f)
                    except Exception as e:
                        periods[p][result_file.stem] = {"error": str(e)}
        return {"run_id": run_id, "periods": periods}
    
    results = {}
    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file, "r") as f:
                results[result_file.stem] = json.load(f)
        except Exception as e:
            results[result_file.stem] = {"error": str(e)}
    
    return {"run_id": run_id, "period": period, "results": results}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("MCP_HOST", "0.0.0.0")  # noqa: S104 - binding configurable via env var
    port = int(os.getenv("MCP_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

