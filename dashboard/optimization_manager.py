"""
EC2 Optimization Manager Dashboard

A Streamlit dashboard to manage and monitor optimization runs:
- Start/stop optimizations
- Monitor progress in real-time
- View results and compare strategies
- Manage PostgreSQL storage
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="Trading System - Optimization Manager",
    page_icon="📈",
    layout="wide",
)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "optimization_results"
OVERNIGHT_DIR = PROJECT_ROOT / "overnight_results"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def get_running_processes():
    """Get list of running optimization processes."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "overnight_optimization|strategy_opt"],
            capture_output=True,
            text=True,
        )
        pids = result.stdout.strip().split("\n") if result.stdout.strip() else []
        return [p for p in pids if p]
    except Exception:
        return []


def get_process_info(pid):
    """Get info about a running process."""
    try:
        result = subprocess.run(
            ["ps", "-p", pid, "-o", "pid,etime,pcpu,pmem,args"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def start_optimization(config, trials, jobs, use_postgres=True):
    """Start a new optimization run."""
    cmd = [
        sys.executable,
        "scripts/overnight_optimization.py",
        "--trials",
        str(trials),
        "--jobs",
        str(jobs),
        "--equity-config",
        config,
    ]

    if use_postgres:
        postgres_url = os.environ.get("OPTUNA_STORAGE_URL", "postgresql://optuna:optuna123@localhost/optuna_db")
        cmd.extend(["--postgres", postgres_url])

    # Start in background
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(PROJECT_ROOT),
        start_new_session=True,
    )

    return process.pid


def stop_optimization():
    """Stop all running optimization processes."""
    subprocess.run(["pkill", "-f", "overnight_optimization"], capture_output=True)
    subprocess.run(["pkill", "-f", "strategy_opt"], capture_output=True)


def get_optuna_progress():
    """Get optimization progress from PostgreSQL."""
    try:
        import optuna

        postgres_url = os.environ.get("OPTUNA_STORAGE_URL", "postgresql://optuna:optuna123@localhost/optuna_db")

        # Get all studies
        studies = optuna.study.get_all_study_summaries(storage=postgres_url)

        progress = []
        for study in studies:
            progress.append(
                {
                    "name": study.study_name,
                    "n_trials": study.n_trials,
                    "best_value": study.best_trial.value if study.best_trial else None,
                    "datetime_start": study.datetime_start,
                }
            )

        return progress
    except Exception as e:
        return [{"error": str(e)}]


def get_latest_results():
    """Get latest optimization results."""
    results = []

    for results_dir in [RESULTS_DIR, OVERNIGHT_DIR]:
        if not results_dir.exists():
            continue

        for json_file in results_dir.glob("**/*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                results.append(
                    {
                        "file": str(json_file.relative_to(PROJECT_ROOT)),
                        "best_value": data.get("best_value", 0),
                        "n_completed": data.get("n_completed", 0),
                        "study_name": data.get("study_name", ""),
                        "modified": datetime.fromtimestamp(json_file.stat().st_mtime),
                    }
                )
            except Exception:
                continue

    return sorted(results, key=lambda x: x["modified"], reverse=True)[:20]


def get_config_files():
    """Get available strategy config files."""
    configs = []
    if CONFIGS_DIR.exists():
        for yaml_file in CONFIGS_DIR.glob("*.yaml"):
            configs.append(str(yaml_file))
    return configs


# Sidebar
st.sidebar.title("⚙️ Controls")

# Status section
st.sidebar.subheader("📊 Status")
running_pids = get_running_processes()

if running_pids:
    st.sidebar.success(f"🟢 {len(running_pids)} process(es) running")
    if st.sidebar.button("🛑 Stop All", type="primary"):
        stop_optimization()
        st.rerun()
else:
    st.sidebar.info("🔴 No optimization running")

# Start new optimization
st.sidebar.subheader("🚀 Start Optimization")

configs = get_config_files()
selected_config = st.sidebar.selectbox(
    "Strategy Config",
    configs,
    index=0 if configs else None,
)

trials = st.sidebar.number_input("Trials", min_value=10, max_value=10000, value=1000, step=100)
jobs = st.sidebar.number_input("Parallel Jobs", min_value=1, max_value=16, value=7)
use_postgres = st.sidebar.checkbox("Use PostgreSQL", value=True)

if st.sidebar.button("▶️ Start Optimization", disabled=len(running_pids) > 0):
    if selected_config:
        pid = start_optimization(selected_config, trials, jobs, use_postgres)
        st.sidebar.success(f"Started! PID: {pid}")
        st.rerun()

# Main content
st.title("📈 Trading System - Optimization Manager")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Progress", "📁 Results", "⚡ Live Study"])

with tab1:
    st.subheader("Running Processes")

    if running_pids:
        for pid in running_pids:
            info = get_process_info(pid)
            if info:
                st.code(info)
    else:
        st.info("No optimization processes currently running.")

    # PostgreSQL progress
    st.subheader("Optuna Studies (PostgreSQL)")

    progress = get_optuna_progress()
    if progress and "error" not in progress[0]:
        for study in progress:
            col1, col2, col3 = st.columns(3)
            col1.metric("Study", study["name"][:30])
            col2.metric("Trials", study["n_trials"])
            col3.metric("Best Sharpe", f"{study['best_value']:.4f}" if study["best_value"] else "N/A")
    else:
        st.warning("Could not connect to PostgreSQL or no studies found.")

with tab2:
    st.subheader("Recent Results")

    results = get_latest_results()

    if results:
        for result in results:
            with st.expander(f"📄 {result['file']} - Sharpe: {result['best_value']:.4f}"):
                st.write(f"**Trials completed:** {result['n_completed']}")
                st.write(f"**Study:** {result['study_name']}")
                st.write(f"**Modified:** {result['modified']}")

                # Load full results
                try:
                    with open(PROJECT_ROOT / result["file"]) as f:
                        full_data = json.load(f)

                    if "best_params" in full_data:
                        st.write("**Best Parameters:**")
                        st.json(full_data["best_params"])
                except Exception:
                    pass
    else:
        st.info("No results found yet.")

with tab3:
    st.subheader("Live Study Viewer")

    try:
        import optuna

        postgres_url = os.environ.get("OPTUNA_STORAGE_URL", "postgresql://optuna:optuna123@localhost/optuna_db")

        studies = optuna.study.get_all_study_summaries(storage=postgres_url)

        if studies:
            study_names = [s.study_name for s in studies]
            selected_study = st.selectbox("Select Study", study_names)

            if selected_study:
                study = optuna.load_study(study_name=selected_study, storage=postgres_url)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Trials", len(study.trials))
                col2.metric("Best Value", f"{study.best_value:.4f}" if study.best_trial else "N/A")
                col3.metric("Complete", len([t for t in study.trials if t.state.name == "COMPLETE"]))
                col4.metric("Failed", len([t for t in study.trials if t.state.name == "FAIL"]))

                if study.best_params:
                    st.write("**Best Parameters:**")
                    st.json(study.best_params)

                # Trial history
                st.write("**Trial History (last 20):**")
                trial_data = []
                for trial in study.trials[-20:]:
                    trial_data.append(
                        {
                            "number": trial.number,
                            "value": trial.value,
                            "state": trial.state.name,
                        }
                    )
                st.dataframe(trial_data, use_container_width=True)
        else:
            st.info("No studies found in PostgreSQL.")

    except Exception as e:
        st.error(f"Error connecting to PostgreSQL: {e}")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Auto-refresh
if running_pids:
    st.caption("⏳ Auto-refreshing every 30 seconds while optimization is running...")
    import time

    time.sleep(30)
    st.rerun()
