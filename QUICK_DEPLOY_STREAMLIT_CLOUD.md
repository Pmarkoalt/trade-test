# Quick Deploy: Streamlit Cloud (Recommended)

## ✅ **No API Keys Required for Basic Deployment!**

The dashboard can run with your existing backtest results - no API keys needed. API keys are only optional if you want to fetch live data/news.

## Step-by-Step Setup

### 1. Install Streamlit Locally (Optional - to test first)

```bash
./scripts/install_streamlit.sh
# Or: pip install -e ".[visualization]"
```

### 2. Push Your Code to GitHub

If not already:
```bash
git add .
git commit -m "Add dashboard deployment config"
git push origin main
```

### 3. Deploy to Streamlit Cloud

1. **Go to**: https://share.streamlit.io
2. **Sign in** with your GitHub account
3. **Click**: "New app"
4. **Select**:
   - Repository: `your-username/trade-test` (or your repo name)
   - Branch: `main` (or your default branch)
   - Main file path: `trading_system/reporting/dashboard.py`
   - App URL: (choose a name, e.g., `trading-dashboard`)
   
5. **Advanced Settings** (click "Advanced settings"):
   - Python version: 3.11
   - Command (optional - leave default):
     ```
     streamlit run trading_system/reporting/dashboard.py --server.port $PORT --server.address 0.0.0.0
     ```

6. **Click**: "Deploy"

### 4. Configure Secrets (Optional - only if you want live data)

If you want live data/news features, go to app settings and add secrets:

```
MASSIVE_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
NEWSAPI_API_KEY=your_key_here
SENDGRID_API_KEY=your_key_here
```

**But you don't need these for basic dashboard!**

## For Trading Assistant Dashboard (Alternative)

If you want to deploy the full trading assistant dashboard instead:

1. Follow steps above, but use:
   - Main file path: `trading_system/dashboard/app.py`

2. That's it! Same process.

## Required Files Check

Make sure these exist in your repo:
- ✅ `trading_system/reporting/dashboard.py` - Backtest results dashboard
- ✅ `trading_system/dashboard/app.py` - Trading assistant dashboard  
- ✅ `requirements.txt` or `pyproject.toml` - Dependencies

## What You Get

- ✅ Free hosting
- ✅ Automatic HTTPS
- ✅ Auto-deploy on git push (if enabled)
- ✅ Public URL: `https://your-app-name.streamlit.app`
- ✅ No API keys needed for basic functionality

## Troubleshooting

**"Module not found" errors:**
- Make sure `requirements.txt` includes all dependencies
- Streamlit Cloud uses `requirements.txt` automatically

**"Results not found":**
- The dashboard needs your `results/` directory
- Either commit results to GitHub (small datasets only)
- Or configure data storage (see DASHBOARD_DEPLOYMENT.md)

**Need help?**
- Check Streamlit Cloud docs: https://docs.streamlit.io/streamlit-community-cloud
- See DASHBOARD_DEPLOYMENT.md for alternative platforms

