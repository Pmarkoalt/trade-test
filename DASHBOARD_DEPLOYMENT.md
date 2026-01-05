# Dashboard Deployment Guide

This guide covers deploying the Trading System dashboard to various cloud platforms.

## Streamlit Installation

Streamlit is an **optional dependency** that needs to be installed separately:

```bash
# Install Streamlit with visualization dependencies
pip install -e ".[visualization]"

# Or install directly
pip install streamlit plotly

# Verify installation
streamlit --version
```

## Platform Comparison

| Platform | Free Tier | Best For | Setup Difficulty | Notes |
|----------|-----------|----------|------------------|-------|
| **Streamlit Cloud** | ✅ Yes | Streamlit apps | ⭐ Easy | Native Streamlit support |
| **Railway** | ✅ Yes (limited) | Docker apps | ⭐⭐ Medium | Great for Docker deployments |
| **Render** | ✅ Yes (with limitations) | Web services | ⭐⭐ Medium | Auto-deploy from Git |
| **Fly.io** | ✅ Yes (generous) | Global deployment | ⭐⭐⭐ Medium-Hard | Fast global network |
| **Replit** | ✅ Yes (limited) | Quick prototypes | ⭐ Easy | Not ideal for production |
| **Heroku** | ❌ No (paid only) | Traditional PaaS | ⭐⭐ Medium | More expensive |
| **AWS/GCP/Azure** | ⚠️ Free tier | Full control | ⭐⭐⭐ Hard | Most flexible, most complex |

## Recommended Platforms

### 🏆 Best for Quick Start: Streamlit Cloud (FREE)

Streamlit Cloud is specifically designed for Streamlit apps and is the easiest option.

#### Setup Steps:

1. **Push your code to GitHub** (make sure it's public or use Streamlit Cloud Teams)

2. **Create a requirements file for Streamlit Cloud**:

   Create `.streamlit/config.toml`:
   ```toml
   [server]
   port = 8501
   address = "0.0.0.0"
   ```

   Create `requirements.txt` with:
   ```txt
   streamlit>=1.28.0
   plotly>=5.0.0
   pandas>=1.5.0
   numpy>=1.24.0
   scipy>=1.9.0
   pydantic>=1.10.0
   pyyaml>=6.0
   matplotlib>=3.5.0
   ```

3. **Connect to Streamlit Cloud**:
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `trading_system/reporting/dashboard.py`
   - Set command: `streamlit run trading_system/reporting/dashboard.py --server.port $PORT --server.address 0.0.0.0 -- --base_path results --run_id <your_run_id>`

**Pros:**
- ✅ Free for public repos
- ✅ Native Streamlit support
- ✅ Automatic HTTPS
- ✅ Easy setup

**Cons:**
- ❌ Limited to Streamlit apps
- ❌ Limited resources on free tier
- ❌ Public repos only (free tier)

---

### 🚂 Railway (Recommended for Docker)

Railway is excellent for Docker deployments and has a generous free tier.

#### Setup Steps:

1. **Create `railway.json`**:
   ```json
   {
     "$schema": "https://railway.app/railway.schema.json",
     "build": {
       "builder": "DOCKERFILE",
       "dockerfilePath": "trading_system/dashboard/Dockerfile"
     },
     "deploy": {
       "startCommand": "streamlit run trading_system/dashboard/app.py --server.port $PORT --server.address 0.0.0.0",
       "restartPolicyType": "ON_FAILURE",
       "restartPolicyMaxRetries": 10
     }
   }
   ```

2. **Update Dockerfile** (if needed):
   ```dockerfile
   # Use existing trading_system/dashboard/Dockerfile
   # Make sure it exposes port correctly
   ENV STREAMLIT_SERVER_PORT=$PORT
   ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
   ```

3. **Deploy to Railway**:
   - Go to https://railway.app
   - Sign in with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect and deploy

**Pros:**
- ✅ Free tier with $5 credit/month
- ✅ Great Docker support
- ✅ Automatic HTTPS
- ✅ Easy database integration
- ✅ Environment variables management

**Cons:**
- ⚠️ Free tier limited (sleeps after inactivity)
- ⚠️ Need to pay for always-on

---

### 🎨 Render (Good for Auto-Deploy)

Render offers automatic deployments from Git with a free tier.

#### Setup Steps:

1. **Create `render.yaml`**:
   ```yaml
   services:
     - type: web
       name: trading-dashboard
       env: docker
       dockerfilePath: ./trading_system/dashboard/Dockerfile
       dockerContext: .
       plan: free
       healthCheckPath: /_stcore/health
       envVars:
         - key: STREAMLIT_SERVER_PORT
           value: 8501
         - key: STREAMLIT_SERVER_ADDRESS
           value: 0.0.0.0
   ```

2. **Deploy to Render**:
   - Go to https://render.com
   - Sign up with GitHub
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Select "Docker" as environment
   - Render will use the dockerfile automatically

**Pros:**
- ✅ Free tier available
- ✅ Auto-deploy from Git
- ✅ Automatic HTTPS
- ✅ Simple setup

**Cons:**
- ❌ Free tier sleeps after 15 min inactivity
- ❌ Slower cold starts
- ❌ Limited resources

---

### ✈️ Fly.io (Best Performance)

Fly.io offers excellent global performance and a generous free tier.

#### Setup Steps:

1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Create `fly.toml`**:
   ```toml
   app = "trading-dashboard"
   primary_region = "iad"

   [build]
     dockerfile = "trading_system/dashboard/Dockerfile"

   [http_service]
     internal_port = 8501
     force_https = true
     auto_stop_machines = true
     auto_start_machines = true
     min_machines_running = 0
     processes = ["app"]

     [[http_service.checks]]
       grace_period = "10s"
       interval = "30s"
       method = "GET"
       timeout = "5s"
       path = "/_stcore/health"
   ```

3. **Deploy**:
   ```bash
   fly launch
   fly deploy
   ```

**Pros:**
- ✅ Generous free tier
- ✅ Global edge network
- ✅ Fast cold starts
- ✅ Great for production

**Cons:**
- ⚠️ More setup steps
- ⚠️ Need CLI knowledge

---

### 🟢 Replit (Quick Prototype Only)

Replit is good for quick testing but **not recommended for production**.

#### Setup Steps:

1. **Create `.replit` file**:
   ```toml
   language = "python3"
   run = "streamlit run trading_system/dashboard/app.py --server.port 8501 --server.address 0.0.0.0"
   ```

2. **Create `replit.nix`** (optional, for dependencies):
   ```nix
   { pkgs }: {
     deps = [
       pkgs.python38Full
     ];
   }
   ```

3. **Deploy**:
   - Import from GitHub in Replit
   - Click "Run"

**Pros:**
- ✅ Very easy to start
- ✅ Free tier available
- ✅ Good for testing

**Cons:**
- ❌ Not suitable for production
- ❌ Limited resources
- ❌ Can be slow
- ❌ No guaranteed uptime

---

## Quick Start: Local Development

Before deploying, test locally:

```bash
# 1. Install Streamlit
pip install -e ".[visualization]"

# 2. Run backtest results dashboard
python -m trading_system dashboard --run-id <your_run_id>

# 3. Or run trading assistant dashboard
python -m trading_system trading-dashboard --port 8501
```

## Data/Results Access

All platforms need access to your data and results. Options:

### Option 1: Include in Git (Small datasets only)
- Commit `results/` directory to Git
- ⚠️ Only for small test datasets
- ❌ Not recommended for production

### Option 2: External Storage (Recommended)
- Use S3, Google Cloud Storage, or similar
- Mount/download at runtime
- ✅ Scalable
- ✅ Production-ready

### Option 3: Database (Best for production)
- Store results in PostgreSQL/SQLite
- Dashboard reads from database
- ✅ Most scalable
- ✅ Best for production

## Environment Variables

Set these in your platform's dashboard:

```bash
# Data paths
TRACKING_DB_PATH=/app/data/tracking.db
FEATURE_DB_PATH=/app/data/features.db

# API Keys (if needed)
MASSIVE_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key

# Streamlit settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## Platform-Specific Considerations

### For Streamlit Cloud:
- Main file must be in root or subdirectory
- Public repos only (free tier)
- Limited to Streamlit apps

### For Railway/Render/Fly.io:
- Use Dockerfile deployment
- Better for full-stack apps
- Can run multiple services

### For Replit:
- Best for quick testing only
- Limited file system persistence
- Not for production workloads

## Recommended Setup

**For Quick Testing:**
1. Start with **Streamlit Cloud** (easiest)

**For Production:**
1. Use **Railway** or **Fly.io** (better performance)
2. Store data in cloud storage (S3, etc.)
3. Use environment variables for secrets
4. Set up monitoring and alerts

## Troubleshooting

### "Streamlit not found"
```bash
pip install -e ".[visualization]"
# Or
pip install streamlit plotly
```

### "Port already in use"
- Use `$PORT` environment variable (platforms set this automatically)
- Streamlit will use it if set

### "Cannot find results/data"
- Ensure data is mounted/copied to container
- Check file paths are correct
- Use absolute paths in Docker

### Dashboard won't start
- Check logs: `docker logs <container>`
- Verify Streamlit is installed: `pip list | grep streamlit`
- Check port configuration

## Next Steps

1. **Choose a platform** based on your needs
2. **Test locally** first
3. **Deploy to platform** following platform-specific steps
4. **Configure environment variables**
5. **Set up monitoring** (if needed)

For more details on each platform, see their official documentation:
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Railway Docs](https://docs.railway.app)
- [Render Docs](https://render.com/docs)
- [Fly.io Docs](https://fly.io/docs)
