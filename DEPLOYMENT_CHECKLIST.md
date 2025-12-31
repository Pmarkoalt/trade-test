# 🚀 Deployment Checklist - Streamlit Cloud

## Pre-Deployment Checklist

- [x] ✅ Code is committed to Git
- [x] ✅ Dashboard files exist (`trading_system/reporting/dashboard.py`)
- [x] ✅ Requirements.txt includes Streamlit
- [x] ✅ Streamlit config exists (`.streamlit/config.toml`)
- [ ] ⏳ Push latest changes to GitHub
- [ ] ⏳ Deploy to Streamlit Cloud

## Step 1: Push to GitHub (If Needed)

Your git status shows everything is clean, but let's make sure the new files are committed:

```bash
# Check what files need to be committed
git status

# If there are new files (like .streamlit/config.toml, requirements.txt updates, etc.)
git add .
git commit -m "Add Streamlit Cloud deployment configuration"
git push origin main
```

## Step 2: Deploy to Streamlit Cloud

### Option A: Backtest Results Dashboard (Recommended to start)

1. **Go to**: https://share.streamlit.io
2. **Sign in** with your GitHub account (authorize if needed)
3. **Click**: "New app" button
4. **Fill in**:
   - **Repository**: Select your repo (e.g., `yourusername/trade-test`)
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `trading_system/reporting/dashboard.py`
   - **App URL** (optional): Choose a name like `trading-dashboard`
5. **Advanced Settings** (click to expand):
   - Keep defaults (Streamlit Cloud handles this automatically)
6. **Click**: "Deploy" button
7. **Wait**: First deployment takes 2-3 minutes
8. **Access**: Your dashboard at `https://trading-dashboard.streamlit.app`

### Option B: Trading Assistant Dashboard (Full featured)

Same steps, but use:
- **Main file path**: `trading_system/dashboard/app.py`

## Step 3: Test Your Deployment

1. Wait for deployment to complete (green checkmark)
2. Click on your app URL
3. Test the dashboard:
   - Verify it loads
   - Check if results are visible
   - Test navigation

## Step 4: Configure Secrets (Optional)

If you want to add API keys later:

1. Go to your app on Streamlit Cloud
2. Click "⚙️ Settings" (top right)
3. Click "Secrets" tab
4. Add secrets in TOML format:
   ```toml
   MASSIVE_API_KEY = "your_key_here"
   ALPHA_VANTAGE_API_KEY = "your_key_here"
   NEWSAPI_API_KEY = "your_key_here"
   SENDGRID_API_KEY = "your_key_here"
   ```
5. Save and redeploy

## Troubleshooting

### "Module not found" errors
- Check that `requirements.txt` includes all dependencies
- Streamlit Cloud automatically installs from `requirements.txt`

### "Results not found" errors
- The dashboard needs access to your `results/` directory
- Options:
  1. **Include in Git** (small datasets only): Commit results folder
  2. **Use database** (production): Store results in database
  3. **External storage** (scalable): Use S3/cloud storage

### Deployment fails
- Check build logs in Streamlit Cloud
- Verify `requirements.txt` syntax is correct
- Ensure Python version compatibility (3.9+)

### Dashboard loads but shows no data
- Verify results directory structure matches expected format
- Check file paths in dashboard code
- Review logs in Streamlit Cloud

## Next Steps After Deployment

1. ✅ Share your dashboard URL
2. ✅ Set up auto-deploy (default: redeploys on git push)
3. ✅ Configure secrets if needed
4. ✅ Monitor usage (Streamlit Cloud shows stats)

## Quick Reference

- **Dashboard URL**: `https://your-app-name.streamlit.app`
- **Settings**: Click ⚙️ in Streamlit Cloud dashboard
- **Logs**: View in Streamlit Cloud app page
- **Redeploy**: Automatic on git push, or manual "Reboot app"

---

**Ready?** Let's deploy! Follow Step 2 above. 🚀

