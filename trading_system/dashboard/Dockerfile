# Dockerfile for Trading Assistant Dashboard
# Build: docker build -t trading-dashboard .
# Run: docker run -p 8501:8501 trading-dashboard

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trading system package
COPY . .

# Install the package
RUN pip install -e .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the dashboard
CMD ["streamlit", "run", "trading_system/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
