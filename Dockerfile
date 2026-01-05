# Trading System Dockerfile
# Multi-stage build for optimized image size

# Build stage (optional - for building extensions if needed in future)
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
COPY pyproject.toml .
# Install package in editable mode with dev dependencies for testing
RUN pip install --no-cache-dir --user -e ".[dev]"

# Runtime stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY trading_system/ ./trading_system/
COPY pyproject.toml .
COPY pytest.ini .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Create directories for data, configs, and results
RUN mkdir -p /app/data /app/configs /app/results /app/tests/fixtures

# Copy test fixtures (optional - for running tests)
COPY tests/ ./tests/

# Copy example configs
COPY EXAMPLE_CONFIGS/ ./EXAMPLE_CONFIGS/

# Copy configs directory
COPY configs/ ./configs/

# Set default command
ENTRYPOINT ["python", "-m", "trading_system"]

# Default to showing help if no command is provided
CMD ["--help"]
