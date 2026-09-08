# Attribution Studio — Docker image for the Flask + gunicorn dashboard.
#
# Works as-is on Hugging Face Spaces (Docker SDK, expects port 7860) and on
# any other Docker-capable host (Render's Docker runtime, Koyeb, Back4app,
# a bare VM) via the $PORT env var, which takes priority when set.
#
# Build:  docker build -t attribution-studio .
# Run:    docker run -p 7860:7860 attribution-studio
FROM python:3.11-slim

# All the heavy dependencies (pandas, numpy, scipy, xgboost, shap, duckdb)
# ship prebuilt manylinux wheels for this base image, so no compiler
# toolchain is installed here — keeps the image smaller and the build faster.
WORKDIR /app

# Copy requirements first so `pip install` is cached across rebuilds that
# only change application code, not dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now the rest of the app, including the bundled real dataset
# (data/raw/real_channel_journeys.csv) that the dashboard reads at runtime.
COPY . .

# Hugging Face Spaces' Docker runtime expects the container to run as a
# non-root user with UID 1000 and to own everything under /app — without
# this, writes to the filesystem (even incidental ones, e.g. .pyc caching)
# can fail with permission errors on their infra.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# HF Spaces (Docker SDK) routes traffic to port 7860 by default; most other
# Docker hosts inject their own $PORT. ${PORT:-7860} picks up either.
EXPOSE 7860
ENV PORT=7860

CMD gunicorn -w 2 -b 0.0.0.0:${PORT:-7860} --timeout 120 dashboards.server:app
