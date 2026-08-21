FROM python:3.11-slim

# Install system deps needed by some transitive packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project definition first so pip layer is cached independently of source changes
COPY pyproject.toml .

# Copy source and data
COPY src/ src/
COPY data/ data/
COPY app.py .
COPY evals/ evals/
COPY .env.example .env.example

# Install the package and all default dependencies
RUN pip install --no-cache-dir -e .

# Streamlit port
EXPOSE 8501

# Healthcheck so docker-compose / orchestrators know when the app is up
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
