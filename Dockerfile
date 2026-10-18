FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 make && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY odyssey_anchor ./odyssey_anchor
RUN pip install --no-cache-dir .

COPY Makefile ./
COPY web ./web
COPY data ./data
COPY reports ./reports

# Builds anything missing from a fresh clone (downloads the public data on first run).
RUN test -f data/processed/diseases.parquet || make data index

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "odyssey_anchor", "serve", "--host", "0.0.0.0", "--port", "8000"]
