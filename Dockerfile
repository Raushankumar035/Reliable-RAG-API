FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    CHROMA_PERSIST_DIR=/app/storage/chroma

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the embedding model so the first request is fast and the container works offline afterwards.
# (Non-fatal: if the build has no internet, the model is downloaded on first use instead.)
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')" \
    || echo "WARNING: embedding model not pre-downloaded"

COPY . .

RUN useradd --create-home appuser \
    && mkdir -p /app/storage \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
