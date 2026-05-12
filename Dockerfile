FROM python:3.12-slim

WORKDIR /app

# Install CPU-only PyTorch first (much smaller download), then remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir "torch<2.3" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY .env ./.env

# Run as non-root user (with home dir for HuggingFace model cache)
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
