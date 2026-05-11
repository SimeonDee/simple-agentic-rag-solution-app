FROM python:3.12-alpine

# Build dependencies for native packages (faiss-cpu, numpy, torch, etc.)
RUN apk add --no-cache \
    build-base \
    gfortran \
    openblas-dev \
    libffi-dev \
    cmake

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY main.py .
COPY data/ ./data/

# Required environment variables (pass via docker run --env-file or -e)
ENV GROQ_API_KEY=""
ENV HUGGINGFACEHUB_ACCESS_TOKEN=""

# Run as non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
