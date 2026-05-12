FROM python:3.12-alpine

# # Build dependencies for native packages (faiss-cpu, numpy, torch, etc.)
# RUN apk add --no-cache \
#     build-base \
#     gfortran \
#     openblas-dev \
#     libffi-dev \
#     cmake

WORKDIR /app

# Copy application code
COPY .env .env
COPY app/ ./app/
COPY data/ ./data/
COPY requirements.txt .

# Install dependencies first for better layer caching
RUN pip install --no-cache-dir -r requirements.txt


# # Required environment variables (pass via docker run --env-file or -e)
# ENV GROQ_API_KEY=""
# ENV HUGGINGFACEHUB_ACCESS_TOKEN=""

# Run as non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
