# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps
COPY services/interceptor/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (no external deps required for stub)
COPY services/interceptor/src ./src
ENV PYTHONPATH=/app/src

# Run the interceptor capture stub
# Entry: [capture.py](services/interceptor/src/capture.py:1)
CMD ["python", "-u", "src/capture.py"]