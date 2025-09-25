FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/app/hf_cache

RUN pip install --no-cache-dir transformers torch

# Pre-download specific models
# @TODO configuration
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('unitary/toxic-bert')"
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('ibm-granite/granite-3.3-8b-instruct')"

FROM python:3.11-slim

ENV HF_HOME=/app/hf_cache

COPY --from=builder /app/hf_cache /app/hf_cache
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["ragent"]
