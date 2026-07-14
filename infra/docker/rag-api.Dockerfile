FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    TORCH_CPU_WHEEL=https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.6.0%2Bcpu-cp312-cp312-linux_x86_64.whl \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /opt/agentweb

COPY infra/docker/requirements.cpu.txt /tmp/requirements.cpu.txt

# Pin the verified CPU wheel before installing the project extras. The constraints file
# prevents sentence-transformers or the prediction extra from replacing it with CUDA torch.
RUN python -m pip install --no-cache-dir --prefer-binary \
      --index-url "${PIP_INDEX_URL}" \
      "${TORCH_CPU_WHEEL}"

COPY services/rag-api /opt/agentweb/services/rag-api
COPY models /opt/agentweb/models
COPY data/processed /opt/agentweb/data/processed
COPY rawData.jsonl /opt/agentweb/rawData.jsonl
COPY infra/docker/entrypoints /opt/agentweb/infra/docker/entrypoints

WORKDIR /opt/agentweb/services/rag-api
RUN python -m pip install --no-cache-dir --prefer-binary \
      --index-url "${PIP_INDEX_URL}" \
      -c /tmp/requirements.cpu.txt \
      ".[postgres,prediction]" \
    && python -c "import torch; assert torch.__version__.startswith('2.6.0+cpu'), torch.__version__"

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin agentweb \
    && chown -R agentweb:agentweb /opt/agentweb

USER agentweb
WORKDIR /opt/agentweb/services/rag-api

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=8 \
  CMD python -c "from urllib.request import urlopen; response = urlopen('http://127.0.0.1:8003/api/rag/health', timeout=3); assert response.status == 200" || exit 1

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
