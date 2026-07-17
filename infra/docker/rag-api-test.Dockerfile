ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# The production image intentionally excludes test tooling. This layer is used only by the
# isolated X1C candidate test container and is never started as an application service.
USER root
RUN python -m pip install --no-cache-dir --prefer-binary \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      "pytest>=8.2.0" "pytest-asyncio>=0.23.0" "ruff>=0.6.0" "httpx>=0.27.0,<1"

# Aggregate-retrieval unit tests load this compact fixture from the repository root.
# Keep the candidate image aligned with the full-workspace CI test layout.
COPY e2e/fixtures /opt/agentweb/e2e/fixtures

USER agentweb
WORKDIR /opt/agentweb/services/rag-api
