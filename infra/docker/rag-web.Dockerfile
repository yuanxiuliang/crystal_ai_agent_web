FROM node:22-bookworm-slim AS build

ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
    COREPACK_NPM_REGISTRY=https://registry.npmmirror.com \
    npm_config_registry=https://registry.npmmirror.com \
    PNPM_HOME=/pnpm \
    PATH=/pnpm:$PATH

WORKDIR /opt/agentweb

RUN corepack enable && corepack prepare pnpm@9.15.0 --activate

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/rag-platform/package.json apps/rag-platform/package.json
COPY apps/inference-platform/package.json apps/inference-platform/package.json
COPY apps/portal/package.json apps/portal/package.json
COPY apps/search-platform/package.json apps/search-platform/package.json
COPY packages/shared-types/package.json packages/shared-types/package.json
COPY packages/ui/package.json packages/ui/package.json

RUN pnpm --filter @agentweb/rag-platform install --frozen-lockfile

COPY apps/rag-platform apps/rag-platform
COPY packages packages

# The browser always uses same-origin /api routes. Next proxies those routes to rag-api on
# the Docker edge network, so neither the LAN nor the eventual public browser sees port 8003.
ENV NEXT_PUBLIC_RAG_API_BASE_URL="" \
    RAG_API_INTERNAL_URL=http://rag-api:8003 \
    NODE_ENV=production

RUN pnpm --filter @agentweb/rag-platform build

FROM node:22-bookworm-slim

ENV NODE_ENV=production \
    PORT=3003 \
    HOSTNAME=0.0.0.0

WORKDIR /opt/agentweb

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin agentweb

COPY --from=build --chown=agentweb:agentweb /opt/agentweb/apps/rag-platform/.next/standalone ./
COPY --from=build --chown=agentweb:agentweb /opt/agentweb/apps/rag-platform/.next/static ./apps/rag-platform/.next/static

USER agentweb

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=8 \
  CMD node -e "fetch('http://127.0.0.1:3003/login').then((response) => { if (!response.ok) process.exit(1); }).catch(() => process.exit(1));" || exit 1

CMD ["node", "apps/rag-platform/server.js"]
