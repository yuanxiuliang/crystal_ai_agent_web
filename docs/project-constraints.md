# AgentWeb Project Constraints

本文档定义 AgentWeb 项目的开发边界、目录约束、平台隔离规则、RAG 平台职责、API 路由规范、数据存储规范和部署约束。

后续所有开发、代码生成、重构、评审和部署方案都应优先遵守本文档。若实现方案与本文档冲突，应先更新本文档并说明原因，再修改代码。

## 1. Project Goal

AgentWeb 是一个承载三个 AI 项目的 Web 平台：

1. 检索平台：`search-platform`
2. 模型推理平台：`inference-platform`
3. RAG 检索增强生成对话平台：`rag-platform`

第一阶段重点开发 `rag-platform`。`search-platform` 和 `inference-platform` 只做空占位，保留独立目录、独立入口和未来扩展边界。

当前开发阶段优先验证 `rag-platform` 的 RAG 主流程，不以 Web 前端为验收入口。`apps/rag-platform` 保留但暂停维护；项目 3 的默认开发入口为 `services/rag-api` 内的 CLI，直接调用智能体图验证路由、检索、证据融合和记忆流转。

## 2. Core Technical Stack

项目默认采用以下技术栈：

| Layer | Technology | Purpose |
| --- | --- | --- |
| Frontend | Next.js App Router + TypeScript + React | 三个平台的 Web 应用 |
| UI | Tailwind CSS + shadcn/ui style components | 统一界面风格和基础组件 |
| Backend | FastAPI + Pydantic | API 服务、流式输出、数据导入、RAG 编排入口 |
| Agent | LangGraph | 是否检索决策、工具调用、回答生成、长短期记忆 |
| Vector DB | Milvus | 单晶生长数据条 BM25 sparse + dense vector + RRF hybrid retrieval、metadata filter |
| Database | PostgreSQL | 用户、会话、消息、记忆、数据条元信息 |
| Cache/Queue | Redis optional | 缓存、后台任务、流式状态；MVP 可暂缓 |
| Deployment | Docker Compose first | 开发和早期部署环境 |

代码不得绑定单一模型厂商。大模型和 embedding 模型必须通过 OpenAI-compatible provider abstraction 接入。

## 3. Monorepo Structure

项目采用 monorepo 多应用结构：

```text
ai_agent_web/
  apps/
    portal/
    search-platform/
    inference-platform/
    rag-platform/

  services/
    search-api/
    inference-api/
    rag-api/

  packages/
    ui/
    config/
    shared-types/

  infra/
    docker/
    compose/
    nginx/
    postgres/
    milvus/

  data/
    raw/
    processed/
    samples/

  docs/
    project-constraints.md
    architecture.md
    api-contract.md

  scripts/
```

目录职责：

| Directory | Responsibility |
| --- | --- |
| `apps/` | 前端应用，一个平台一个 app |
| `services/` | 后端服务，一个平台一个 service |
| `packages/` | 共享前端包、共享类型、共享配置 |
| `infra/` | Docker、Nginx、数据库、向量库等基础设施配置 |
| `data/` | 原始数据、处理后数据、小样本数据 |
| `docs/` | 架构、接口、项目约束文档 |
| `scripts/` | 开发、构建、数据导入、运维脚本 |

## 4. Application Boundaries

三个平台必须隔离开发：

1. `search-platform` 不得直接 import `rag-platform` 代码。
2. `inference-platform` 不得直接 import `rag-platform` 代码。
3. `rag-platform` 不得直接 import `search-platform` 或 `inference-platform` 代码。
4. 平台之间共享代码只能通过 `packages/` 暴露。
5. 每个平台必须可以独立启动、独立构建、独立部署。
6. 不得为了未来平台需求提前污染当前 `rag-platform` 的业务边界。

后端服务也必须隔离：

1. `search-api` 不得直接 import `rag-api` 的内部模块。
2. `inference-api` 不得直接 import `rag-api` 的内部模块。
3. `rag-api` 不得直接 import `search-api` 或 `inference-api` 的内部模块。
4. 跨服务复用能力应通过 API、队列、共享 schema 或独立 package 实现，不得直接耦合内部实现。

## 5. Frontend Applications

前端应用目录固定为：

```text
apps/portal
apps/search-platform
apps/inference-platform
apps/rag-platform
```

### 5.1 portal

`portal` 是 Web 总入口，只负责：

1. 平台入口展示。
2. 统一导航。
3. 用户登录态衔接。
4. 跳转到三个独立平台。

`portal` 不得承载复杂业务逻辑，不得实现 RAG 检索、模型推理或数据管理逻辑。

### 5.2 search-platform

`search-platform` 是检索平台前端。

第一阶段只做占位页面，保留：

1. 独立目录。
2. 独立 package。
3. 独立路由入口。
4. 独立构建能力。

### 5.3 inference-platform

`inference-platform` 是模型推理平台前端。

第一阶段只做占位页面，保留：

1. 独立目录。
2. 独立 package。
3. 独立路由入口。
4. 独立构建能力。

### 5.4 rag-platform

`rag-platform` 是第一阶段核心前端应用。

当前阶段 `rag-platform` 前端暂停维护，不作为主流程验证入口。RAG 流程先通过 `services/rag-api` 的 CLI 验证；待智能体流程、检索质量和回答质量稳定后，再恢复 Web UI 接入。

`rag-platform` 负责：

1. 对话 UI。
2. 会话列表。
3. 消息流式展示。
4. 检索证据展示。
5. 引用数据条展示。
6. 模型和检索参数控制。
7. 用户对回答和检索结果的反馈入口。

`rag-platform` 不负责：

1. LangGraph 编排。
2. Milvus 查询。
3. embedding 生成。
4. 记忆写入策略。
5. 数据导入和清洗。
6. 直接连接 PostgreSQL、Milvus 或模型服务。

上述逻辑必须放在 `services/rag-api`。

## 6. Backend Services

后端服务目录固定为：

```text
services/search-api
services/inference-api
services/rag-api
```

第一阶段只实现 `services/rag-api`。`search-api` 和 `inference-api` 仅保留占位目录和 README，避免后续扩展时重构根目录。

## 7. RAG API Scope

`services/rag-api` 是 RAG 平台唯一后端入口。

当前阶段 `services/rag-api` 的主验证入口为命令行：

```bash
cd services/rag-api
source ~/.zshrc
.venv/bin/python -m src.cli.rag_chat --trace
```

CLI 必须直接复用 RAG 图编排，不得复制一套独立 RAG 逻辑。FastAPI 路由保留，用于后续 Web 接入和服务化部署。

`rag-api` 负责：

1. 对话 API。
2. SSE 或等价机制的流式响应。
3. LangGraph 智能体编排。
4. 判断是否需要检索。
5. 生成检索 query 和 metadata filters。
6. Milvus BM25 + dense vector + RRF hybrid retrieval。
7. 可选 rerank。
8. 检索结果融合生成。
9. 短期记忆读取和写入。
10. 长期记忆读取和写入。
11. 单晶生长数据条导入。
12. 数据规范化。
13. embedding 构建和重建。
14. PostgreSQL 和 Milvus 的服务端访问。

建议内部结构：

```text
services/rag-api/src/
  main.py
  config.py

  api/
    chat.py
    sessions.py
    retrieval.py
    memory.py
    admin.py

  agent/
    graph.py
    state.py
    prompts.py
    nodes/
      route_intent.py
      retrieve.py
      answer.py
      write_memory.py

  retrieval/
    embeddings.py
    milvus_store.py
    hybrid_search.py
    rerank.py

  ingestion/
    import_records.py
    normalize_records.py
    build_embeddings.py

  memory/
    short_term.py
    long_term.py

  db/
    session.py
    models.py
    migrations/

  schemas/
    chat.py
    records.py
    memory.py
```

## 8. Shared Packages

共享代码只能放在：

```text
packages/ui
packages/config
packages/shared-types
```

### 8.1 packages/ui

`packages/ui` 只能放通用 UI 组件，例如：

1. Button。
2. Dialog。
3. Sidebar。
4. Tabs。
5. Data table。
6. Toast。
7. Form controls。

不得放入任何平台业务逻辑。

不得放入 RAG 专属逻辑，例如：

1. Chat message domain state。
2. Retrieval evidence model。
3. LangGraph state。
4. Milvus result adapter。

### 8.2 packages/config

`packages/config` 放共享前端配置，例如：

1. TypeScript config。
2. ESLint config。
3. Tailwind config。
4. Prettier config。

### 8.3 packages/shared-types

`packages/shared-types` 放稳定共享类型。

仅允许放跨平台稳定类型，不允许放快速变化的平台内部类型。

如果某个类型只被 `rag-platform` 使用，应放在 `apps/rag-platform` 内部。

如果某个 schema 只被 `rag-api` 使用，应放在 `services/rag-api` 内部。

## 9. Data Storage

### 9.1 PostgreSQL

PostgreSQL 用于：

1. 用户。
2. 会话。
3. 消息。
4. 短期记忆 checkpoint。
5. 长期记忆。
6. 单晶生长数据条 metadata。
7. 数据导入任务状态。
8. 用户反馈。

### 9.2 Milvus

Milvus 用于：

1. 单晶生长数据条向量检索。
2. dense vector search。
3. BM25 sparse search。
4. RRF hybrid retrieval。
5. metadata filtering。

Milvus collection 必须至少支持：

1. `record_id`。
2. `material_formula`。
3. `material_name`。
4. `growth_method`。
5. `temperature_program`。
6. `atmosphere`。
7. `doi`。
8. `source_text`。
9. `confidence`。
10. `normalized_text`。
11. `text_sparse`。
12. `text_dense`。

### 9.3 data directory

`data/raw` 只保存原始数据，应用代码不得直接修改。

`data/processed` 保存处理后的中间数据，可由导入脚本生成。

`data/samples` 保存小样本测试数据，用于开发和自动化测试。

## 10. Growth Record Schema

单晶生长数据条应保留原始文本和规范化字段。

建议字段：

```text
id
title
doi
material_formula
material_name
target_phase
growth_method
precursor_materials
flux_or_transport_agent
crucible
atmosphere
pressure
temperature_program
cooling_rate
annealing
crystal_size
characterization
source_text
source_file
confidence
created_at
updated_at
```

用于 embedding 的文本不得直接等同于原始全文，应生成规范化表示，例如：

```text
Material: Mn3GaN.
Growth method: flux growth.
Precursors: ...
Temperature program: ...
Atmosphere: ...
Crystal result: ...
Source evidence: ...
```

## 11. API Routing

开发环境端口：

| Service | URL |
| --- | --- |
| portal | `http://localhost:3000` |
| search-platform | `http://localhost:3001` |
| inference-platform | `http://localhost:3002` |
| rag-platform | `http://localhost:3003` |
| search-api | `http://localhost:8001` |
| inference-api | `http://localhost:8002` |
| rag-api | `http://localhost:8003` |
| PostgreSQL | `localhost:5432` |
| Milvus | `localhost:19530` |
| Redis | `localhost:6379` |

生产环境统一路由：

```text
/                     -> portal
/search/              -> search-platform
/inference/           -> inference-platform
/rag/                 -> rag-platform

/api/search/          -> search-api
/api/inference/       -> inference-api
/api/rag/             -> rag-api
```

RAG API 路由应统一挂在 `/api/rag/` 下。

建议第一阶段 API：

```text
POST /api/rag/chat/stream
POST /api/rag/retrieve
GET  /api/rag/sessions
POST /api/rag/sessions
GET  /api/rag/sessions/{session_id}/messages
POST /api/rag/admin/growth-records/import
GET  /api/rag/admin/growth-records
POST /api/rag/admin/embeddings/rebuild
```

新增 API 必须在 `docs/api-contract.md` 中记录：

1. path。
2. method。
3. request schema。
4. response schema。
5. error shape。
6. authentication requirement。

## 12. LLM Provider Constraint

代码不得绑定单一模型厂商。

所有模型调用必须通过统一 provider abstraction。

必须通过环境变量配置：

```text
LLM_BASE_URL
LLM_API_KEY
LLM_MODEL
EMBEDDING_MODEL
```

可选配置：

```text
RERANK_MODEL
LLM_TIMEOUT_SECONDS
LLM_MAX_RETRIES
LLM_TEMPERATURE
LLM_MAX_TOKENS
```

前端不得直接调用模型服务。

所有模型调用必须经过 `services/rag-api` 或未来对应后端服务。

## 13. Agent Workflow Constraint

RAG 对话智能体必须显式区分以下节点：

1. 读取短期记忆。
2. 读取长期记忆。
3. 判断是否需要检索。
4. 生成检索 query 和 filters。
5. 执行检索。
6. 可选 rerank。
7. 融合检索结果生成回答。
8. 写入短期记忆。
9. 按规则写入长期记忆。

不得将完整 RAG 流程写成一个不可拆分的单函数。

是否检索的决策必须可观测，至少应记录：

1. `should_retrieve`。
2. `reason`。
3. `query`。
4. `filters`。
5. `top_k`。

## 14. Memory Constraint

### 14.1 Short-term memory

短期记忆用于当前会话上下文。

短期记忆可以包含：

1. 最近消息。
2. 当前研究材料。
3. 当前约束条件。
4. 最近引用过的数据条。
5. 当前任务阶段。

短期记忆应与 session 或 thread 绑定。

### 14.2 Long-term memory

长期记忆用于跨会话的稳定信息。

长期记忆只能保存稳定、可复用的信息，例如：

1. 用户长期关注的材料体系。
2. 用户偏好的回答格式。
3. 用户常用实验约束。
4. 用户明确要求记住的信息。

不得把每轮对话无条件写入长期记忆。

长期记忆写入必须满足至少一个条件：

1. 用户明确要求记住。
2. 用户明确确认某个偏好。
3. 多轮对话中重复出现的稳定研究偏好。
4. 系统判断为长期有用，且经过规则过滤。

长期记忆应支持查看、修正和删除。

## 15. Retrieval Constraint

RAG 检索必须支持：

1. 语义检索。
2. 关键词检索。
3. metadata filter。
4. BM25 + dense vector + RRF hybrid retrieval。
5. 引用来源返回。

检索结果必须携带：

1. `record_id`。
2. `score`。
3. `source_text`。
4. `metadata`。
5. `doi` or source identifier when available。

模型回答涉及数据条时，必须返回引用信息。

回答中必须区分：

1. 数据中明确存在的事实。
2. 模型根据数据做出的推断。
3. 需要用户确认的不确定信息。

不得让模型伪造不存在的文献、DOI、实验参数或数据条。

## 16. Frontend RAG UX Constraint

`rag-platform` 第一阶段至少应支持：

1. 新建会话。
2. 历史会话列表。
3. 流式回答。
4. Markdown 渲染。
5. 检索证据面板。
6. 引用数据条查看。
7. 强制检索开关。
8. `top_k` 控制。
9. 模型名称展示。
10. 错误状态展示。

前端应避免把检索证据隐藏在纯文本回答中。证据应在结构化区域展示，便于用户核查。

## 17. Development Constraint

开发时优先保证 `rag-platform` 和 `rag-api` 可独立运行。

禁止事项：

1. 禁止跨 `apps/` 直接 import。
2. 禁止跨 `services/` 直接 import 内部模块。
3. 禁止前端直接连接 PostgreSQL。
4. 禁止前端直接连接 Milvus。
5. 禁止前端直接调用模型服务。
6. 禁止把平台业务逻辑放入 `packages/ui`。
7. 禁止把 RAG agent 节点写入前端。
8. 禁止把原始数据修改逻辑写入页面组件。
9. 禁止把密钥写入代码仓库。
10. 禁止把临时实验脚本混入核心服务目录。

推荐事项：

1. 平台内部逻辑优先放在平台自己的目录。
2. 只有稳定复用的代码才进入 `packages/`。
3. RAG 智能体逻辑放在 `services/rag-api/src/agent`。
4. 检索逻辑放在 `services/rag-api/src/retrieval`。
5. 数据导入逻辑放在 `services/rag-api/src/ingestion`。
6. 数据库模型放在 `services/rag-api/src/db`。
7. API schema 放在 `services/rag-api/src/schemas`。

## 18. Deployment Constraint

第一阶段使用 Docker Compose。

开发和早期部署至少包含：

1. `rag-platform`。
2. `rag-api`。
3. `postgres`。
4. `milvus-standalone`、`milvus-etcd`、`milvus-minio`。

可选包含：

1. `portal`。
2. `search-platform`。
3. `inference-platform`。
4. `redis`。
5. `nginx`。

`search-platform` 和 `inference-platform` 可以只作为静态占位服务存在。

后续若迁移到 Kubernetes，不应改变应用内部目录边界和 API 边界。

## 19. Environment Constraint

每个应用或服务应有自己的环境变量边界。

根目录提供 `.env.example`，不得提交真实 `.env`。

建议环境变量前缀：

```text
NEXT_PUBLIC_        # 前端可公开变量
RAG_API_            # RAG API 服务变量
POSTGRES_           # PostgreSQL 配置
MILVUS_             # Milvus 配置
REDIS_              # Redis 配置
LLM_                # 大模型配置
EMBEDDING_          # embedding 配置
```

密钥只能存在于本地 `.env`、部署平台 secret 或安全配置系统中。

## 20. Documentation Constraint

架构变化必须同步更新文档。

核心文档：

```text
docs/project-constraints.md
docs/architecture.md
docs/api-contract.md
docs/rag-design.md
docs/rag-langgraph-design.md
docs/deployment.md
```

文档职责：

| Document | Responsibility |
| --- | --- |
| `project-constraints.md` | 项目硬约束，开发必须遵守 |
| `architecture.md` | 总体架构说明，解释为什么这样设计 |
| `api-contract.md` | 前后端 API 合同 |
| `rag-design.md` | RAG 智能体、检索、记忆设计 |
| `rag-langgraph-design.md` | LangGraph 图结构、节点、状态和流式事件设计 |
| `deployment.md` | 本地、测试、生产部署说明 |

## 21. Review Checklist

提交代码或生成新模块前，应检查：

1. 是否违反跨平台 import 规则。
2. 是否把平台业务逻辑放进 `packages/ui`。
3. 是否让前端直接访问数据库、向量库或模型服务。
4. 是否破坏 `rag-platform` 和 `rag-api` 的独立运行能力。
5. 是否新增 API 但没有更新 API 文档。
6. 是否新增架构边界但没有更新本文档。
7. 是否让长期记忆无条件写入。
8. 是否让模型回答缺少引用证据。
9. 是否泄露密钥或本地私有配置。
10. 是否把临时实验文件混入正式目录。

## 22. Current Phase

当前阶段：

```text
Phase 1: RAG platform MVP
```

本阶段目标：

1. 建立 monorepo 骨架。
2. 建立三个前端平台目录。
3. 建立三个后端服务目录。
4. 让 `search-platform` 和 `inference-platform` 保持占位。
5. 重点实现 `rag-platform`。
6. 重点实现 `rag-api`。
7. 支持单晶生长方法数据导入。
8. 支持基于 Milvus 的 BM25 + dense vector + RRF 混合检索。
9. 支持 LangGraph 决策是否检索。
10. 支持流式对话。
11. 支持短期记忆。
12. 支持受控长期记忆。

超出本阶段的内容应谨慎引入，避免影响 RAG MVP 的交付速度和边界清晰度。
