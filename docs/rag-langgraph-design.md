# RAG LangGraph Design

本文档定义项目 3：RAG 检索增强生成对话平台的 LangGraph 设计版本。

当前版本：

```text
GrowthRAG Graph v0.1
```

该版本服务于第一阶段 MVP，重点解决：

1. 用户对话输入。
2. 判断是否需要检索单晶生长数据条。
3. 生成检索计划。
4. 执行 Milvus BM25 + dense vector + RRF hybrid retrieval。
5. 判断检索结果是否足够。
6. 融合检索证据生成回答。
7. 维护短期会话记忆。
8. 按规则写入长期用户记忆。
9. 向前端输出可观察的流式事件。

当前开发阶段不以 Web 前端作为默认入口。主验证入口为 `services/rag-api` 的 CLI，用于在终端直接观察节点流转、路由决策、检索结果、证据评分和最终回答。FastAPI/SSE 与 Web UI 保留为后续接入路径。

## 1. Design Principles

本设计遵守 `docs/project-constraints.md`。

核心原则：

1. LangGraph 只放在 `services/rag-api`。
2. 前端 `rag-platform` 不直接接触 LangGraph、Milvus、PostgreSQL 或模型服务。
3. 所有模型调用通过 OpenAI-compatible provider abstraction。
4. 短期记忆使用有界的会话状态持久化；资源允许时可接入 LangGraph checkpointer。
5. 长期记忆使用有配额的结构化 store；生产环境使用外置 PostgreSQL。
6. 检索结果必须可追溯到 `record_id` 和 `source_text`。
7. 回答必须区分数据事实、模型推断和不确定信息。
8. 长期记忆不得无条件写入。

LangGraph 官方定位是面向 long-running、stateful agent 的低层编排框架，核心能力包括 durable execution、streaming、human-in-the-loop 和 persistence。该能力组合与本项目的“检索决策 + 长短期记忆 + 流式对话”需求匹配。

## 2. Graph Version

```text
Name: GrowthRAG Graph
Version: v0.1
Primary service: services/rag-api
Primary development entrypoint: services/rag-api/src/cli/rag_chat.py
Service entrypoint: POST /api/rag/chat/stream
Graph thread key: session_id
User memory namespace: (user_id, "growth_rag", "memories")
```

CLI 运行方式：

```bash
cd services/rag-api
source ~/.zshrc
.venv/bin/python -m src.cli.rag_chat --trace
```

单次问题模式：

```bash
.venv/bin/python -m src.cli.rag_chat "ZnIn2S4 的 CVT 生长温度是多少？" --trace
```

MVP 阶段只做单图，不引入多智能体和子图。

后续版本可以演进为：

| Version | Change |
| --- | --- |
| v0.1 | 单图 RAG 对话，支持检索决策、证据回答、短期记忆、受控长期记忆 |
| v0.2 | 增加 rerank、检索评测、失败恢复 |
| v0.3 | 增加 human approval，用于写入长期记忆或生成实验建议前确认 |
| v0.4 | 拆分 planner/retriever/critic 子图 |
| v1.0 | 生产版，完整观测、评测、权限、审计 |

## 3. High-level Workflow

```text
User message
  |
  v
ingest_input
  |
  v
load_short_term_memory
  |
  v
load_long_term_memory
  |
  v
normalize_question
  |
  v
route_intent
  |
  +-- direct_answer ----------+
  |                           |
  +-- clarify ----------------+
  |                           |
  +-- retrieve ---------------+
                              |
                              v
                      build_retrieval_plan
                              |
                              v
                      retrieve_growth_records
                              |
                              v
                      judge_retrieval_sufficiency
                              |
               +--------------+--------------+
               |                             |
               v                             v
        answer_with_evidence          answer_with_limits
               |                             |
               +-------------+---------------+
                             |
                             v
                    propose_memory_update
                             |
                             v
                    write_short_term_memory
                             |
                             v
                    write_long_term_memory
                             |
                             v
                         finalize
```

## 4. Graph Nodes

### 4.1 ingest_input

职责：

1. 接收用户消息。
2. 校验 `user_id`、`session_id`、`message_id`。
3. 读取前端传入的控制参数。
4. 写入本轮输入状态。

输入：

```text
user_id
session_id
message_id
user_message
force_retrieve
top_k
model
retrieval_mode
```

输出：

```text
current_user_message
runtime_options
```

约束：

1. `session_id` 用作 LangGraph `thread_id`。
2. `thread_id` 必须保持短且稳定，建议使用 UUID。
3. 不在此节点调用模型。

### 4.2 load_short_term_memory

职责：

1. 从有界会话状态存储恢复当前 thread state。
2. 读取当前会话内最近消息。
3. 读取当前会话摘要。
4. 读取最近引用过的数据条。

短期记忆内容：

```text
recent_messages
conversation_summary
active_materials
active_constraints
last_retrieval_refs
```

说明：

当前实现为 1 vCPU / 1 GiB 主机使用有界 SQLite 会话存储：只保存最近消息窗口、固定长度摘要和当前任务槽位，不保存每个节点的无限 checkpoint 历史。多实例生产环境可将同一 schema 指向外置 PostgreSQL，或在确有 time-travel / interrupt 需求时接入 PostgreSQL-backed LangGraph checkpointer。

### 4.3 load_long_term_memory

职责：

1. 读取用户长期偏好。
2. 读取用户长期研究上下文。
3. 读取用户实验室约束。
4. 读取用户历史确认的输出偏好。

命名空间：

```text
(user_id, "growth_rag", "memories")
(user_id, "growth_rag", "preferences")
(user_id, "growth_rag", "lab_constraints")
```

长期记忆示例：

```json
{
  "type": "lab_constraint",
  "content": "用户实验室最高炉温为 1200 C",
  "confidence": 0.95,
  "source": "user_confirmed",
  "created_at": "2026-07-08T00:00:00Z"
}
```

说明：

长期记忆通过独立的有界 store 保存，按 `(user_id, memory_type, memory_key)` Upsert，避免重复追加。单机使用 SQLite；多实例生产环境使用外置 PostgreSQL，不新增记忆向量库。

### 4.4 normalize_question

职责：

1. 轻量标准化用户问题。
2. 提取显式材料名、化学式、方法名、温度、气氛等。
3. 合并短期上下文中的省略信息。

例子：

```text
上一轮用户问：Mn3GaN 的生长方法
本轮用户问：那温度怎么设置？

normalize_question 输出：
用户想询问 Mn3GaN 单晶生长方法中的温度程序和温度设置。
```

输出：

```text
normalized_question
detected_entities
```

### 4.5 route_intent

职责：

判断本轮是否需要检索。

输出必须是结构化 JSON：

```json
{
  "intent": "retrieve",
  "should_retrieve": true,
  "reason": "用户询问具体材料 Mn3GaN 的单晶生长条件，需要查询数据条",
  "answer_mode": "evidence_grounded",
  "missing_slots": [],
  "confidence": 0.91
}
```

允许的 `intent`：

| Intent | Meaning |
| --- | --- |
| `direct_answer` | 不需要检索，可直接回答 |
| `retrieve` | 需要检索数据条 |
| `clarify` | 信息不足，先追问 |
| `smalltalk` | 闲聊或非任务性对话 |
| `unsupported` | 超出系统能力或不应回答 |

强制规则：

1. 如果 `force_retrieve=true`，必须进入检索路径。
2. 用户询问具体材料、具体生长方法、实验条件、温度程序、文献记录、数据条对比时，默认进入检索路径。
3. 用户只问通用概念时，可以直接回答。
4. 用户问题缺少关键槽位时，进入澄清路径。

### 4.6 clarify_question

职责：

在信息不足时追问用户。

触发条件：

1. 用户问题过宽。
2. 缺少目标材料。
3. 缺少任务目标。
4. 检索条件无法构造。

例子：

```text
用户：帮我找一个好的单晶生长方法。

系统：请先告诉我目标材料或材料体系，例如 Mn3GaN、Bi2Se3 或氧化物体系。你也可以补充可用炉温范围、气氛和是否接受助熔剂法。
```

输出：

```text
assistant_message
next_action = "await_user"
```

### 4.7 direct_answer

职责：

回答不需要检索的问题。

适用场景：

1. 解释通用概念。
2. 总结当前对话。
3. 说明系统能力。
4. 回复简单交互。

约束：

1. 不得伪造数据条。
2. 如果回答中涉及具体单晶生长记录，应改走检索路径。

### 4.8 build_retrieval_plan

职责：

把用户问题转为检索计划。

输出：

```json
{
  "query_text": "Mn3GaN single crystal growth temperature program flux method",
  "dense_query": "Mn3GaN single crystal growth temperature profile",
  "sparse_query": "Mn3GaN flux growth temperature cooling rate atmosphere",
  "filters": {
    "material_formula": "Mn3GaN",
    "growth_method": null,
    "temperature_min": null,
    "temperature_max": null,
    "atmosphere": null
  },
  "top_k": 12,
  "retrieval_mode": "hybrid",
  "must_have": ["material_formula"],
  "nice_to_have": ["temperature_program", "growth_method", "atmosphere"]
}
```

规则：

1. 化学式、材料名、方法名优先进入 metadata filter。
2. 温度、气氛、助熔剂、运输剂等同时进入 query 和 filter。
3. 如果材料名不确定，不强行加 filter，避免漏召回。
4. `top_k` 默认 12，可由前端传入，但应设置上限。

### 4.9 retrieve_growth_records

职责：

调用 Milvus BM25 + dense vector + RRF hybrid retrieval。

检索层输入：

```text
dense_query
sparse_query
filters
top_k
retrieval_mode
```

检索层输出：

```json
[
  {
    "record_id": "uuid",
    "score": 0.87,
    "material_formula": "Mn3GaN",
    "growth_method": "flux growth",
    "temperature_program": "...",
    "atmosphere": "argon",
    "doi": "10.xxxx/xxxx",
    "source_text": "...",
    "matched_fields": ["material_formula", "temperature_program"]
  }
]
```

约束：

1. 检索结果必须携带 `record_id`。
2. 检索结果必须携带可展示的证据文本。
3. 检索结果不得只返回向量分数。
4. Milvus 查询只允许在 `services/rag-api/src/retrieval` 中实现。

### 4.10 judge_retrieval_sufficiency

职责：

判断检索结果是否足够回答。

输入：

```text
normalized_question
retrieval_results
retrieval_plan
```

输出：

```json
{
  "is_sufficient": true,
  "reason": "找到 4 条 Mn3GaN 相关记录，其中 3 条包含温度程序",
  "usable_record_ids": ["record-1", "record-2", "record-3"],
  "missing_evidence": [],
  "answer_strategy": "compare_and_summarize"
}
```

不足场景：

1. 没有相关记录。
2. 记录相关但缺少用户询问的字段。
3. 检索结果材料不一致。
4. 用户要求推荐实验方案，但证据不足以支持确定性推荐。

不足时不应硬答，应进入 `answer_with_limits`。

### 4.11 answer_with_evidence

职责：

基于检索结果生成回答。

回答结构建议：

```text
结论
证据摘要
关键实验条件
可操作建议
不确定性
引用数据条
```

约束：

1. 必须引用 `record_id` 或 DOI。
2. 不得编造未检索到的温度、气氛、助熔剂、晶体尺寸。
3. 必须区分“数据中明确记载”和“根据数据推断”。
4. 如果多个数据条冲突，应说明冲突。
5. 如果用户要求实验建议，应给出条件边界和风险提示。

### 4.12 answer_with_limits

职责：

当检索不足时给出受限回答。

回答原则：

1. 明确说明检索不足。
2. 展示已找到的部分证据。
3. 说明缺失信息。
4. 给出下一步建议或追问。

例子：

```text
我没有在当前数据条中找到直接针对 Mn3GaN 的完整温度程序。已有记录只覆盖了相近氮化物体系的助熔剂法，因此不能直接给出确定实验方案。你可以补充目标相、可接受助熔剂和最高炉温，我再扩大到相近体系检索。
```

### 4.13 propose_memory_update

职责：

从本轮对话中提出候选长期记忆。

输出：

```json
{
  "short_term_updates": {
    "active_materials": ["Mn3GaN"],
    "active_constraints": ["用户关注温度程序"]
  },
  "long_term_candidates": [
    {
      "type": "research_interest",
      "content": "用户关注 Mn3GaN 单晶生长",
      "source": "inferred",
      "confidence": 0.72,
      "write_policy": "defer_until_repeated_or_confirmed"
    }
  ]
}
```

### 4.14 write_short_term_memory

职责：

更新当前 thread state。

写入内容：

1. 本轮用户消息。
2. 本轮助手消息。
3. 本轮检索计划。
4. 本轮检索引用。
5. 当前材料和约束。
6. 必要时更新会话摘要。

说明：

短期记忆由 checkpointer 持久化。长对话中应定期摘要和裁剪旧消息，避免上下文无限增长。

### 4.15 write_long_term_memory

职责：

按规则写入长期记忆。

允许写入条件：

1. 用户明确要求记住。
2. 用户明确确认某个偏好。
3. 多轮对话中重复出现的稳定偏好。
4. 系统判断为长期有用，且通过规则过滤。

默认策略：

| Candidate Source | Action |
| --- | --- |
| `user_confirmed` | 允许写入 |
| `explicit_remember_request` | 允许写入 |
| `inferred_high_confidence` | 可写入，但标记为 inferred |
| `inferred_low_confidence` | 不写入，保留在短期记忆 |
| `sensitive_or_private` | 不写入 |

MVP 阶段建议：

1. 先自动写入用户明确要求记住的信息。
2. 其他候选仅记录为短期候选，不进入长期 store。
3. 后续版本再加入 human approval 或前端确认。

### 4.16 finalize

职责：

构造最终 API 输出。

输出：

```json
{
  "message_id": "assistant-message-id",
  "answer": "...",
  "citations": [
    {
      "record_id": "uuid",
      "doi": "10.xxxx/xxxx",
      "source_text": "...",
      "score": 0.87
    }
  ],
  "retrieval_trace": {
    "should_retrieve": true,
    "query": "...",
    "filters": {},
    "top_k": 12
  },
  "memory_trace": {
    "short_term_updated": true,
    "long_term_written": false
  }
}
```

## 5. State Schema

建议 LangGraph state 包含以下字段：

```text
user_id: str
session_id: str
message_id: str

current_user_message: str
normalized_question: str
detected_entities: dict

runtime_options: dict

recent_messages: list
conversation_summary: str | null
active_materials: list[str]
active_constraints: list[str]
last_retrieval_refs: list[str]

long_term_memories: list[dict]

route_decision: dict
retrieval_plan: dict | null
retrieval_results: list[dict]
retrieval_sufficiency: dict | null

assistant_message: str
citations: list[dict]

short_term_updates: dict
long_term_candidates: list[dict]
long_term_writes: list[dict]

errors: list[dict]
trace: list[dict]
```

约束：

1. state 中不得保存密钥。
2. state 中不得保存过大的原始全文。
3. 检索结果中的 `source_text` 应限制长度。
4. 大对象应存 PostgreSQL 或 Milvus，通过 ID 引用。

## 6. Conditional Edges

`route_intent` 后的条件边：

```text
intent == "retrieve"      -> build_retrieval_plan
intent == "direct_answer" -> direct_answer
intent == "smalltalk"     -> direct_answer
intent == "clarify"       -> clarify_question
intent == "unsupported"   -> direct_answer
```

`judge_retrieval_sufficiency` 后的条件边：

```text
is_sufficient == true  -> answer_with_evidence
is_sufficient == false -> answer_with_limits
```

所有回答节点之后：

```text
answer_* -> propose_memory_update -> write_short_term_memory -> write_long_term_memory -> finalize
```

## 7. Streaming Design

`POST /api/rag/chat/stream` 应向前端输出 SSE。

建议事件类型：

```text
run_started
node_started
route_decision
retrieval_plan
retrieval_result
retrieval_sufficiency
token
citation
memory_update
final
error
run_finished
```

示例：

```json
{
  "event": "route_decision",
  "data": {
    "should_retrieve": true,
    "reason": "用户询问具体材料的生长条件"
  }
}
```

LangGraph 当前推荐使用 event streaming 作为多数应用代码的流式模型，因为它提供 messages、values、subgraphs、output、interrupts 等 typed projections。MVP 阶段 `rag-api` 可以消费 LangGraph event stream，再转换为前端稳定 SSE 协议。

## 8. Persistence Design

### 8.1 Short-term persistence

短期记忆：

```text
bounded session store
scope: session_id / thread_id
backend: SQLite on the small host, external PostgreSQL in production
```

用途：

1. 会话连续性。
2. 当前图状态恢复。
3. 长对话摘要。
4. 最近检索引用。
5. 后续 human-in-the-loop 能力。

注意：

1. 生产环境不得使用内存持久化。
2. `thread_id` 应使用 UUID，不超过 255 字符。
3. 需要 checkpoint retention policy，避免长期无限增长。

### 8.2 Long-term persistence

长期记忆：

```text
bounded structured store
scope: user_id namespace
backend: SQLite on the small host, external PostgreSQL in production
```

命名空间：

```text
(user_id, "growth_rag", "memories")
(user_id, "growth_rag", "preferences")
(user_id, "growth_rag", "lab_constraints")
```

长期记忆必须支持：

1. 创建。
2. 查询。
3. 修改。
4. 删除。
5. 标记来源。
6. 标记置信度。

## 9. Retrieval Design

MVP 检索模式：

```text
hybrid = dense vector + sparse keyword + metadata filter
```

输入来源：

1. 用户当前问题。
2. 短期记忆中的当前材料和约束。
3. 长期记忆中的用户偏好和实验室约束。

过滤字段：

```text
material_formula
material_name
growth_method
atmosphere
temperature_min
temperature_max
doi
source_file
```

召回策略：

1. 明确化学式时优先 metadata filter。
2. 化学式不确定时扩大语义召回。
3. 方法名明确时加入 filter 或 sparse query。
4. 温度和气氛既进入 filter，也进入 query text。
5. 无结果时放宽 filter 重试一次。

MVP 可先不做 rerank，但保留 `rerank.py` 和 `judge_retrieval_sufficiency` 节点，方便后续升级。

## 10. Error Handling

错误类型：

| Error | Handling |
| --- | --- |
| LLM timeout | 返回可恢复错误，可重试 |
| LLM invalid JSON | 进入修复解析或 fallback route |
| Milvus unavailable | 退化为无法检索，并说明系统暂时不可用 |
| PostgreSQL unavailable | 阻断请求，避免记忆和会话状态不一致 |
| No retrieval result | 进入 `answer_with_limits` |
| Low confidence answer | 明确提示证据不足 |

`errors` 必须写入 state trace，但不得向前端暴露内部密钥、连接串或堆栈详情。

## 11. Observability

每轮对话应记录：

1. `run_id`。
2. `user_id`。
3. `session_id`。
4. route decision。
5. retrieval plan。
6. retrieved record ids。
7. retrieval sufficiency。
8. model name。
9. token usage when available。
10. latency。
11. errors。

建议后续接入 LangSmith 或等价 tracing；MVP 可以先落 PostgreSQL 日志表。

## 12. File Mapping

设计对应实现文件：

```text
services/rag-api/src/agent/
  graph.py
  state.py
  prompts.py

services/rag-api/src/agent/nodes/
  ingest_input.py
  load_short_term_memory.py
  load_long_term_memory.py
  normalize_question.py
  route_intent.py
  clarify_question.py
  direct_answer.py
  build_retrieval_plan.py
  retrieve_growth_records.py
  judge_retrieval_sufficiency.py
  answer_with_evidence.py
  answer_with_limits.py
  propose_memory_update.py
  write_short_term_memory.py
  write_long_term_memory.py
  finalize.py

services/rag-api/src/retrieval/
  embeddings.py
  milvus_hybrid.py
  growth_record_text.py
  rerank.py

services/rag-api/src/memory/
  short_term.py
  long_term.py

services/rag-api/src/api/
  chat.py
```

## 13. MVP Build Order

推荐实现顺序：

1. 定义 state schema。
2. 实现 graph skeleton。
3. 实现 `route_intent` 的结构化输出。
4. 实现 direct answer 和 clarify。
5. 实现 retrieval plan。
6. 实现 Milvus retrieval adapter。
7. 实现 answer with evidence。
8. 接入 checkpointer。
9. 接入长期 memory store 的只读能力。
10. 增加受控长期记忆写入。
11. 接入 SSE streaming。
12. 增加 trace 和错误处理。

## 14. Non-goals for v0.1

v0.1 暂不做：

1. 多智能体协作。
2. 子图拆分。
3. 自动实验方案执行。
4. 无限制长期记忆写入。
5. 复杂 human approval。
6. 自动文献下载。
7. 训练或微调模型。
8. 检索平台和模型推理平台的正式功能。

## 15. Formal Node Definitions

本节是 v0.1 的关键节点定义。实现时以本节为准，上文的 Graph Nodes 章节作为解释性说明。

### 15.1 Node Function Convention

所有节点遵守同一函数契约：

```text
async def node_name(
    state: GrowthRAGState,
    config: RunnableConfig
) -> PartialGrowthRAGState
```

约束：

1. 节点只返回需要更新的 state 字段，不返回完整 state。
2. 节点不得修改输入 state 对象本身。
3. 节点不得读取前端环境变量。
4. 节点不得直接访问未封装的数据库连接。
5. 外部服务访问必须通过 service adapter，例如 `LLMClient`、`RetrievalService`、`MemoryStore`。
6. 每个节点必须向 `trace` 写入最小可观测信息。
7. 节点失败时应返回结构化 `errors`，除非是必须中断请求的基础设施错误。

节点返回格式约定：

```text
{
  "field_to_update": value,
  "trace": [trace_event],
  "errors": [error_event]
}
```

### 15.2 Node Index

| Node | Type | Required Inputs | Writes | External IO | Next |
| --- | --- | --- | --- | --- | --- |
| `ingest_input` | deterministic | request payload | current input, runtime options | no | `load_short_term_memory` |
| `load_short_term_memory` | IO | user_id, session_id | recent messages, summary, active context | PostgreSQL/checkpointer | `load_long_term_memory` |
| `load_long_term_memory` | IO | user_id | long-term memories | PostgreSQL/store | `normalize_question` |
| `normalize_question` | LLM or deterministic hybrid | user message, recent context | normalized question, detected entities | optional LLM | `route_intent` |
| `route_intent` | LLM structured | normalized question, memory, options | route decision | LLM | conditional |
| `clarify_question` | LLM or template | route decision, missing slots | assistant message | optional LLM | `propose_memory_update` |
| `direct_answer` | LLM | question, memory | assistant message | LLM | `propose_memory_update` |
| `build_retrieval_plan` | LLM structured | normalized question, entities, memory | retrieval plan | LLM | `retrieve_growth_records` |
| `retrieve_growth_records` | IO | retrieval plan | retrieval results | Milvus, embedding model | `judge_retrieval_sufficiency` |
| `judge_retrieval_sufficiency` | LLM structured or rules | question, plan, results | sufficiency decision | optional LLM | conditional |
| `answer_with_evidence` | LLM streaming | question, results, memory | assistant message, citations | LLM | `propose_memory_update` |
| `answer_with_limits` | LLM streaming | question, partial results | assistant message, citations | LLM | `propose_memory_update` |
| `propose_memory_update` | LLM structured + rules | conversation turn | memory candidates | optional LLM | `write_short_term_memory` |
| `write_short_term_memory` | IO | assistant message, trace | short-term state | checkpointer/PostgreSQL | `write_long_term_memory` |
| `write_long_term_memory` | IO + rules | long-term candidates | long-term writes | PostgreSQL/store | `finalize` |
| `finalize` | deterministic | final state | response envelope | no | END |

### 15.3 ingest_input

Purpose:

```text
把 API request payload 转换为 graph state 的本轮输入。
```

Required state fields:

```text
user_id
session_id
message_id
current_user_message
runtime_options
```

Input contract:

```json
{
  "user_id": "string",
  "session_id": "string",
  "message_id": "string",
  "message": "string",
  "options": {
    "force_retrieve": false,
    "top_k": 12,
    "retrieval_mode": "hybrid",
    "model": "string | null",
    "stream_trace": true
  }
}
```

Writes:

```text
current_user_message
runtime_options
trace
```

Validation rules:

1. `message` 不能为空。
2. `session_id` 必须存在；如果 API 层未提供，应由 API 层创建。
3. `top_k` 默认 12，最大不超过 30。
4. `retrieval_mode` 只允许 `dense`、`sparse`、`hybrid`。
5. `force_retrieve` 只允许 boolean。

Failure behavior:

| Failure | Behavior |
| --- | --- |
| Empty message | return validation error and stop |
| Invalid `top_k` | clamp to allowed range |
| Invalid retrieval mode | fallback to `hybrid` |

### 15.4 load_short_term_memory

Purpose:

```text
恢复当前会话 thread state，并提取本轮决策需要的短期上下文。
```

Reads:

```text
session_id
user_id
```

Writes:

```text
recent_messages
conversation_summary
active_materials
active_constraints
last_retrieval_refs
trace
```

External IO:

```text
LangGraph checkpointer
PostgreSQL messages table
```

Read policy:

1. 最近消息默认读取 12 条。
2. 如果存在会话摘要，摘要优先进入 state。
3. 最近引用数据条默认读取最近 5 个 `record_id`。
4. 如果 checkpoint 和 messages table 信息冲突，以 messages table 为审计源，以 checkpoint 为运行态源。

Failure behavior:

| Failure | Behavior |
| --- | --- |
| Checkpoint missing | initialize empty short-term state |
| Messages missing | continue with empty recent messages |
| PostgreSQL unavailable | hard fail |

### 15.5 load_long_term_memory

Purpose:

```text
读取跨会话用户长期记忆，用于检索决策、回答风格和实验条件约束。
```

Reads:

```text
user_id
normalized_question optional
```

Writes:

```text
long_term_memories
trace
```

External IO:

```text
LangGraph store or PostgreSQL-backed memory store
```

Namespaces:

```text
(user_id, "growth_rag", "memories")
(user_id, "growth_rag", "preferences")
(user_id, "growth_rag", "lab_constraints")
```

Selection policy:

1. 读取最多 20 条长期记忆。
2. 优先读取 `user_confirmed` 来源。
3. 优先读取与当前材料、方法、温度、气氛相关的记忆。
4. 低置信度 inferred memory 只作为弱提示，不作为硬约束。

Failure behavior:

| Failure | Behavior |
| --- | --- |
| Store unavailable | continue without long-term memory and record warning |
| Memory parse error | skip invalid memory |

### 15.6 normalize_question

Purpose:

```text
把用户当前问题和短期上下文合并成可检索、可决策的规范问题。
```

Reads:

```text
current_user_message
recent_messages
conversation_summary
active_materials
active_constraints
long_term_memories
```

Writes:

```text
normalized_question
detected_entities
trace
```

Output contract:

```json
{
  "normalized_question": "string",
  "detected_entities": {
    "materials": ["string"],
    "formulas": ["string"],
    "growth_methods": ["string"],
    "temperature_mentions": ["string"],
    "atmosphere_mentions": ["string"],
    "precursor_mentions": ["string"],
    "task_type": "explain | retrieve | compare | recommend | summarize | unknown"
  }
}
```

Implementation policy:

1. MVP 可先用规则 + LLM 结构化输出。
2. 化学式识别应保留大小写，例如 `Mn3GaN` 不得改写成自然语言。
3. 对“它”“这个材料”“刚才那个方法”等指代，应结合短期记忆补全。
4. 如果无法补全，不得强行猜测，应交给 `route_intent` 进入澄清。

Failure behavior:

| Failure | Behavior |
| --- | --- |
| LLM invalid JSON | fallback to original user message and empty entities |
| Ambiguous reference | keep ambiguity in `detected_entities` |

### 15.7 route_intent

Purpose:

```text
决定本轮走直接回答、检索回答、澄清追问、闲聊回复或拒绝/不支持路径。
```

Reads:

```text
current_user_message
normalized_question
detected_entities
runtime_options
recent_messages
long_term_memories
```

Writes:

```text
route_decision
trace
```

Output contract:

```json
{
  "intent": "retrieve",
  "should_retrieve": true,
  "answer_mode": "evidence_grounded",
  "reason": "用户询问具体材料的单晶生长条件，需要检索数据条",
  "missing_slots": [],
  "confidence": 0.91
}
```

Allowed values:

```text
intent:
  direct_answer
  retrieve
  clarify
  smalltalk
  unsupported

answer_mode:
  direct
  evidence_grounded
  ask_clarification
  refuse_or_redirect
```

Hard routing rules:

1. `force_retrieve=true` 必须设置 `intent=retrieve`。
2. 询问具体材料、化学式、温度程序、气氛、助熔剂、运输剂、晶体尺寸、文献记录，默认 `retrieve`。
3. 询问“什么是 CVT”“助熔剂法是什么”这类通用概念，默认 `direct_answer`。
4. 用户要求“帮我找一个方法”但没有材料或材料体系，默认 `clarify`。
5. 用户要求与单晶生长无关的任务，默认 `unsupported` 或 `direct_answer` 简短说明能力边界。

Conditional edge mapping:

```text
retrieve      -> build_retrieval_plan
direct_answer -> direct_answer
smalltalk     -> direct_answer
clarify       -> clarify_question
unsupported   -> direct_answer
```

Failure behavior:

| Failure | Behavior |
| --- | --- |
| LLM invalid JSON | fallback to rules |
| Low confidence and missing material | clarify |
| Low confidence but `force_retrieve=true` | retrieve |

### 15.8 clarify_question

Purpose:

```text
当缺少必要信息时，向用户提出最少数量的澄清问题。
```

Reads:

```text
route_decision
normalized_question
detected_entities
active_materials
active_constraints
```

Writes:

```text
assistant_message
citations = []
trace
```

Output constraints:

1. 一次最多问 2 个问题。
2. 优先询问目标材料或材料体系。
3. 如果用户目标是实验方案，优先询问最高温度、气氛、是否接受助熔剂法。
4. 不输出检索引用。

Failure behavior:

| Failure | Behavior |
| --- | --- |
| LLM unavailable | use deterministic clarification template |

### 15.9 direct_answer

Purpose:

```text
回答无需检索的问题，或说明系统能力边界。
```

Reads:

```text
normalized_question
recent_messages
long_term_memories
route_decision
```

Writes:

```text
assistant_message
citations = []
trace
```

Output constraints:

1. 不得引用不存在的数据条。
2. 不得假装已检索。
3. 涉及具体材料实验条件时，应建议切换到检索路径，或由 route 直接进入检索路径。

Failure behavior:

| Failure | Behavior |
| --- | --- |
| LLM timeout | return recoverable error |

### 15.10 build_retrieval_plan

Purpose:

```text
把规范问题转换为 Milvus 可执行的检索计划。
```

Reads:

```text
normalized_question
detected_entities
active_materials
active_constraints
long_term_memories
runtime_options
```

Writes:

```text
retrieval_plan
trace
```

Output contract:

```json
{
  "query_text": "string",
  "dense_query": "string",
  "sparse_query": "string",
  "filters": {
    "material_formula": "string | null",
    "material_name": "string | null",
    "growth_method": "string | null",
    "temperature_min": "number | null",
    "temperature_max": "number | null",
    "atmosphere": "string | null",
    "doi": "string | null"
  },
  "top_k": 12,
  "retrieval_mode": "hybrid",
  "relaxation_policy": {
    "allow_filter_relaxation": true,
    "relax_order": ["temperature", "atmosphere", "growth_method", "material_name"]
  },
  "must_have": ["material_formula"],
  "nice_to_have": ["temperature_program", "growth_method", "atmosphere"]
}
```

Planning rules:

1. 化学式确定时进入 `filters.material_formula`。
2. 化学式不确定时不要强行 filter，放入 query。
3. 方法名确定时可以进入 `growth_method` filter。
4. 温度范围可以进入 filter，但必须允许放宽。
5. `top_k` 由 `runtime_options.top_k` 决定，但上限 30。

Failure behavior:

| Failure | Behavior |
| --- | --- |
| LLM invalid JSON | construct simple plan from normalized question |
| Empty query | route to clarify by adding error and empty plan |

### 15.11 retrieve_growth_records

Purpose:

```text
执行单晶生长数据条检索。
```

Reads:

```text
retrieval_plan
```

Writes:

```text
retrieval_results
trace
errors optional
```

External IO:

```text
Embedding model
Milvus
PostgreSQL metadata lookup optional
```

Output contract:

```json
[
  {
    "record_id": "string",
    "score": 0.87,
    "dense_score": 0.81,
    "sparse_score": 0.76,
    "material_formula": "Mn3GaN",
    "material_name": "string | null",
    "growth_method": "flux growth",
    "temperature_program": "string | null",
    "atmosphere": "string | null",
    "doi": "string | null",
    "source_text": "string",
    "source_file": "string | null",
    "matched_fields": ["material_formula", "temperature_program"]
  }
]
```

Retrieval behavior:

1. 先按原始 plan 检索。
2. 如果无结果且 `allow_filter_relaxation=true`，按 `relax_order` 放宽过滤条件重试一次。
3. source_text 应截断到可展示长度，完整原文通过 `record_id` 再查。
4. 返回结果按综合分数排序。

Failure behavior:

| Failure | Behavior |
| --- | --- |
| Embedding service unavailable | if sparse available, fallback to sparse search |
| Milvus unavailable | return hard retrieval error and route to limited answer |
| No results | return empty list, not exception |

### 15.12 judge_retrieval_sufficiency

Purpose:

```text
判断检索结果是否足以支撑回答。
```

Reads:

```text
normalized_question
retrieval_plan
retrieval_results
detected_entities
```

Writes:

```text
retrieval_sufficiency
trace
```

Output contract:

```json
{
  "is_sufficient": true,
  "reason": "找到多条目标材料记录且包含温度程序",
  "usable_record_ids": ["record-1", "record-2"],
  "missing_evidence": [],
  "answer_strategy": "compare_and_summarize",
  "confidence": 0.84
}
```

Decision rules:

1. `retrieval_results` 为空，`is_sufficient=false`。
2. 用户问具体字段，而结果没有该字段，`is_sufficient=false`。
3. 用户要求对比，至少需要 2 条可用记录。
4. 用户要求推荐实验条件，必须存在温度、方法、原料或气氛中的关键证据。
5. 检索结果材料与用户目标材料不一致时，`is_sufficient=false`，除非问题明确要求相近体系。

Conditional edge mapping:

```text
is_sufficient == true  -> answer_with_evidence
is_sufficient == false -> answer_with_limits
```

Failure behavior:

| Failure | Behavior |
| --- | --- |
| LLM invalid JSON | fallback to rules |
| Conflicting evidence | sufficient with answer_strategy=`compare_conflicts` |

### 15.13 answer_with_evidence

Purpose:

```text
基于检索证据生成可追溯回答。
```

Reads:

```text
normalized_question
retrieval_results
retrieval_sufficiency
recent_messages
long_term_memories
```

Writes:

```text
assistant_message
citations
trace
```

Output contract:

```json
{
  "assistant_message": "string",
  "citations": [
    {
      "record_id": "string",
      "doi": "string | null",
      "source_text": "string",
      "score": 0.87
    }
  ]
}
```

Answer rules:

1. 必须引用 `record_id`。
2. 有 DOI 时同时引用 DOI。
3. 不得输出检索结果中不存在的实验条件。
4. 推断性建议必须标注为推断。
5. 数据冲突时必须列出冲突。
6. 实验建议必须说明适用边界。

Streaming:

1. 文本 token 通过 `token` 事件输出。
2. 引用通过 `citation` 事件输出。
3. 回答完成后写入 `assistant_message`。

Failure behavior:

| Failure | Behavior |
| --- | --- |
| LLM timeout | return partial/error event |
| Citation missing | block finalization and return error |

### 15.14 answer_with_limits

Purpose:

```text
在证据不足时生成受限回答，避免模型硬编。
```

Reads:

```text
normalized_question
retrieval_results
retrieval_sufficiency
retrieval_plan
```

Writes:

```text
assistant_message
citations
trace
```

Answer rules:

1. 第一段明确说明证据不足。
2. 如果有部分结果，列出可用部分。
3. 明确缺失字段，例如温度程序、气氛、助熔剂。
4. 给出下一步追问或扩大检索建议。
5. 不得给出确定性实验方案。

Failure behavior:

| Failure | Behavior |
| --- | --- |
| LLM unavailable | use deterministic limited-answer template |

### 15.15 propose_memory_update

Purpose:

```text
生成短期记忆更新和长期记忆候选。
```

Reads:

```text
current_user_message
assistant_message
normalized_question
detected_entities
retrieval_results
route_decision
```

Writes:

```text
short_term_updates
long_term_candidates
trace
```

Output contract:

```json
{
  "short_term_updates": {
    "active_materials": ["Mn3GaN"],
    "active_constraints": ["用户关注温度程序"],
    "last_retrieval_refs": ["record-1", "record-2"]
  },
  "long_term_candidates": [
    {
      "type": "research_interest",
      "content": "用户关注 Mn3GaN 单晶生长",
      "source": "inferred",
      "confidence": 0.72,
      "write_policy": "defer_until_repeated_or_confirmed"
    }
  ]
}
```

Rules:

1. 当前材料、当前约束、最近引用记录可以写入短期记忆。
2. 长期候选必须有 `source`、`confidence`、`write_policy`。
3. 用户明确说“记住”时，`write_policy=write_now`。
4. 普通推断默认 `defer_until_repeated_or_confirmed`。
5. 敏感信息不得写入长期候选。

Failure behavior:

| Failure | Behavior |
| --- | --- |
| LLM invalid JSON | write only deterministic short-term updates |

### 15.16 write_short_term_memory

Purpose:

```text
持久化当前会话的短期记忆和对话状态。
```

Reads:

```text
short_term_updates
current_user_message
assistant_message
citations
retrieval_plan
retrieval_results
```

Writes:

```text
memory_trace.short_term_updated
trace
```

External IO:

```text
LangGraph checkpointer
PostgreSQL messages table
```

Write policy:

1. 用户消息和助手消息写入 messages table。
2. 当前材料和约束写入 graph state。
3. 最近引用记录最多保留 20 个。
4. 长对话达到阈值时触发摘要更新。

Failure behavior:

| Failure | Behavior |
| --- | --- |
| Checkpointer write failed | hard fail |
| Message audit write failed | hard fail |

### 15.17 write_long_term_memory

Purpose:

```text
按长期记忆策略写入用户级 memory store。
```

Reads:

```text
long_term_candidates
user_id
```

Writes:

```text
long_term_writes
memory_trace.long_term_written
trace
```

External IO:

```text
LangGraph store or PostgreSQL-backed memory store
```

Write rules:

1. `write_policy=write_now` 可以写入。
2. `source=user_confirmed` 可以写入。
3. `source=explicit_remember_request` 可以写入。
4. `defer_until_repeated_or_confirmed` 不写入，只记录 trace。
5. `sensitive_or_private` 不写入。

Memory record contract:

```json
{
  "memory_id": "string",
  "user_id": "string",
  "namespace": ["user_id", "growth_rag", "memories"],
  "type": "research_interest | preference | lab_constraint | fact",
  "content": "string",
  "source": "user_confirmed | explicit_remember_request | inferred",
  "confidence": 0.95,
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

Failure behavior:

| Failure | Behavior |
| --- | --- |
| Store unavailable | continue response, record warning |
| Duplicate memory | update existing memory instead of inserting duplicate |

### 15.18 finalize

Purpose:

```text
把 graph state 转换为 API 最终响应。
```

Reads:

```text
assistant_message
citations
route_decision
retrieval_plan
retrieval_results
retrieval_sufficiency
memory_trace
errors
trace
```

Writes:

```text
final_response
trace
```

Output contract:

```json
{
  "message_id": "string",
  "session_id": "string",
  "answer": "string",
  "citations": [],
  "retrieval_trace": {
    "should_retrieve": true,
    "query": "string",
    "filters": {},
    "top_k": 12,
    "result_count": 5
  },
  "memory_trace": {
    "short_term_updated": true,
    "long_term_written": false
  },
  "errors": []
}
```

Finalization rules:

1. 如果 answer 节点生成了 citations，final response 必须保留。
2. 如果 route 为 retrieve，必须返回 retrieval trace。
3. 如果有非致命错误，放入 `errors`，不得暴露内部堆栈。
4. 如果是致命错误，API 层转换为对应 HTTP error 或 SSE error event。

## 16. References

1. LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
2. LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
3. LangGraph memory overview: https://docs.langchain.com/oss/python/concepts/memory
4. LangGraph stores: https://docs.langchain.com/oss/python/langgraph/stores
5. LangGraph event streaming: https://docs.langchain.com/oss/python/langgraph/event-streaming
6. LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
