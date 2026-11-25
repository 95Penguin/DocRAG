# DocRAG

面向科研论文与技术长文档问答的检索增强与证据驱动监督微调项目。

项目关注两个问题：检索结果能否覆盖回答所需的完整证据，以及模型面对噪声、缺失证据和
不可回答问题时，能否给出有依据的答案、准确引用或合理拒答。计划基于 QASPER，比较
层级检索、混合检索和多证据状态 SFT 对长文档问答效果的影响。

仓库地址：[95Penguin/DocRAG](https://github.com/95Penguin/DocRAG)。

## 当前状态

本仓库由 SupportPilot 技术支持 Demo、Knowledge Studio 多知识库系统演进而来。
现有代码已实现 Spring Boot、FastAPI 与 React 在线 RAG 系统；QASPER 数据处理、
Parent-Child 层级索引、BGE-M3 + FAISS 和 Qwen3-8B QLoRA 尚待实现，当前没有这些
算法实验的训练结果或性能结论。

| 部分 | 状态 | 说明 |
| --- | --- | --- |
| 在线知识库系统 | 已实现 | 文档管理、结构分块、多路检索、引用回答、会话与 SSE |
| 检索与回答评测 | 已有基础 | Redis 固定评测集、上传知识库回归评测与逐题报告 |
| QASPER 数据准备 | 待实现 | 官方划分、稳定 ID、证据映射与数据统计 |
| 长文档检索实验 | 待实现 | 固定切块 Baseline、Parent-Child、BGE-M3、FAISS |
| 证据条件 SFT | 待实现 | 四种证据状态数据、QLoRA、普通 SFT 对照 |
| 鲁棒性与消融实验 | 待实现 | 噪声、证据缺失、引用质量与错误拒答分析 |

在线系统中的 E5、标题图检索和只读 Agent 属于现有工程能力。算法实验按独立配置开展，
暂不涉及强化学习、Agent 决策训练或多轮自适应检索。

## 研究目标与实验计划

### 数据与检索

使用 [QASPER](https://huggingface.co/datasets/allenai/qasper) 官方训练、验证和测试划分，
检查论文级数据泄漏，保留章节、段落、多个参考答案及原始证据标注。初期限定已知论文内
检索，直接解析结构化正文，并建立文档、章节、段落、父块与子块的稳定 ID 和映射关系。

- Baseline：固定 300-token Chunk、50-token Overlap、BGE-M3、FAISS、Top5、原始 Qwen3-8B。
- 检索优化：Parent-Child、BM25 与 Dense 双路 Top20 召回、加权 RRF（`k=60`）、BGE Reranker。
- 父块回溯：按 Parent ID 去重，在统一 Token 预算内组织最终证据。
- 查询改写与子问题拆解：可选，通过独立实验决定是否保留。
- 生成：统一 Qwen Chat Template、非思考模式、固定解码配置和上下文预算。

### 多证据状态 SFT

| 类型 | 输入证据 | 监督目标 |
| --- | --- | --- |
| A：充分证据 | 支撑完整答案的证据集合 | 正确回答并引用有效证据 |
| B：噪声证据 | 正确证据与经过检查的 Hard Negative | 排除干扰，正确回答并引用 |
| C：缺失证据 | 经验证缺少必要信息的部分证据 | 判断当前证据不足并合理拒答 |
| D：不可回答 | 原始不可回答问题及实际检索结果 | 拒答，避免编造答案与引用 |

计划采用 `messages` 格式，Assistant 输出统一为：

```json
{
  "answer": "基于当前证据的答案",
  "evidence_ids": ["p_001", "p_005"],
  "is_answerable": true
}
```

`is_answerable` 表示当前提供的证据是否足以支持完整回答。证据不足时设为 `false`，
返回明确的不足说明及空引用列表。引用 ID 必须来自实际输入。未标注段落不能直接视为负样本，
移除标注证据也不能直接视为证据不足；数据构造需检查替代证据、假负样本和截断情况。

### 实验与指标

| 实验 | 检索方案 | 生成模型 |
| --- | --- | --- |
| A | 固定切块 + 单路向量检索 | 原始 Qwen3-8B |
| B | Parent-Child + 混合检索 + Rerank | 原始 Qwen3-8B |
| C | 优化检索 | 仅充分证据 SFT |
| D | 优化检索 | 多证据状态 SFT |

另设 Oracle Evidence 参考实验，直接提供人工标注的正确证据，分析检索与生成误差。
主实验保持相同测试集、Prompt、解码配置和上下文预算；两种 SFT 尽量保持相同训练 Token
预算与优化配置，仅用验证集调参。受控噪声测试设置 0%、25%、50%、75% 噪声，保留必要
正确证据并固定比例计算口径；证据缺失测试需验证移除的证据确实必要。两类受控测试与
真实检索结果上的端到端评估分别报告。主实验完成后，按资源补充检索和训练数据消融。

计划指标包括 Evidence Recall@5/@10、Evidence Precision、完整证据覆盖率、QASPER
官方 Answer F1、Citation Precision/Recall、Faithfulness、无依据回答率、错误拒答率、
不可回答问题的错误回答率、证据不足样本的合理拒答率，以及耗时、Token 数和训练峰值显存。
证据指标优先映射回原始标注段落，避免 Chunk 粒度影响比较。引用 ID 合法不等于语义支持，
参考答案不匹配也不直接等于幻觉；语义评价需明确规则并通过人工抽查验证。

### 训练与开发路线

目标服务器为单卡 **RTX 3090 24GB**。计划采用 Qwen3-8B + QLoRA，初始配置为
4-bit NF4、Double Quantization、LoRA `r=16 / alpha=32 / dropout=0.05`、学习率
`1e-4`、最大序列长度 `2048`、Batch Size `1`、梯度累积 `8` 或 `16`、训练 `2` 个 Epoch，
开启 Gradient Checkpointing，并在环境支持时使用 BF16。参数以服务器兼容性与显存实测为准。

正式训练前，用约 100 条样本与小规模 Qwen 完成 Smoke Test，检查 Assistant-only Loss
Masking、前后向传播、截断、显存和权重保存。现有在线服务的 Linux PyTorch 依赖指向 CPU
源，后续算法模块需使用独立 CUDA 环境与依赖配置。

1. **Stage 1：环境与数据。** 检查服务器，解析 QASPER，建立稳定 ID 和证据映射，输出真实统计与样例。
2. **Stage 2：Baseline。** 跑通固定切块、检索、生成和基础评价，分析失败案例。
3. **Stage 3：检索优化。** 实现层级索引，复用混合检索与重排，完成对比实验。
4. **Stage 4：训练数据。** 构造四种证据状态，输出质量报告和标准 SFT 文件。
5. **Stage 5：监督微调。** 完成 Smoke Test，训练普通证据 SFT 与多证据状态 SFT。
6. **Stage 6：实验评估。** 完成主实验、噪声与缺失鲁棒性实验，按资源补充消融。
7. **Stage 7：整理与集成。** 固化配置、脚本与报告，按实验结论选择性接入在线系统。

按阶段验收并记录实际命令、结果和问题；正式训练前检查 GPU 可用资源。大型数据、模型权重
与索引不直接提交到 Git。后续建立算法模块时维护 `algorithm/PROGRESS.md`。

## 已实现的在线系统

现有主流程：

```text
创建知识库 -> 上传文档 -> 解析与结构分块 -> 建立关键词/向量/结构图索引
-> 模型理解问题、改写与检索规划 -> 多路召回 -> RRF 融合 -> Cross Encoder 精排
-> 跨片段综合回答与服务端引用校验 -> SSE 事件流 -> 引用与推荐追问
```

Redis 7.4 官方文档保留为内置示例库。用户还可以分别创建“项目文档”“课程资料”等多个
知识库，上传 Markdown、TXT、HTML 或 PDF 后独立检索，数据按用户与知识库隔离。

### 功能

- 多知识库管理：创建、列出、修改名称和描述、文档上传、替换与删除、按知识库限定检索。
  聊天页顶部只显示当前库信息；侧边栏每个用户知识库的“管理”按钮打开编辑和文档管理面板。
- 对话历史：按用户持久化知识库问答、引用与耗时；左侧可打开历史会话继续追问。
- 文档处理：Markdown/TXT/HTML 解析、PDF 文本提取与扫描页 OCR 兜底，按标题层级和段落
  组织文本，对超长普通文本尝试按句拆分，保留重叠上下文与来源元数据。OCR 由 Docker 镜像提供。
- 索引生命周期：文档以 SHA-256 去重，替换和删除后重建 BM25 索引，只对新增或变更的
  Chunk 重新生成 E5 Embedding；整套索引写入新版本后再原子切换，失败不影响当前版本。
- 多路召回：关键词、E5 向量和标题层级图检索，通过稳定 `chunk_id` 去重并执行
  加权 RRF 融合。
- 精排与回答：BGE Cross Encoder 精排；模型只能引用程序提供的证据 ID，引用在服务端
  二次校验，低相关度或非法引用会拒答。
- 问题理解：模型结合最近对话，把省略主语的追问改写成独立问题，判断事实、概览、比较或
  多跳意图，并生成互补的检索查询；规划失败时退回可检查的规则改写。
- 知识库 Agent：简单事实问题走快速 RAG；概览、比较、原因等复杂问题由模型在每次观察
  工具结果后选择下一步，默认最多执行 3 步，并保留最终回答的时间预算。只读工具包括知识库目录、限定范围的精排检索、
  章节阅读和相邻 Chunk 阅读。最终由模型综合证据生成回答，而不是把片段直接输出；证据
  ID 与引用仍由服务端校验。Agent 不能修改文档，也不调用旧工单流程的业务工具。
- 多轮与流式事件：服务端从会话库加载最近对话，工具完成时立即推送 `tool` 事件；`status`、`trace`、
  `answer_delta`、`citation`、`follow_up`、
  `complete`、`error` 等 SSE 事件分别返回。答案文本必须通过引用校验后才推送，因此不是未校验的模型 token 流。
- 可观测与评测：请求级 Trace 记录检索、重排、生成耗时和 Token；仓库包含 50 条检索
  评测集、MRR/NDCG/Recall 对比脚本与失败样本分析。
- MCP 工具：Spring Boot 暴露 `/mcp` JSON-RPC 端点，支持 `initialize`、`tools/list`、
  `tools/call`；Python Agent 会发现 Schema、校验参数并调用 Java 工具。原技术支持工单
  流程作为 MCP 与 Human in the Loop 的扩展示例保留，不是通用知识库首页的依赖。

## 项目结构

```text
frontend/       React + Vite：知识库、上传、SSE 问答、引用、Trace
ai-service/     FastAPI：解析分块、检索、RAG、Agent、MCP Client、评测
backend/        Spring Boot：登录鉴权、MCP Server、业务工具权限与审计
data/           内置示例语料、索引、评测集与实验结果
docs/           现有系统的需求、检索实验记录与架构说明
knowledge/      补充知识文档
scripts/        现有语料构建、索引与检索评测脚本
```

计划新增独立 `algorithm/` 目录，容纳数据处理、检索实验、训练、推理、评测、配置与测试。
该目录当前尚未创建，算法实验通过验证后再考虑接入在线系统。现有 `support-pilot-*`
命令、Java 包名与 Docker 镜像名暂时沿用，仓库改名不影响这些入口。

Spring Boot 与 FastAPI 的职责边界：Java 负责用户身份和需要强权限、事务、审计的业务工具；
Python 负责模型、文档处理、检索和 Agent 编排。React 在本地开发时分别请求两个服务，
生产构建由 FastAPI 同源托管。

## 两种检索运行模式

`RETRIEVAL_MODE=local` 是默认可复现模式：关键词索引、E5 向量和标题图均保存到本地文件，
用于本地演示和离线评测。Dense 当前采用 E5 向量全量点积扫描，不是 FAISS 索引。

`RETRIEVAL_MODE=external` 使用 Elasticsearch 关键词检索与 Milvus 向量检索，并继续使用
标题图召回、RRF 和 BGE 精排。配置 `ELASTICSEARCH_*` 与 `MILVUS_*` 后运行：

```dotenv
RETRIEVAL_MODE=external
ELASTICSEARCH_URL=http://127.0.0.1:9200
ELASTICSEARCH_INDEX=knowledge_studio_chunks
ELASTICSEARCH_API_KEY=
MILVUS_URL=http://127.0.0.1:19530
MILVUS_COLLECTION=knowledge_studio_chunks
MILVUS_TOKEN=root:Milvus
```

本地 Elasticsearch 未开启安全认证时 `ELASTICSEARCH_API_KEY` 留空；若开启认证，填写
API Key 的实际值，不要添加 `ApiKey ` 前缀。`root:Milvus` 是常见本地 Standalone
初始化凭据，如部署时修改过账号密码，就填写实际的 `用户名:密码`。

```bash
cd ai-service
uv run --env-file ../.env --extra embeddings support-pilot-sync-external
```

这两种模式共用相同的 `Retriever` 接口和下游链路。上述 `127.0.0.1` 地址用于在宿主机
直接运行 Python；Docker Compose 会自动改用服务名 `elasticsearch` 和 `milvus`。
当前 ES/Milvus 外部索引用于内置示例语料；用户上传的独立知识库仍使用本地索引。

## 文档更新与授权资料导入

在侧边栏选中知识库并点击“管理”后，可编辑信息、上传、替换或删除文档。上传支持 `.md`、`.txt`、`.html`、
`.htm` 和 `.pdf`，单文件不超过 10 MiB；PDF 最多 100 页，其中扫描页 OCR 最多 20 页。
OCR 使用 `pdftoppm` 和 Tesseract（英语、简体中文），Docker 镜像已安装。宿主机直接运行
Python 时，处理扫描件也需安装这些命令；普通文字 PDF 不依赖 OCR。

上传表单可填写原始来源 URL，系统**仅将其作为出处保存，不会访问或抓取 URL**。例如获得
书面许可的网页导出内容，可以作为 HTML/Markdown 文件上传并记录原始页面 URL；后续检索
和引用仍基于上传文件的原文。旧知识库无需手工迁移，首次更新文档时会生成新版索引；旧版本
保留在知识库目录的 `revisions/` 中用于短期恢复，不再参与查询。默认保留当前版和前两版；
后续更新会清理更旧的版本，不能把这个机制当作长期备份。删除先从当前知识库和检索中
移除，旧快照中的内容会随版本轮换被清理。

仅导入有权使用的资料，并保留来源信息；上传接口不会批量抓取外部站点。

## 在线系统启动

### 获取代码与配置

```bash
git clone https://github.com/95Penguin/DocRAG.git
cd DocRAG
```

已有本地副本可继续使用原目录名。以下准备步骤均从仓库根目录执行。
源码启动需要 Python 3.11～3.13、uv、JDK 17、Maven，以及满足 Vite 7 要求的 Node.js/npm。
容器启动需要 Docker Compose，并需先在宿主机完成前端构建。

复制环境变量并配置 OpenAI Compatible 模型：

```bash
cp .env.example .env
```

编辑 `.env` 中的 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`。
模型服务需支持 `/chat/completions` 和 JSON Object 输出；`LLM_BASE_URL` 填 API 前缀，
不要包含 `/chat/completions`。此处配置用于现有在线系统，不代表已接入微调后的 Qwen。

### 内置索引准备

当前问答服务启动时会加载内置 BM25 与 E5 索引。确认以下文件可用：

- `data/indexes/bm25/index.json`
- `data/indexes/dense/multilingual-e5-base.json`

仓库中已有索引可直接使用。若需重建，在根目录执行：

```bash
bash scripts/fetch_redis_docs.sh
bash scripts/build_corpus.sh
bash scripts/build_bm25.sh
bash scripts/build_dense.sh
```

下载脚本会记录上游 commit；复现实验时需使用相同语料快照与配置。原始 Redis 文档和
`data/processed/chunks.jsonl` 不随普通 Git 克隆提供，外部索引同步前需先准备这些数据。

### 源码启动

以下三个服务分别在独立终端启动，每段命令均从仓库根目录执行。

启动 Java：

```bash
cd backend
mvn spring-boot:run
```

启动 Python：

```bash
cd ai-service
uv sync --extra dev --extra embeddings
uv run --env-file ../.env uvicorn app.main:app --reload --port 8001
```

开发 React 页面：

```bash
cd frontend
npm install
npm run dev
```

访问 `http://127.0.0.1:5173`。也可以先运行 `npm run build`，再访问
`http://127.0.0.1:8001`，FastAPI 会优先托管 `frontend/dist`。
本地初始化账号为 `demo@supportpilot.dev`，密码为 `demo1234`。

Java 源码启动默认使用本地 H2 数据库；Docker Compose 使用 PostgreSQL。

### Docker Compose

在仓库根目录执行：

```bash
cd frontend && npm run build && cd ..
docker compose up --build
```

这条命令保持默认的本地检索模式。若要用 Docker 启动 Elasticsearch 和 Milvus，先确认
`data/processed/chunks.jsonl` 已生成、`.env` 中填好可用的 `LLM_API_KEY` 等模型配置，
再运行：

```bash
cd frontend && npm run build && cd ..
docker compose -f compose.yaml -f compose.external.yaml up --build
```

首次启动会拉取 ES、Milvus、etcd、MinIO 镜像，下载 E5 向量模型，并由一次性
`external-indexer` 容器把内置语料写入 ES/Milvus；随后启动 AI 服务。Docker Desktop
建议分配至少 8 GB 内存。可用 `docker compose -f compose.yaml -f compose.external.yaml ps`
查看健康状态；停止时用相同的 `-f` 参数加 `down`，数据卷会保留，不要随意加 `-v`。
ES 的 9200 和 Milvus 的 19530 仅绑定到本机回环地址；它们在容器内的地址分别是
`http://elasticsearch:9200` 和 `http://milvus:19530`，不需要手动填到 `.env`。

## 在线系统验证与评测

以下检查针对现有在线系统，不包含尚待实现的 QASPER 与 SFT 实验。

```bash
cd ai-service
uv run --extra dev pytest
uv run --extra dev ruff check app tests

cd ../backend
mvn test

cd ../frontend
npm run build
```

### 上传知识库的小规模回归评测

`data/eval/uploaded-knowledge.v1.jsonl` 含宠物指南与《Attention Is All You Need》各 10 题，
其中各有 8 道可回答题、2 道应拒答题。标签使用文档名、章节路径和片段内的短锚点；
运行时必须先在当前活动索引中解析到真实 Chunk，否则报错，不会把失效标签当作检索失败。
数据集针对这两份现有文档；替换文档后需要人工检查并更新标签。

在已启动的 Docker 服务中运行离线检索对比（不调用付费模型）：

```bash
docker compose -f compose.yaml -f compose.external.yaml exec ai-service \
  python -m app.knowledge_evaluation \
  --output /var/lib/support-pilot/uploaded-eval-report.json
```

报告包含 BM25、E5、三路 RRF 与 BGE 精排的 Recall@5、MRR、NDCG@5、P50/P95，
并逐题列出未召回的证据。负例不纳入检索召回指标，因为“检索到相关文字”不等于
“应当回答”。要额外评测最终回答的应答/拒答、引用命中和延迟，可加 `--answers`；
它会调用已配置的模型并可能产生费用。`required_answer_points` 只供人工复核，
脚本不会凭关键词包含关系冒充语义正确性判断。

原始 Redis 语料与 `data/processed/` 受 `.gitignore` 管理。若未准备 `data/processed/chunks.jsonl`，
依赖完整语料的评测集一致性测试会跳过或需要先运行语料构建脚本；其余单元和 API 测试不
依赖网络模型。

## 现有工程文档

- [检索方案对比](docs/retrieval-comparison-v1.md)与 [RRF 权重实验](docs/rrf-weight-experiment-v1.md)：Redis 评测记录，不能直接推广为 QASPER 上的结论。
- [带证据生成](docs/grounded-generation.md)：Prompt、引用白名单与回答评测。
- [语料选择](docs/corpus-selection.md)与 [Corpus Review](docs/corpus-review-v1.md)：内置语料与分块检查记录。
- [MCP 工具契约](docs/tool-contracts.md)与 [受控 Agent](docs/controlled-agent.md)：原技术支持流程。

`docs/scope.md`、`docs/requirements.md` 和 `docs/P0.md` 记录早期 SupportPilot 的范围与
验收要求，其中“暂不做微调”等条目属于历史阶段；DocRAG 后续算法工作的范围以本 README
的研究计划为准。
