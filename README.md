# PuzzleResearch

PuzzleResearch 是一个基于 LangGraph 的全功能 AI Agent 研究平台。

本项目参考 Weaver 项目进行功能和架构复刻，目标包括：

- 智能任务路由
- 普通问答
- Web 搜索
- Deep Research
- 线性研究
- 树状研究
- 多角色 Agent 协作
- 工具调用
- Python 代码执行
- 浏览器自动化
- RAG 文档问答
- MCP 工具接入
- SSE 流式响应
- WebSocket 浏览器画面
- Checkpoint 会话恢复
- Human-in-the-Loop 人工审核
- 质量评估和报告修订
- 文档、表格和演示文稿生成
- 语音输入和输出
- 定时任务和 Webhook
- Docker 部署

## 当前技术栈

- Python 3.11
- FastAPI
- LangGraph
- LangChain
- DeepSeek API
- Next.js
- PostgreSQL
- Redis
- ChromaDB
- Playwright
- Docker

## 项目目标

本项目不是简单复制代码，而是在理解原项目设计的基础上，逐步实现功能等价的完整系统。

核心代码会按照学习计划逐步手动实现，外围重复代码和配置文件可以直接使用。

## 原项目参考

- 项目：Weaver
- 地址：https://github.com/skygazer42/Weaver
- 参考版本：745693afbf1e8a5b68dad5f616b2681cb2042ee9
- 许可证：MIT

## 开发状态

当前阶段：阶段 13 已完成，已具备路由、Web Research、Deep Research、评估和迭代修订主流程。

当前可以稳定运行的路径：

- `direct`：直接问答
- `web`：规划查询、并行搜索、写作和评估
- `deep`：多查询、多轮补充研究、综合写作和评估修订
- `clarify`：生成澄清问题并暂停本次执行

`agent` 路径已经可以被识别，但工具执行节点尚未接入，会明确返回暂停状态，不会伪装成已执行工具。

安装依赖：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

API Key 只放在项目根目录的 `.env` 中；该文件已被 Git 忽略，不应提交。

详细进度见：

- `docs/feature-matrix.md`
- `docs/learning-log.md`
