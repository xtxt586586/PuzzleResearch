# Agent 核心追踪与评测

## 零模型费用的核心回归

在项目根目录执行：

```powershell
.venv\Scripts\python.exe scripts/evaluate_agent_core.py --output reports/agent-core-mock.json
```

脚本运行现有的补研派单、分支隔离、工具审批恢复、并发恢复和 SSE 流式输出测试，生成逐项通过状态与测试耗时。它使用模拟模型和工具；测试耗时不是用户请求耗时，结果不代表真实研究质量或线上成功率。报告记录 Git HEAD；工作区可能有未提交改动。

## 固定题集与真实运行

`eval/benchmarks/agent_core_v1.jsonl` 包含 10 个固定主题和预期维度。默认只生成计划，不调用模型：

```powershell
.venv\Scripts\python.exe scripts/benchmark_deep_research.py --bench-file eval/benchmarks/agent_core_v1.jsonl --max-cases 10 --output reports/agent-research-plan.json
```

确认模型和搜索服务配置后，可从 1 个题目开始进行真实评测：

```powershell
.venv\Scripts\python.exe scripts/benchmark_deep_research.py --bench-file eval/benchmarks/agent_core_v1.jsonl --max-cases 1 --mode tree --hierarchical-agents --execute --base-url asgi --output reports/agent-research-live.json
```

`--execute` 会调用真实模型及可能的搜索服务，产生费用；未启用 `--execute` 时数值是 `null`，不是零。真实执行报告记录完成状态、耗时、首字时间、模型调用数、服务端证据指标和可用的追踪摘要。质量字段中的维度匹配属于简单文本匹配，不等同于人工核验；引用覆盖也不等于引用内容真实支持结论。报告包含题目和报告预览，请妥善保管。

## 单次运行追踪

配置 `ENABLE_TRACING=true` 并重启后端，通过会话 thread ID 查询：

```text
GET /api/runs/{thread_id}
GET /api/traces/{thread_id}/summary
GET /api/traces/{thread_id}
```

接口需遵循项目已有的身份鉴权。运行指标与追踪默认在进程内存，重启后不保留；是否有节点、工具、Token 数据取决于相应事件是否实际被采集。追踪摘要的 LLM span 数按事件生成，可能包含同一次调用的多个事件，不能直接当作模型调用次数；优先参考运行指标中的 `model_call_count`，并检查其可用性。不要把缺失指标当作零。当前机制不是分布式持久追踪，也不自动计算费用。
