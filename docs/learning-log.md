# PuzzleResearch 学习日志

## 项目基本信息

- 项目名称：PuzzleResearch
- Python 版本：3.11.6
- 开发工具：PyCharm 2025.3.3
- 操作系统：Windows
- 模型服务：DeepSeek
- 参考项目：Weaver
- 参考版本：745693afbf1e8a5b68dad5f616b2681cb2042ee9

---

## 学习记录

### 阶段 0：项目初始化

状态：已完成

掌握内容：

- 创建 Python 项目目录
- 初始化 Git 仓库
- 创建 Python 虚拟环境
- 配置 PyCharm Python 解释器
- 验证 Python 和 pip 路径

关键环境：

```text
项目路径：
D:\pythonproject\PuzzleResearch

虚拟环境：
D:\pythonproject\PuzzleResearch\.venv

Python：
3.11.6

### 阶段 1-4：配置、状态和模型工厂

状态：已完成

掌握内容：

- 使用 Pydantic Settings 管理应用、模型、搜索和研究配置
- 使用 `AgentState` 描述 LangGraph 的共享状态
- 使用消息 reducer 合并并裁剪历史消息
- 使用 `ChatOpenAI` 的兼容接口接入 DeepSeek

### 阶段 5-10：路由、搜索和研究写作

状态：已完成

掌握内容：

- 使用条件边实现 `direct`、`web`、`deep` 和 `clarify` 路由
- 使用 Tavily 和 DuckDuckGo Provider 统一搜索结果结构
- 使用 LangGraph `Send` 并行分发多个搜索查询
- 使用 Planner、Writer 和引用列表生成 Web/Deep Research 报告
- 为 DeepSeek 不支持的结构化输出增加普通 JSON 降级

### 阶段 11-13：评估、修订和多轮深度研究

状态：已完成

掌握内容：

- 使用 Evaluator 检查准确性、完整性和引用情况
- 使用 Reviser 在预算内修订报告
- 根据知识缺口生成后续查询并执行多轮 Deep Research
- 使用最大修订次数和最大研究轮次阻止无限循环
- 在当时阶段明确记录 Agent 工具执行能力尚未接入

### 阶段 13 后代码审查修复

状态：已完成

修复内容：

- 修正 Deep Search 结果数配置未生效的问题
- 修正筛选后的 Deep Research 上下文没有实际传给 Writer 的问题
- 修正 `pass` 与非空缺失主题同时出现的质量契约冲突
- 修正澄清和未接入工具任务被错误标记为 completed 的问题
- 将用户输入写入消息历史，并对并行搜索结果按 URL 去重
- 增加依赖清单、标准库回归测试和与实现同步的功能矩阵


### 阶段 14：Agent 工具调用与审批

状态：已完成

掌握内容：

- 使用 `ToolRegistry` 注册、查询和调用工具
- 使用独立子进程执行 Python 代码
- 使用超时和输出长度限制控制工具执行
- 使用工具白名单限制可调用工具
- 使用工具调用次数限制避免无限执行
- 将 Agent 工具流程拆分为 `agent_plan_node` 和 `agent_tool_node`
- 使用 LangGraph `interrupt` 暂停工具审批
- 使用 Checkpointer 和 `Command(resume=...)` 恢复审批后的执行
- 为工具规划、工具执行和图级审批流程增加回归测试

安全边界：

当前 Python 执行使用 Python `-I` 隔离模式和子进程控制，但这不等同于严格安全沙箱。后续如需执行不可信代码，应考虑 Docker、E2B 或其他隔离运行环境。