```markdown
# PuzzleResearch 功能对照表

状态说明：

- [ ] 未开始
- [~] 开发中
- [x] 已完成
- [!] 需要进一步验证

---

## 一、基础工程

- [x] Python 3.11 虚拟环境
- [x] PyCharm 解释器配置
- [x] Git 仓库
- [x] 项目 README
- [x] 环境变量管理
- [ ] 日志系统
- [x] 配置系统
- [ ] Docker 基础环境

## 二、核心 Agent 架构

- [x] AgentState
- [x] 消息 reducer
- [x] LangGraph StateGraph
- [x] 路由节点
- [x] 条件边
- [x] 普通回答节点
- [x] Agent 节点（工具规划、白名单、调用限制、Python 执行和审批恢复）
- [x] 澄清节点
- [x] Planner 节点
- [x] Writer 节点
- [x] Evaluator 节点
- [x] Reviser 节点
- [ ] Coordinator 节点

## 三、智能路由

- [x] direct 路由
- [x] agent 路由（工具规划、执行和审批中断）
- [x] web 路由
- [x] deep 路由
- [x] clarify 路由
- [x] 路由置信度
- [ ] 用户显式模式覆盖
- [x] 路由异常降级

## 四、搜索系统

- [x] Tavily Provider
- [x] DuckDuckGo Provider
- [ ] Brave Provider
- [ ] Serper Provider
- [x] 多 Provider 搜索
- [x] Provider fallback
- [ ] 搜索缓存
- [x] URL 去重
- [x] 搜索结果标准化
- [ ] Provider 统计

## 五、Deep Research

- [x] 查询生成
- [x] 研究计划
- [x] 多轮线性研究
- [ ] 树状研究
- [x] 分支并行
- [ ] 搜索结果摘要
- [ ] 网页正文抓取
- [ ] 证据段落提取
- [ ] 知识缺口检测
- [x] 后续查询生成
- [x] 来源整理
- [x] 报告生成
- [x] 引用附加
- [x] 研究预算
- [ ] 研究取消

## 六、Agent 工具系统

- [x] Tool Registry
- [ ] 动态工具装配
- [ ] Agent Profile
- [x] 工具白名单
- [ ] 工具黑名单
- [ ] 工具去重
- [ ] 工具事件包装
- [ ] 工具选择器
- [ ] 工具重试
- [x] 工具调用限制
- [ ] 上下文清理
- [ ] Todo List
- [x] 工具人工审批

## 七、代码和浏览器

- [x] Python 代码执行
- [ ] Docker 沙箱
- [ ] E2B 沙箱适配
- [ ] Playwright 浏览器
- [ ] 浏览器导航
- [ ] 浏览器点击
- [ ] 浏览器输入
- [ ] 页面内容提取
- [ ] 浏览器截图
- [ ] WebSocket 浏览器画面
- [ ] 桌面自动化

## 八、RAG 和记忆

- [ ] 文档上传
- [ ] PDF 解析
- [ ] DOCX 解析
- [ ] 文本切分
- [ ] 本地 Embedding
- [ ] ChromaDB
- [ ] 向量检索
- [ ] LangGraph Store
- [ ] 用户长期记忆
- [ ] Mem0 适配
- [ ] 本地记忆降级

## 九、质量控制

- [x] Evaluator
- [ ] coverage 评估
- [ ] accuracy 评估
- [ ] freshness 评估
- [ ] coherence 评估
- [ ] QualityAssessor
- [ ] ClaimVerifier
- [ ] citation gate
- [x] 缺失主题检测
- [x] 报告修订
- [x] 最大修订次数

## 十、会话和人工介入

- [x] InMemorySaver（开发/测试）
- [ ] PostgreSQL Checkpointer
- [ ] Redis Store
- [ ] 计划审核
- [ ] 来源审核
- [ ] 草稿审核
- [ ] 最终报告审核
- [x] 工具审批中断恢复
- [ ] 任务取消
- [ ] 会话列表
- [ ] 会话权限隔离

## 十一、前端

- [ ] Next.js 项目
- [ ] 聊天页面
- [ ] SSE Hook
- [ ] 状态展示
- [ ] 搜索活动面板
- [ ] 工具活动面板
- [ ] 代码输出展示
- [ ] 引用展示
- [ ] 报告 Markdown 渲染
- [ ] Mermaid 图表
- [ ] 会话历史
- [ ] 人工审核界面
- [ ] 文件上传
- [ ] 文件下载

## 十二、扩展功能

- [ ] MCP
- [ ] ASR
- [ ] TTS
- [ ] 报告导出
- [ ] 定时任务
- [ ] Webhook
- [ ] API 鉴权
- [ ] 请求限流
- [ ] Prometheus 指标
- [ ] OpenAPI
- [ ] Docker Compose
- [ ] Benchmark
- [ ] 完整测试
