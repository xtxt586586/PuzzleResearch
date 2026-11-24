import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import ts from "typescript"

const source = await readFile(
  new URL("../lib/research-tree.ts", import.meta.url),
  "utf8",
)
const compiled = ts.transpile(source, {
  module: ts.ModuleKind.ES2022,
  target: ts.ScriptTarget.ES2022,
})
const helpers = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`,
)

const tree = helpers.normalizeResearchTree(
  {
    topic: "主题",
    epoch: 2,
    nodes: {
      root: {
        topic: "主题",
        status: "completed",
        depth: 0,
        findings_count: 2,
        sources_count: 1,
      },
      branch: {
        topic: "分支",
        status: "pending",
        depth: 1,
      },
    },
  },
  {
    budget_stop_reason: "token_budget_exceeded",
    tokens_used: 128,
    elapsed_seconds: 4.5,
    searches_used: 3,
  },
)

assert.equal(tree.total, 2)
assert.equal(tree.completed, 1)
assert.equal(tree.pending, 1)
assert.equal(tree.nodes[0].id, "root")
assert.equal(tree.budget.stopReason, "token_budget_exceeded")
assert.equal(tree.budget.tokensUsed, 128)

const started = helpers.mergeResearchNodeUpdate(undefined, {
  node_id: "branch-1",
  topic: "新分支",
  query: "补充检索词",
  depth: 1,
  parent_id: "root",
}, "running")
const progressing = helpers.mergeResearchNodeUpdate(started, {
  node_id: "branch-1",
  topic: "新分支",
  depth: 1,
  status: "running",
  result_count: 1,
  sources_count: 1,
  results: [{
    title: "阶段结果",
    url: "https://example.test/a",
    snippet: "第一条搜索结果",
  }],
}, "running")
const completed = helpers.mergeResearchNodeUpdate(progressing, {
  node_id: "branch-1",
  topic: "新分支",
  depth: 1,
  finding_count: 2,
  sources: ["https://example.test/a"],
  findings: [
    {
      title: "阶段结果",
      url: "https://example.test/a",
      snippet: "第一条搜索结果",
    },
    {
      title: "最终结果",
      url: "https://example.test/b",
      snippet: "第二条搜索结果",
    },
  ],
  status: "completed",
})

assert.equal(completed.total, 1)
assert.equal(completed.completed, 1)
assert.equal(completed.nodes[0].resultCount, 2)
assert.equal(completed.nodes[0].sourceCount, 1)
assert.equal(completed.nodes[0].query, "补充检索词")
assert.equal(completed.nodes[0].parentId, "root")
assert.equal(completed.nodes[0].findings.length, 2)
assert.equal(completed.nodes[0].findings[1].title, "最终结果")

const staleCounts = {
  ...completed,
  total: 2,
  completed: 0,
  active: 0,
  pending: 0,
  nodes: [
    { ...completed.nodes[0], status: "completed" },
    { ...completed.nodes[0], id: "branch-2", status: "completed" },
  ],
}
assert.deepEqual(helpers.countResearchTreeNodes(staleCounts), {
  total: 2,
  completed: 2,
  failed: 0,
  active: 0,
  pending: 0,
})

const diagnosed = helpers.normalizeResearchTree(
  { topic: "AI 最新进展", nodes: [] },
  {
    query_coverage_score: 0.8,
    query_dimensions_covered: ["freshness", "official"],
    query_dimensions_missing: ["risk"],
    time_sensitive_query: true,
    freshness_warning: "low_freshness_for_time_sensitive_query",
    freshness_summary: {
      total_results: 4,
      known_count: 3,
      unknown_count: 1,
      fresh_30_count: 1,
      fresh_30_ratio: 1 / 3,
    },
  },
)

assert.equal(diagnosed.quality.queryCoverageScore, 0.8)
assert.deepEqual(diagnosed.quality.dimensionsMissing, ["risk"])
assert.equal(diagnosed.quality.freshness.knownCount, 3)
assert.equal(diagnosed.quality.freshnessWarning, "low_freshness_for_time_sensitive_query")

const restoredDiagnostics = helpers.normalizeResearchTree(
  { topic: "AI", nodes: [] },
  {
    quality_summary: {
      query_coverage_score: 0.6,
      freshness_summary: { total_results: 5, known_count: 2 },
    },
    tree_budget: { tokens_used: 64 },
  },
)
assert.equal(restoredDiagnostics.quality.queryCoverageScore, 0.6)
assert.equal(restoredDiagnostics.quality.freshness.totalResults, 5)
assert.equal(restoredDiagnostics.budget.tokensUsed, 64)

const completedSnapshot = helpers.normalizeResearchSessionSnapshot({
  state: {
    research_tree: {
      topic: "多 Agent 研究",
      nodes: [
        { id: "root", topic: "多 Agent 研究", status: "completed", depth: 0 },
        { id: "worker", topic: "研究分支", status: "completed", depth: 1 },
      ],
    },
    deepsearch_artifacts: {
      research_budget: {
        max_tokens: 10000,
        tokens_used: 4200,
        stop_reason: "",
      },
      quality_summary: { query_coverage_score: 0.8 },
    },
  },
})
assert.equal(completedSnapshot.tree.total, 2)
assert.equal(completedSnapshot.tree.completed, 2)
assert.equal(completedSnapshot.tree.active, 0)
assert.equal(completedSnapshot.budget.tokensUsed, 4200)

const restoredLinearBudget = helpers.normalizeResearchBudget({
  research_budget: {
    max_tokens: 500,
    tokens_used: 128,
    elapsed_seconds: 4.5,
    searches_used: 3,
    stop_reason: "token_budget_exceeded",
  },
})
assert.equal(restoredLinearBudget.maxTokens, 500)
assert.equal(restoredLinearBudget.tokensUsed, 128)
assert.equal(restoredLinearBudget.stopReason, "token_budget_exceeded")

const streamedBudget = helpers.mergeResearchBudget(restoredLinearBudget, {
  deepsearch_tokens_used: 144,
  deepsearch_elapsed_seconds: 5,
  budget_stop_reason: "time_budget_exceeded",
})
assert.equal(streamedBudget.tokensUsed, 144)
assert.equal(streamedBudget.elapsedSeconds, 5)
assert.equal(streamedBudget.stopReason, "time_budget_exceeded")

const updatedQuality = helpers.mergeResearchTreeQuality(diagnosed, {
  query_coverage_score: 1,
  query_dimensions_covered: ["freshness", "official", "risk"],
  query_dimensions_missing: [],
})
assert.equal(updatedQuality.quality.queryCoverageScore, 1)
assert.equal(updatedQuality.quality.freshness.knownCount, 3)

console.log("research-tree.test.mjs passed")
