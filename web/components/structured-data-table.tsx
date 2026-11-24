"use client"

import { useEffect, useState } from "react"
import { ChevronLeft, ChevronRight, Download, Table2 } from "lucide-react"
import Papa from "papaparse"

export type StructuredDataFormat = "json" | "csv"

export type StructuredDataset = {
  format: StructuredDataFormat
  raw: string
  headers: string[]
  rows: Array<Record<string, unknown>>
}

export function parseStructuredData(
  format: StructuredDataFormat,
  raw: string,
): StructuredDataset | null {
  if (format === "csv") {
    const parsed = Papa.parse<Record<string, string>>(raw, {
      header: true,
      skipEmptyLines: "greedy",
    })

    if (parsed.errors.length > 0 || !parsed.meta.fields?.length) {
      return null
    }

    return {
      format,
      raw,
      headers: parsed.meta.fields,
      rows: parsed.data,
    }
  }

  try {
    const value: unknown = JSON.parse(raw)
    const records = Array.isArray(value) ? value : [value]
    if (
      records.length === 0 ||
      records.some(
        (record) =>
          record === null ||
          typeof record !== "object" ||
          Array.isArray(record),
      )
    ) {
      return null
    }

    const headers = Array.from(
      new Set(records.flatMap((record) => Object.keys(record))),
    )
    if (headers.length === 0) {
      return null
    }

    return {
      format,
      raw,
      headers,
      rows: records as Array<Record<string, unknown>>,
    }
  } catch {
    return null
  }
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "-"
  }
  if (typeof value === "object") {
    return JSON.stringify(value)
  }
  return String(value)
}

export function StructuredDataTable({
  dataset,
}: {
  dataset: StructuredDataset
}) {
  const [page, setPage] = useState(0)
  const pageSize = 5
  const pageCount = Math.max(1, Math.ceil(dataset.rows.length / pageSize))
  const currentPage = Math.min(page, pageCount - 1)
  const visibleRows = dataset.rows.slice(
    currentPage * pageSize,
    (currentPage + 1) * pageSize,
  )

  useEffect(() => {
    setPage(0)
  }, [dataset.raw])

  function downloadSource() {
    const mime = dataset.format === "json" ? "application/json" : "text/csv"
    const url = URL.createObjectURL(new Blob([dataset.raw], { type: mime }))
    const link = document.createElement("a")
    link.href = url
    link.download = `research-data.${dataset.format}`
    document.body.append(link)
    link.click()
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 0)
  }

  return (
    <section className="assistant-data-table" aria-label="结构化数据表格">
      <header className="assistant-data-toolbar">
        <span>
          <Table2 size={15} aria-hidden="true" />
          {dataset.format.toUpperCase()} · {dataset.rows.length} 行
        </span>
        <button
          type="button"
          title={`下载原始 ${dataset.format.toUpperCase()} 数据`}
          onClick={downloadSource}
        >
          <Download size={14} aria-hidden="true" />
          下载
        </button>
      </header>

      <div className="assistant-data-scroll">
        <table>
          <thead>
            <tr>
              {dataset.headers.map((header) => (
                <th scope="col" key={header}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.length > 0 ? (
              visibleRows.map((row, rowIndex) => (
                <tr key={`${currentPage}-${rowIndex}`}>
                  {dataset.headers.map((header) => (
                    <td key={header}>{formatCell(row[header])}</td>
                  ))}
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={dataset.headers.length}>没有数据行</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <footer className="assistant-data-pagination">
        <span>第 {currentPage + 1} / {pageCount} 页</span>
        <div>
          <button
            type="button"
            aria-label="上一页"
            title="上一页"
            disabled={currentPage === 0}
            onClick={() => setPage(currentPage - 1)}
          >
            <ChevronLeft size={15} />
          </button>
          <button
            type="button"
            aria-label="下一页"
            title="下一页"
            disabled={currentPage >= pageCount - 1}
            onClick={() => setPage(currentPage + 1)}
          >
            <ChevronRight size={15} />
          </button>
        </div>
      </footer>
    </section>
  )
}
