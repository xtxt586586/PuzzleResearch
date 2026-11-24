"use client"

import { useEffect, useId, useState } from "react"
import { LoaderCircle } from "lucide-react"

type MermaidRenderer = typeof import("mermaid").default

let mermaidRenderer: Promise<MermaidRenderer> | null = null

function loadMermaid(): Promise<MermaidRenderer> {
  if (!mermaidRenderer) {
    mermaidRenderer = import("mermaid").then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: "neutral",
        fontFamily: "inherit",
      })
      return mermaid
    })
  }

  return mermaidRenderer
}

export function MermaidDiagram({ code }: { code: string }) {
  const reactId = useId().replace(/[^a-zA-Z0-9_-]/g, "")
  const [svg, setSvg] = useState("")
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading")

  useEffect(() => {
    let active = true
    setStatus("loading")

    const timeout = window.setTimeout(() => {
      void loadMermaid()
        .then((mermaid) =>
          mermaid.render(`diagram-${reactId}`, code),
        )
        .then(({ svg: renderedSvg }) => {
          if (active) {
            setSvg(renderedSvg)
            setStatus("ready")
          }
        })
        .catch(() => {
          if (active) {
            setSvg("")
            setStatus("error")
          }
        })
    }, 300)

    return () => {
      active = false
      window.clearTimeout(timeout)
    }
  }, [code, reactId])

  if (status === "error") {
    return (
      <div className="assistant-mermaid-error" role="status">
        <strong>图表无法渲染，原始代码如下</strong>
        <pre>{code}</pre>
      </div>
    )
  }

  return (
    <figure className="assistant-mermaid-figure">
      {status === "loading" ? (
        <div className="assistant-mermaid-loading" role="status">
          <LoaderCircle size={16} className="spin" />
          <span>正在绘制图表…</span>
        </div>
      ) : (
        <div
          className="assistant-mermaid-svg"
          role="img"
          aria-label="Mermaid 图表"
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      )}
    </figure>
  )
}
