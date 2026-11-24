import type { Metadata } from "next"
import "katex/dist/katex.min.css"
import "./globals.css"

export const metadata: Metadata = {
  title: "PuzzleResearch",
  description: "PuzzleResearch 深度研究工作台",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  )
}
