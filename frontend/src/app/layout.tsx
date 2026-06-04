import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "智能文档问答系统",
  description: "基于 LangGraph 的 RAG 文档问答系统",
};

/**
 * 根布局 — 全局 HTML 结构
 */
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body style={{ margin: 0, padding: 0 }}>{children}</body>
    </html>
  );
}
