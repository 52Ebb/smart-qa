"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

interface Props {
  content: string;
}

/**
 * Markdown 渲染器 — 支持代码高亮、引用块高亮、表格等
 * 引用块（blockquote）使用蓝色左边框高亮，模拟文档引用效果
 */
export function MarkdownRenderer({ content }: Props) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        // 代码块高亮
        code({ className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          const codeStr = String(children).replace(/\n$/, "");

          if (match) {
            return (
              <SyntaxHighlighter
                style={oneDark}
                language={match[1]}
                PreTag="div"
              >
                {codeStr}
              </SyntaxHighlighter>
            );
          }

          return (
            <code className={className} {...props}>
              {children}
            </code>
          );
        },
        // 引用块高亮（模拟文档引用）
        blockquote({ children }) {
          return (
            <blockquote
              style={{
                borderLeft: "4px solid #3b82f6",
                background: "#f0f7ff",
                margin: "12px 0",
                padding: "8px 16px",
                borderRadius: "0 8px 8px 0",
              }}
            >
              {children}
            </blockquote>
          );
        },
        // 表格样式
        table({ children }) {
          return (
            <table
              style={{
                borderCollapse: "collapse",
                width: "100%",
                margin: "12px 0",
              }}
            >
              {children}
            </table>
          );
        },
        th({ children }) {
          return (
            <th
              style={{
                border: "1px solid #d1d5db",
                padding: "8px 12px",
                background: "#f3f4f6",
                textAlign: "left",
              }}
            >
              {children}
            </th>
          );
        },
        td({ children }) {
          return (
            <td
              style={{
                border: "1px solid #d1d5db",
                padding: "8px 12px",
              }}
            >
              {children}
            </td>
          );
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
