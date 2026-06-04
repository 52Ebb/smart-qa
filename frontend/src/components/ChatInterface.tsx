"use client";

import { useState, useRef, useEffect } from "react";
import {
  uploadDocument,
  askQuestionStream,
  getIndexStatus,
  clearIndex,
  type IndexStatus,
} from "@/lib/api";
import { MarkdownRenderer } from "./MarkdownRenderer";

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

/**
 * 聊天界面 — 文件上传 + SSE 流式问答 + Markdown 渲染
 */
export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const conversationIdRef = useRef<string>("");

  // 初始化时查询索引状态
  useEffect(() => {
    refreshIndexStatus();
  }, []);

  // 自动滚到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function refreshIndexStatus() {
    try {
      const status = await getIndexStatus();
      setIndexStatus(status);
    } catch {
      // 后端未启动时静默处理
    }
  }

  async function handleUpload(file: File) {
    setIsUploading(true);
    setUploadMessage("");
    try {
      const result = await uploadDocument(file);
      setUploadMessage(
        `上传成功！文件 "${result.file_name}" 已处理为 ${result.chunk_count} 个文本块`
      );
      await refreshIndexStatus();
    } catch (err: any) {
      setUploadMessage(`上传失败: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  }

  function handleSend() {
    const query = input.trim();
    if (!query || isStreaming) return;
    if (!indexStatus?.indexed) {
      setUploadMessage("请先上传文档再提问");
      return;
    }

    // 添加用户消息
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setInput("");
    setIsStreaming(true);

    // 创建一个占位的 assistant 消息，用于流式更新
    const assistantMsgIndex = messages.length + 1;
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "" },
    ]);

    abortRef.current = askQuestionStream(
      { query, conversation_id: conversationIdRef.current },
      // onToken — 逐字追加
      (token) => {
        setMessages((prev) => {
          const updated = [...prev];
          if (updated[assistantMsgIndex]) {
            updated[assistantMsgIndex] = {
              ...updated[assistantMsgIndex],
              content: updated[assistantMsgIndex].content + token,
            };
          }
          return updated;
        });
      },
      // onDone
      () => {
        setIsStreaming(false);
        abortRef.current = null;
      },
      // onError
      (error) => {
        setMessages((prev) => {
          const updated = [...prev];
          if (updated[assistantMsgIndex]) {
            updated[assistantMsgIndex] = {
              ...updated[assistantMsgIndex],
              content: `错误: ${error}`,
            };
          }
          return updated;
        });
        setIsStreaming(false);
        abortRef.current = null;
      }
    );
  }

  function handleStop() {
    abortRef.current?.abort();
    setIsStreaming(false);
  }

  async function handleClearIndex() {
    try {
      await clearIndex();
      setMessages([]);
      setUploadMessage("索引已清空");
      await refreshIndexStatus();
    } catch (err: any) {
      setUploadMessage(`清空失败: ${err.message}`);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div style={styles.container}>
      {/* 顶部栏 */}
      <header style={styles.header}>
        <h1 style={styles.title}>智能文档问答系统</h1>
        <div style={styles.headerRight}>
          <span style={styles.statusBadge(indexStatus?.indexed)}>
            {indexStatus === null
              ? "检查连接中..."
              : indexStatus.indexed
              ? `已索引 ${indexStatus.document_count} 个文本块`
              : "未索引文档"}
          </span>
          {indexStatus?.indexed && (
            <button onClick={handleClearIndex} style={styles.clearBtn}>
              清空索引
            </button>
          )}
        </div>
      </header>

      {/* 消息区域 */}
      <main style={styles.chatArea}>
        {messages.length === 0 && (
          <div style={styles.emptyState}>
            <p style={styles.emptyIcon}>📄</p>
            <p style={styles.emptyTitle}>上传文档开始问答</p>
            <p style={styles.emptyDesc}>
              支持 PDF 和 Word 格式。上传后即可对文档内容进行提问。
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              ...styles.messageRow,
              justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
            }}
          >
            <div
              style={{
                ...styles.messageBubble,
                background: msg.role === "user" ? "#3b82f6" : "#f3f4f6",
                color: msg.role === "user" ? "#fff" : "#111827",
              }}
            >
              {msg.role === "assistant" ? (
                <MarkdownRenderer content={msg.content || "思考中..."} />
              ) : (
                <p style={styles.messageText}>{msg.content}</p>
              )}
            </div>
          </div>
        ))}

        <div ref={messagesEndRef} />
      </main>

      {/* 底部输入区 */}
      <footer style={styles.inputArea}>
        {/* 上传状态信息 */}
        {uploadMessage && (
          <div style={styles.uploadMsg}>{uploadMessage}</div>
        )}

        {/* 文件上传行 */}
        <div style={styles.uploadRow}>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.doc"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleUpload(file);
              e.target.value = "";
            }}
            style={{ display: "none" }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            style={styles.uploadBtn}
          >
            {isUploading ? "上传中..." : "📎 上传文档"}
          </button>
        </div>

        {/* 文本输入行 */}
        <div style={styles.inputRow}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              indexStatus?.indexed
                ? "输入问题，按 Enter 发送..."
                : "请先上传文档..."
            }
            disabled={!indexStatus?.indexed || isStreaming}
            rows={2}
            style={styles.textarea}
          />
          {isStreaming ? (
            <button onClick={handleStop} style={styles.stopBtn}>
              停止
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              style={{
                ...styles.sendBtn,
                opacity: input.trim() ? 1 : 0.5,
              }}
            >
              发送
            </button>
          )}
        </div>
      </footer>
    </div>
  );
}

// ==================== 内联样式 ====================

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    maxWidth: "900px",
    margin: "0 auto",
    background: "#fff",
    boxShadow: "0 0 20px rgba(0,0,0,0.05)",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "16px 24px",
    borderBottom: "1px solid #e5e7eb",
    background: "#fafafa",
  },
  title: {
    fontSize: "18px",
    fontWeight: 600,
    margin: 0,
  },
  headerRight: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
  },
  statusBadge: (indexed?: boolean) => ({
    fontSize: "12px",
    padding: "4px 12px",
    borderRadius: "12px",
    background: indexed ? "#dcfce7" : "#fef3c7",
    color: indexed ? "#166534" : "#92400e",
  }),
  clearBtn: {
    fontSize: "12px",
    padding: "4px 12px",
    borderRadius: "6px",
    border: "1px solid #fca5a5",
    background: "#fff",
    color: "#dc2626",
    cursor: "pointer",
  },
  chatArea: {
    flex: 1,
    overflowY: "auto",
    padding: "24px",
  },
  emptyState: {
    textAlign: "center" as const,
    paddingTop: "80px",
    color: "#9ca3af",
  },
  emptyIcon: {
    fontSize: "48px",
    margin: "0 0 12px 0",
  },
  emptyTitle: {
    fontSize: "18px",
    fontWeight: 500,
    margin: "0 0 8px 0",
    color: "#6b7280",
  },
  emptyDesc: {
    fontSize: "14px",
    margin: 0,
  },
  messageRow: {
    display: "flex",
    marginBottom: "16px",
  },
  messageBubble: {
    maxWidth: "80%",
    padding: "12px 18px",
    borderRadius: "16px",
    lineHeight: 1.6,
  },
  messageText: {
    margin: 0,
    whiteSpace: "pre-wrap" as const,
  },
  inputArea: {
    borderTop: "1px solid #e5e7eb",
    padding: "16px 24px",
    background: "#fafafa",
  },
  uploadMsg: {
    fontSize: "13px",
    color: "#6b7280",
    marginBottom: "8px",
    padding: "6px 12px",
    background: "#f3f4f6",
    borderRadius: "8px",
  },
  uploadRow: {
    marginBottom: "8px",
  },
  uploadBtn: {
    fontSize: "13px",
    padding: "6px 16px",
    borderRadius: "8px",
    border: "1px dashed #d1d5db",
    background: "#fff",
    cursor: "pointer",
  },
  inputRow: {
    display: "flex",
    gap: "8px",
    alignItems: "flex-end",
  },
  textarea: {
    flex: 1,
    padding: "10px 14px",
    borderRadius: "12px",
    border: "1px solid #d1d5db",
    fontSize: "14px",
    lineHeight: 1.5,
    resize: "none" as const,
    outline: "none",
    fontFamily: "inherit",
  },
  sendBtn: {
    padding: "10px 24px",
    borderRadius: "12px",
    border: "none",
    background: "#3b82f6",
    color: "#fff",
    fontSize: "14px",
    fontWeight: 500,
    cursor: "pointer",
    whiteSpace: "nowrap" as const,
  },
  stopBtn: {
    padding: "10px 24px",
    borderRadius: "12px",
    border: "none",
    background: "#ef4444",
    color: "#fff",
    fontSize: "14px",
    fontWeight: 500,
    cursor: "pointer",
    whiteSpace: "nowrap" as const,
  },
};
