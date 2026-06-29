/**
 * 后端 API 调用封装
 */

const API_BASE = "/api";

export interface ChatRequest {
  query: string;
  conversation_id?: string;
}

export interface ChatResponse {
  answer: string;
  sources: string[];
  conversation_id: string;
}

export interface UploadResponse {
  message: string;
  file_id: string;
  chunk_count: number;
  file_name: string;
}

export interface IndexStatus {
  indexed: boolean;
  document_count: number;
  bm25_indexed: boolean;
}

/** 统一解析错误响应，提取后端返回的 detail 字段 */
async function parseError(res: Response, fallback: string): Promise<string> {
  try {
    const error = await res.json();
    return error.detail || error.message || fallback;
  } catch {
    return fallback;
  }
}

/**
 * 上传文档到后端
 */
export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(await parseError(res, "上传失败"));
  }

  return res.json();
}

/**
 * 发送问答请求（非流式）
 */
export async function askQuestion(request: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    throw new Error(await parseError(res, "问答请求失败"));
  }

  return res.json();
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onDone: () => void;
  onError: (error: string) => void;
  onConversationId?: (id: string) => void;
  onStatus?: (status: string) => void;
}

/**
 * 发送问答请求（SSE 流式输出）
 * 返回一个可读流，逐 token 推送。
 * 正确处理 conversation_id / status / token / error 事件，
 * 并在结束/出错/中止时释放 reader 以避免连接泄漏。
 */
export function askQuestionStream(
  request: ChatRequest,
  callbacks: StreamCallbacks,
): AbortController {
  const controller = new AbortController();
  const { onToken, onDone, onError, onConversationId, onStatus } = callbacks;

  fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal: controller.signal,
  }).then(async (res) => {
    if (!res.ok) {
      onError(await parseError(res, `HTTP ${res.status}: ${res.statusText}`));
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      onError("无法读取响应流");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let sawDone = false;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") {
            sawDone = true;
            onDone();
            return;
          }
          try {
            const parsed = JSON.parse(data);
            switch (parsed.type) {
              case "token":
                onToken(parsed.content);
                break;
              case "error":
                onError(parsed.content);
                break;
              case "conversation_id":
                onConversationId?.(parsed.content);
                break;
              case "status":
                onStatus?.(parsed.content);
                break;
            }
          } catch {
            // 忽略无法解析的事件行
          }
        }
      }
      // 流自然结束但未收到 [DONE]（可能连接被截断）
      if (!sawDone) onDone();
    } finally {
      // 释放 reader，避免连接泄漏
      try {
        reader.cancel();
      } catch {
        // 忽略
      }
      // 刷新 decoder 残留字节
      decoder.decode();
    }
  }).catch((err) => {
    if (err.name !== "AbortError") {
      onError(err instanceof Error ? err.message : String(err));
    }
  });

  return controller;
}

/**
 * 查询索引状态
 */
export async function getIndexStatus(): Promise<IndexStatus> {
  const res = await fetch(`${API_BASE}/index/status`);
  if (!res.ok) {
    throw new Error(await parseError(res, "获取索引状态失败"));
  }
  return res.json();
}

/**
 * 清空索引
 */
export async function clearIndex(): Promise<void> {
  const res = await fetch(`${API_BASE}/index`, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(await parseError(res, "清空失败"));
  }
}
