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
    const error = await res.json();
    throw new Error(error.detail || "上传失败");
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
    const error = await res.json();
    throw new Error(error.detail || "问答请求失败");
  }

  return res.json();
}

/**
 * 发送问答请求（SSE 流式输出）
 * 返回一个可读流，逐 token 推送
 */
export function askQuestionStream(
  request: ChatRequest,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (error: string) => void,
): AbortController {
  const controller = new AbortController();

  fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal: controller.signal,
  }).then(async (res) => {
    if (!res.ok) {
      onError(`HTTP ${res.status}: ${res.statusText}`);
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      onError("无法读取响应流");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6).trim();
          if (data === "[DONE]") {
            onDone();
            return;
          }
          try {
            const parsed = JSON.parse(data);
            if (parsed.type === "token") {
              onToken(parsed.content);
            } else if (parsed.type === "error") {
              onError(parsed.content);
            }
          } catch {
            // 忽略解析错误
          }
        }
      }
    }
    onDone();
  }).catch((err) => {
    if (err.name !== "AbortError") {
      onError(err.message);
    }
  });

  return controller;
}

/**
 * 查询索引状态
 */
export async function getIndexStatus(): Promise<IndexStatus> {
  const res = await fetch(`${API_BASE}/index/status`);
  return res.json();
}

/**
 * 清空索引
 */
export async function clearIndex(): Promise<void> {
  const res = await fetch(`${API_BASE}/index`, { method: "DELETE" });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || "清空失败");
  }
}
