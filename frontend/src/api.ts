export type ServiceState = "ok" | "error";
export type ReadinessState = "ok" | "degraded";

export interface ComponentHealth {
  status: ServiceState;
  latency_ms: number;
  detail: string;
}

export interface ReadinessResponse {
  status: ReadinessState;
  services: Record<string, ComponentHealth>;
}

export interface PaperSummary {
  id: string;
  title: string;
  status: "pending" | "processing" | "ready" | "failed";
}

export interface PaperDetail extends PaperSummary {
  error_message: string | null;
}

export interface RetrievalResponse {
  question: string;
  paper_ids: string[];
  candidate_count: number;
  sources: {
    source_id: number;
    paper_id: string;
    paper_title: string;
    page_number: number;
    chunk_id: string;
    excerpt: string;
    score: number;
  }[];
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { Accept: "application/json", ...init?.headers },
  });

  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    if (
      body &&
      typeof body === "object" &&
      "detail" in body &&
      typeof body.detail === "string"
    ) {
      throw new Error(body.detail);
    }
    throw new Error(`Request failed with status ${String(response.status)}`);
  }

  return (await response.json()) as T;
}

export function getReadiness(): Promise<ReadinessResponse> {
  return requestJson<ReadinessResponse>("/api/v1/health/ready");
}

export function getPapers(): Promise<PaperSummary[]> {
  return requestJson<PaperSummary[]>("/api/v1/papers");
}

export function uploadPaper(file: File): Promise<PaperDetail> {
  const body = new FormData();
  body.append("file", file);
  return requestJson<PaperDetail>("/api/v1/papers", { method: "POST", body });
}

export function getPaper(id: string): Promise<PaperDetail> {
  return requestJson<PaperDetail>(`/api/v1/papers/${encodeURIComponent(id)}`);
}

export function searchPapers(
  question: string,
  paperIds: string[],
): Promise<RetrievalResponse> {
  return requestJson<RetrievalResponse>("/api/v1/debug/retrieval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, paper_ids: paperIds, limit: 8 }),
  });
}
