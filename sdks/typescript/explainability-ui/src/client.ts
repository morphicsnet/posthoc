import { HIFGraph, TraceStatus, isHIFGraph } from './types';

export class HTTPError extends Error {
  public readonly status: number;
  public readonly url: string;
  constructor(status: number, url: string, message?: string) {
    super(message ?? `HTTP ${status} for ${url}`);
    this.status = status;
    this.url = url;
  }
}

export interface ExplainabilityClientOptions {
  baseUrl: string;
  apiKey?: string;
  headers?: Record<string, string>;
}

export class ExplainabilityClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly extraHeaders: Record<string, string>;

  constructor(opts: ExplainabilityClientOptions) {
    if (!opts?.baseUrl) {
      throw new Error('ExplainabilityClient requires baseUrl');
    }
    this.baseUrl = opts.baseUrl.replace(/\/+$/, '');
    this.apiKey = opts.apiKey;
    this.extraHeaders = opts.headers ?? {};
  }

  private buildHeaders(): HeadersInit {
    const headers: Record<string, string> = {
      Accept: 'application/json',
      ...this.extraHeaders,
    };
    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  async getGraph(traceId: string, signal?: AbortSignal): Promise<HIFGraph> {
    const url = `${this.baseUrl}/v1/traces/${encodeURIComponent(traceId)}/graph`;
    const res = await fetch(url, {
      method: 'GET',
      headers: this.buildHeaders(),
      signal,
    });

    if (!res.ok) {
      // 404 - not ready; 410 - expired
      const msg = await safeText(res);
      throw new HTTPError(res.status, url, msg || undefined);
    }

    const data = await res.json();
    if (!isHIFGraph(data)) {
      // Defensive: some deployments may wrap in ExplanationResponse by mistake
      // Try unwrapping data.hypergraph if present
      const maybe = (data as any)?.hypergraph;
      if (isHIFGraph(maybe)) {
        return maybe;
      }
      throw new Error('Response did not match HIFGraph schema');
    }
    return data;
  }

  async getStatus(traceId: string, signal?: AbortSignal): Promise<TraceStatus> {
    const url = `${this.baseUrl}/v1/traces/${encodeURIComponent(traceId)}/status`;
    const res = await fetch(url, {
      method: 'GET',
      headers: this.buildHeaders(),
      signal,
    });

    if (!res.ok) {
      const msg = await safeText(res);
      throw new HTTPError(res.status, url, msg || undefined);
    }
    // Minimal shape validation inline to avoid another type guard
    const status = (await res.json()) as TraceStatus;
    if (
      !status ||
      typeof status.trace_id !== 'string' ||
      typeof status.state !== 'string'
    ) {
      throw new Error('Response did not match TraceStatus schema');
    }
    return status;
  }
}

async function safeText(res: Response): Promise<string | null> {
  try {
    return await res.text();
  } catch {
    return null;
  }
}

export type { HIFGraph, TraceStatus } from './types.ts';