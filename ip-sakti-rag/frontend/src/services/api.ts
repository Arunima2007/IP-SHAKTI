import type { ChatRequest, ChatResponse, DocumentListResponse, HealthResponse } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export async function sendChatQuery(payload: ChatRequest): Promise<ChatResponse> {
  const url = `${API_BASE_URL}/api/chat`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorDetail = 'Failed to process legal query';
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || errJson.error || errorDetail;
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

export async function checkHealth(): Promise<HealthResponse> {
  const url = `${API_BASE_URL}/health`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }
  return response.json();
}

export async function fetchDocuments(): Promise<DocumentListResponse> {
  const url = `${API_BASE_URL}/api/documents`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load document catalog: ${response.status}`);
  }
  return response.json();
}
