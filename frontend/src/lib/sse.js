/**
 * Server-Sent Events client for `POST /chat/stream`.
 *
 * `EventSource` cannot send an `Authorization` header (and cannot POST), so the
 * stream is consumed manually from `fetch` + `ReadableStream`, parsing the SSE
 * framing ourselves.
 */

import { API_BASE_URL, API_PREFIX, tokenStore } from "./api";

/**
 * Parse an SSE byte stream into `{ event, data }` objects.
 * @param {ReadableStream<Uint8Array>} stream
 */
export async function* parseEventStream(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Frames are separated by a blank line.
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const parsed = parseFrame(frame);
        if (parsed) yield parsed;
        boundary = buffer.indexOf("\n\n");
      }
    }
    const tail = parseFrame(buffer);
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(frame) {
  const lines = frame.split("\n");
  let event = "message";
  const dataLines = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
    // `:` prefixed lines are comments/heartbeats - intentionally ignored.
  }

  if (dataLines.length === 0) return null;
  const raw = dataLines.join("\n");
  try {
    return { event, data: JSON.parse(raw) };
  } catch {
    return { event, data: { raw } };
  }
}

/**
 * Stream a RAG answer.
 *
 * @param {object} payload - `{ question, conversation_id?, document_ids?, top_k? }`
 * @param {object} handlers - `onStart`, `onSources`, `onToken`, `onDone`, `onError`
 * @param {AbortSignal} [signal]
 */
export async function streamChat(payload, handlers = {}, signal) {
  const { onStart, onSources, onToken, onDone, onError } = handlers;

  const response = await fetch(`${API_BASE_URL}${API_PREFIX}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...(tokenStore.getAccessToken()
        ? { Authorization: `Bearer ${tokenStore.getAccessToken()}` }
        : {}),
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    let message = `Stream failed with status ${response.status}`;
    try {
      const body = await response.json();
      message = body.message || body.detail || message;
    } catch {
      /* non-JSON error body */
    }
    onError?.({ message, code: `http_${response.status}` });
    return;
  }

  if (!response.body) {
    onError?.({ message: "The server returned an empty stream.", code: "empty_stream" });
    return;
  }

  for await (const { event, data } of parseEventStream(response.body)) {
    switch (event) {
      case "start":
        onStart?.(data);
        break;
      case "sources":
        onSources?.(data.sources || []);
        break;
      case "token":
        onToken?.(data.text || "", data.index ?? 0);
        break;
      case "done":
        onDone?.(data);
        break;
      case "error":
        onError?.(data);
        break;
      case "heartbeat":
        break;
      default:
        break;
    }
  }
}
