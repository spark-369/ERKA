"use client";

/**
 * Chat console: SSE-streamed RAG answers with grounded, cited sources.
 *
 * Layout: a conversation rail on the left, the transcript in the centre. Answers
 * arrive as progressive `token` events, followed by a `done` event carrying the
 * retrieved sources (rendered as citation chips) and answer metadata.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ChatIcon,
  CopyIcon,
  DocumentsIcon,
  EditIcon,
  PlusIcon,
  SendIcon,
  SparkIcon,
  StopIcon,
  TrashIcon,
} from "@/components/icons";
import {
  Alert,
  Badge,
  Button,
  Card,
  Checkbox,
  EmptyState,
  Spinner,
} from "@/components/ui";
import { api } from "@/lib/api";
import { streamChat } from "@/lib/sse";
import { useToast } from "@/lib/toast";

const SUGGESTIONS = [
  "Summarise the key policies in my knowledge base.",
  "What are the main steps in the documented process?",
  "Which document mentions the retention period?",
];

/**
 * Normalise a stored assistant message. History persists citations and sources
 * as one JSON envelope; live messages already carry those as plain fields.
 */
function parseStored(raw) {
  const empty = { sources: [], citations: [] };
  if (!raw) return empty;
  let parsed = raw;
  if (typeof raw === "string") {
    try {
      parsed = JSON.parse(raw);
    } catch {
      return empty;
    }
  }
  // Backwards compatible: older rows stored a bare array of sources.
  const sources = Array.isArray(parsed) ? parsed : parsed?.sources || [];
  const citations = sources.reduce((acc, source, index) => {
    if (source?.cited) acc.push(index);
    return acc;
  }, []);
  return { sources, citations };
}

// --------------------------------------------------------------------------- //
// Answer pieces
// --------------------------------------------------------------------------- //
function SourceList({ sources, citations, activeSource, onFocusSource }) {
  if (!sources?.length) return null;
  const cited = new Set(citations || []);
  const anyCited = cited.size > 0;

  return (
    <div className="mt-4 flex flex-col gap-2">
      <p className="flex items-center gap-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">
        Sources
        {anyCited ? (
          <span className="font-normal normal-case text-slate-400">{cited.size} cited</span>
        ) : null}
      </p>
      {sources.map((source, index) => {
        const isCited = cited.has(index);
        const isActive = activeSource === index;
        return (
          <button
            type="button"
            key={`${source.chunk_id}-${index}`}
            onClick={() => onFocusSource?.(isActive ? null : index)}
            className={`rounded-lg border px-3 py-2 text-left text-xs transition-colors ${
              isActive
                ? "border-slate-400 bg-slate-50 dark:border-slate-500 dark:bg-slate-800"
                : isCited
                  ? "border-emerald-200 bg-emerald-50 dark:border-emerald-900 dark:bg-emerald-950"
                  : "border-slate-200 bg-slate-50 hover:border-slate-300 dark:border-slate-800 dark:bg-slate-900 dark:hover:border-slate-700"
            }`}
          >
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="flex min-w-0 items-center gap-1.5">
                <span className="flex h-4 min-w-4 shrink-0 items-center justify-center rounded bg-slate-200 px-1 text-[10px] font-semibold text-slate-600 dark:bg-slate-700 dark:text-slate-300">
                  {index + 1}
                </span>
                <span className="truncate font-medium">{source.document_name}</span>
              </span>
              <span className="flex shrink-0 items-center gap-1">
                {isCited ? <Badge tone="green">cited</Badge> : null}
                <Badge tone="blue">{Math.round((source.score || 0) * 100)}%</Badge>
              </span>
            </div>
            <p className="leading-5 text-slate-600 dark:text-slate-400">
              {isActive ? source.snippet : `${source.snippet.slice(0, 160)}…`}
            </p>
          </button>
        );
      })}
    </div>
  );
}

function MessageBubble({ message, onFocusSource, activeSource }) {
  const isUser = message.role === "user";
  const sources = Array.isArray(message.sources) ? message.sources : [];
  const toast = useToast();
  const [copied, setCopied] = useState(false);
  // An assistant turn that finished with zero retrieved chunks was not grounded
  // in any document. Rendering it identically to a cited answer hides that from
  // the user, who reasonably assumes the answer came from their corpus.
  const ungrounded = !isUser && !message.streaming && Boolean(message.content) && sources.length === 0;

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(message.content || "");
      setCopied(true);
      toast.success("Answer copied to clipboard.");
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Clipboard is unavailable in this browser context.");
    }
  }

  if (isUser) {
    return (
      <div className="animate-fade-in flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-brand-600 px-4 py-3 text-sm leading-6 whitespace-pre-wrap text-white">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in flex gap-3">
      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-brand-600 text-white">
        <SparkIcon className="h-4 w-4" />
      </span>
      <div className="min-w-0 flex-1 rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3 text-sm dark:border-slate-800 dark:bg-slate-900">
        {message.streaming && !message.content ? (
          <p className="flex items-center gap-2 text-slate-400">
            <Spinner className="h-3.5 w-3.5" /> Thinking…
          </p>
        ) : (
          <p className="leading-7 whitespace-pre-wrap">
            {message.content}
            {message.streaming ? <span className="ml-1 animate-pulse">▍</span> : null}
          </p>
        )}

        <SourceList
          sources={sources}
          citations={message.citations}
          activeSource={activeSource}
          onFocusSource={onFocusSource}
        />

        {!message.streaming && message.content ? (
          <div className="mt-3 flex items-center justify-between gap-3 border-t border-slate-100 pt-2 dark:border-slate-800">
            <span className="flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
              {ungrounded ? <Badge tone="amber">no sources</Badge> : null}
              {message.latency_ms ? <span>{message.latency_ms} ms</span> : null}
              {message.token_count ? <span>{message.token_count} tokens</span> : null}
            </span>
            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1 text-xs text-slate-400 transition-colors hover:text-slate-600 dark:hover:text-slate-200"
            >
              <CopyIcon className="h-3.5 w-3.5" />
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}


function ConversationRail({
  conversations,
  activeId,
  onOpen,
  onDelete,
  onRename,
  onNew,
  documents,
  selectedDocumentIds,
  onToggleDocument,
  onClearScope,
  disabled,
}) {
  const [query, setQuery] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [draftTitle, setDraftTitle] = useState("");
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return conversations;
    return conversations.filter((item) => item.title.toLowerCase().includes(needle));
  }, [conversations, query]);

  function beginRename(conversation) {
    setEditingId(conversation.id);
    setDraftTitle(conversation.title);
  }

  function submitRename(id) {
    const next = draftTitle.trim();
    setEditingId(null);
    const current = conversations.find((item) => item.id === id);
    if (!next || next === current?.title) return;
    onRename?.(id, next);
  }

  return (
    <aside className="flex h-full flex-col gap-3">
      <Button onClick={onNew} className="w-full">
        <PlusIcon className="h-4 w-4" />
        New conversation
      </Button>

      <div className="flex flex-col gap-1.5">
        {conversations.length > 6 ? (
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Filter conversations…"
            className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs outline-none focus:border-brand-500 dark:border-slate-800 dark:bg-slate-950"
          />
        ) : null}
        <div className="flex max-h-[40vh] flex-col gap-1 overflow-y-auto">
          {filtered.length === 0 ? (
            <p className="px-1 py-2 text-xs text-slate-500">
              {conversations.length === 0 ? "No conversations yet." : "No matches."}
            </p>
          ) : (
            filtered.map((conversation) => {
              const active = conversation.id === activeId;
              return (
                <div
                  key={conversation.id}
                  className={`group flex items-center gap-1 rounded-lg px-2 py-1.5 text-sm ${
                    active
                      ? "bg-brand-600 text-white"
                      : "hover:bg-slate-100 dark:hover:bg-slate-800"
                  }`}
                >
                  {editingId === conversation.id ? (
                    <form
                      className="flex min-w-0 flex-1 items-center gap-1"
                      onSubmit={(event) => {
                        event.preventDefault();
                        submitRename(conversation.id);
                      }}
                    >
                      <input
                        autoFocus
                        value={draftTitle}
                        onChange={(event) => setDraftTitle(event.target.value)}
                        onBlur={() => submitRename(conversation.id)}
                        onKeyDown={(event) => {
                          if (event.key === "Escape") setEditingId(null);
                        }}
                        maxLength={255}
                        className="min-w-0 flex-1 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs text-slate-900 outline-none focus:border-brand-500 dark:border-slate-600 dark:bg-slate-950 dark:text-slate-100"
                      />
                    </form>
                  ) : (
                    <>
                      <button
                        type="button"
                        onClick={() => onOpen(conversation.id)}
                        className="flex min-w-0 flex-1 items-center gap-2 text-left"
                        title={conversation.title}
                      >
                        <ChatIcon className="h-3.5 w-3.5 shrink-0 opacity-60" />
                        <span className="truncate">{conversation.title}</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => beginRename(conversation)}
                        className="shrink-0 opacity-0 transition-opacity group-hover:opacity-70"
                        aria-label="Rename conversation"
                      >
                        <EditIcon className="h-3.5 w-3.5" />
                      </button>
                      <button
                        type="button"
                        onClick={() => onDelete(conversation.id)}
                        className="shrink-0 opacity-0 transition-opacity group-hover:opacity-70"
                        aria-label="Delete conversation"
                      >
                        <TrashIcon className="h-3.5 w-3.5" />
                      </button>
                    </>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>

      <div className="flex flex-col gap-2 border-t border-slate-200 pt-3 dark:border-slate-800">
        <div className="flex items-center justify-between">
          <p className="text-xs font-semibold tracking-wide text-slate-500 uppercase">Scope</p>
          {selectedDocumentIds.length ? (
            <button
              type="button"
              onClick={onClearScope}
              className="text-xs text-slate-500 hover:underline"
            >
              All documents
            </button>
          ) : null}
        </div>
        <p className="text-xs text-slate-500">
          {selectedDocumentIds.length
            ? `Searching ${selectedDocumentIds.length} selected document(s).`
            : "Searching every uploaded document."}
        </p>
        {documents.length === 0 ? (
          <p className="px-1 text-xs text-slate-500">No ready documents yet.</p>
        ) : (
          <div className="flex max-h-[26vh] flex-col gap-0.5 overflow-y-auto">
            {documents.map((doc) => (
              <label
                key={doc.id}
                className="flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1 text-xs text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
              >
                <input
                  type="checkbox"
                  checked={selectedDocumentIds.includes(doc.id)}
                  onChange={() => onToggleDocument(doc.id)}
                  disabled={disabled}
                  className="h-3.5 w-3.5 rounded border-slate-300 text-brand-600 focus:ring-brand-500 dark:border-slate-600 dark:bg-slate-800"
                />
                <span className="truncate" title={doc.original_filename}>
                  {doc.original_filename}
                </span>
              </label>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}

export default function ChatPage() {
  const toast = useToast();
  const [conversations, setConversations] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState("");
  const [useHistory, setUseHistory] = useState(true);
  const [documents, setDocuments] = useState([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState([]);
  const [activeSource, setActiveSource] = useState(null);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const abortRef = useRef(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  const loadConversations = useCallback(async () => {
    try {
      const page = await api.listConversations({ page_size: 50 });
      setConversations(page.items || []);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  const loadDocuments = useCallback(async () => {
    try {
      const page = await api.listDocuments({ page_size: 100, status: "ready" });
      setDocuments(page.items || []);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    loadConversations();
    loadDocuments();
  }, [loadConversations, loadDocuments]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // Auto-grow the composer: reset to the natural height, then expand up to the
  // CSS cap (max-h-48). Keeps a one-line box that grows with the message and
  // never scrolls internally, which is the expected production chat behaviour.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [question]);

  // Drop selected documents that no longer exist. The rail is loaded once per
  // mount, so a document deleted (then re-uploaded under a new id) could linger
  // in the selection and be sent as an explicit scope that matches nothing -
  // which suppresses every source. Pruning keeps the scope honest.
  useEffect(() => {
    setSelectedDocumentIds((prev) => {
      if (!prev.length) return prev;
      const alive = new Set(documents.map((doc) => doc.id));
      const next = prev.filter((id) => alive.has(id));
      return next.length === prev.length ? prev : next;
    });
  }, [documents]);

  function toggleDocumentScope(id) {
    setSelectedDocumentIds((prev) =>
      prev.includes(id) ? prev.filter((value) => value !== id) : [...prev, id],
    );
  }

  function startNewConversation() {
    setConversationId(null);
    setMessages([]);
    setError("");
    setActiveSource(null);
    inputRef.current?.focus();
  }

  async function openConversation(id) {
    setError("");
    setActiveSource(null);
    setLoadingHistory(true);
    try {
      const detail = await api.getConversation(id);
      setConversationId(id);
      setMessages(
        (detail.messages || []).map((message) => ({
          ...message,
          ...parseStored(message.sources),
        })),
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingHistory(false);
    }
  }

  async function removeConversation(id) {
    try {
      await api.deleteConversation(id);
      if (id === conversationId) startNewConversation();
      await loadConversations();
      toast.success("Conversation deleted.");
    } catch (err) {
      toast.error(err.message);
    }
  }

  async function renameConversation(id, title) {
    try {
      await api.renameConversation(id, { title });
      await loadConversations();
      toast.success("Conversation renamed.");
    } catch (err) {
      toast.error(err.message);
    }
  }

  async function handleAsk(event) {
    event?.preventDefault();
    const text = question.trim();
    if (!text || streaming) return;

    setError("");
    setQuestion("");
    setStreaming(true);
    setActiveSource(null);

    const assistantId = `local-assistant-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      { id: `local-user-${Date.now()}`, role: "user", content: text },
      { id: assistantId, role: "assistant", content: "", sources: [], streaming: true },
    ]);

    const controller = new AbortController();
    abortRef.current = controller;

    const updateAssistant = (updater) =>
      setMessages((prev) =>
        prev.map((message) =>
          message.id === assistantId ? { ...message, ...updater(message) } : message,
        ),
      );

    try {
      await streamChat(
        {
          question: text,
          conversation_id: conversationId,
          use_history: useHistory,
          document_ids: selectedDocumentIds.length ? selectedDocumentIds : undefined,
        },
        {
          onStart: (data) => {
            if (data.conversation_id && data.conversation_id !== conversationId) {
              setConversationId(data.conversation_id);
            }
          },
          onSources: (sources) => updateAssistant(() => ({ sources })),
          onToken: (token) =>
            updateAssistant((message) => ({ content: `${message.content}${token}` })),
          onDone: (data) => {
            updateAssistant(() => ({
              streaming: false,
              content: data.answer,
              latency_ms: data.latency_ms,
              token_count: data.token_count,
              citations: data.citations || [],
            }));
          },
          onError: (payload) => {
            setError(payload?.message || "The stream failed.");
            updateAssistant(() => ({ streaming: false }));
          },
        },
        controller.signal,
      );
    } catch (err) {
      if (err.name !== "AbortError") setError(err.message || "Streaming failed.");
      updateAssistant(() => ({ streaming: false }));
    } finally {
      setStreaming(false);
      abortRef.current = null;
      loadConversations();
    }
  }

  function stopStreaming() {
    abortRef.current?.abort();
    setStreaming(false);
  }

  return (
    <div className="flex min-h-0 flex-col gap-4 lg:grid lg:h-[calc(100dvh-6.5rem)] lg:grid-cols-[260px_1fr] lg:gap-6">
      <ConversationRail
        conversations={conversations}
        activeId={conversationId}
        onOpen={openConversation}
        onDelete={removeConversation}
        onRename={renameConversation}
        onNew={startNewConversation}
        documents={documents}
        selectedDocumentIds={selectedDocumentIds}
        onToggleDocument={toggleDocumentScope}
        onClearScope={() => setSelectedDocumentIds([])}
        disabled={streaming}
      />

      <Card className="flex min-h-[70vh] flex-1 flex-col overflow-hidden p-0 lg:min-h-0">
        <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-5">
          {loadingHistory ? (
            <div className="flex h-full items-center justify-center gap-2 text-sm text-slate-500">
              <Spinner /> Loading conversation…
            </div>
          ) : messages.length === 0 ? (
            <EmptyState
              icon={SparkIcon}
              title="Ask about your knowledge base"
              description="Every answer is grounded in your uploaded documents and cited back to the exact chunks that produced it."
              action={
                <div className="flex flex-wrap justify-center gap-2">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      onClick={() => {
                        setQuestion(suggestion);
                        inputRef.current?.focus();
                      }}
                      className="rounded-full border border-slate-200 px-3 py-1.5 text-xs text-slate-600 transition-colors hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              }
            />
          ) : (
            messages.map((message, index) => (
              <MessageBubble
                key={message.id}
                message={message}
                activeSource={index === messages.length - 1 ? activeSource : null}
                onFocusSource={setActiveSource}
              />
            ))
          )}
          <div ref={bottomRef} />
        </div>

        <form
          onSubmit={handleAsk}
          className="shrink-0 border-t border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-900"
        >
          <Alert tone="error" onDismiss={() => setError("")}>
            {error}
          </Alert>

          {/* Composer shell: textarea and the Send/Stop button share one row,
              vertically centred (items-center) so the button sits in the middle
              of the input as it grows. */}
          <div
            className={`flex items-center gap-2 rounded-2xl border bg-white p-2 pl-3 shadow-sm transition-colors focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-100 dark:bg-slate-950 dark:focus-within:ring-brand-950 ${
              streaming ? "border-slate-200 dark:border-slate-800" : "border-slate-300 dark:border-slate-700"
            }`}
          >
            <textarea
              ref={inputRef}
              rows={1}
              name="question"
              placeholder="Ask about your knowledge base…"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  handleAsk(event);
                }
              }}
              disabled={streaming}
              className="block max-h-48 flex-1 resize-none bg-transparent py-1.5 text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400 disabled:text-slate-500 dark:text-slate-100"
            />

            {streaming ? (
              <Button
                variant="secondary"
                onClick={stopStreaming}
                size="sm"
                className="h-9 shrink-0 self-center rounded-xl px-3"
                aria-label="Stop generating"
              >
                <StopIcon className="h-4 w-4" />
                Stop
              </Button>
            ) : (
              <Button
                type="submit"
                size="sm"
                disabled={!question.trim()}
                className="h-9 shrink-0 self-center rounded-xl px-3"
                aria-label="Send question"
              >
                <SendIcon className="h-4 w-4" />
                Send
              </Button>
            )}
          </div>

          <div className="mt-2 flex items-center justify-between gap-2 px-1">
            <Checkbox
              label="Include conversation history"
              checked={useHistory}
              onChange={(event) => setUseHistory(event.target.checked)}
              disabled={streaming}
            />
            <div className="flex items-center gap-2">
              <span className="inline-flex min-w-0 items-center gap-1 rounded-full border border-slate-200 px-2 py-0.5 text-[11px] text-slate-500 dark:border-slate-800 dark:text-slate-400">
                <DocumentsIcon className="h-3 w-3 shrink-0" />
                <span className="truncate">
                  {selectedDocumentIds.length
                    ? `${selectedDocumentIds.length} doc${selectedDocumentIds.length > 1 ? "s" : ""} in scope`
                    : "All documents"}
                </span>
              </span>
              <span className="hidden text-[11px] text-slate-400 sm:inline">
                Enter to send · Shift+Enter for newline
              </span>
            </div>
          </div>
        </form>
      </Card>
    </div>
  );
}
