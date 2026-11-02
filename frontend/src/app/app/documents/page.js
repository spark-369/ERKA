"use client";

/**
 * Documents console: drag-and-drop upload, ingestion monitoring, chunk
 * inspection and lifecycle actions (rename / reindex / delete).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { DocumentsIcon, RefreshIcon, TrashIcon, UploadIcon } from "@/components/icons";
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  Input,
  Modal,
  ProgressBar,
  SearchInput,
  Select,
  Skeleton,
  StatusBadge,
  formatBytes,
  formatDate,
  formatNumber,
} from "@/components/ui";
import { api } from "@/lib/api";
import { useToast } from "@/lib/toast";

const ACCEPT = ".pdf,.txt,.md,.csv,.docx,.json,.html";
const PAGE_SIZE = 10;

const STATUS_OPTIONS = [
  { value: "", label: "All statuses" },
  { value: "ready", label: "Ready" },
  { value: "processing", label: "Processing" },
  { value: "pending", label: "Pending" },
  { value: "failed", label: "Failed" },
];

function StatTile({ label, value, tone = "slate" }) {
  const tones = {
    slate: "text-slate-900 dark:text-slate-100",
    green: "text-emerald-600 dark:text-emerald-400",
    amber: "text-amber-600 dark:text-amber-400",
    blue: "text-blue-600 dark:text-blue-400",
    red: "text-red-600 dark:text-red-400",
  };
  return (
    <Card className="px-4 py-3">
      <p className="text-xs font-medium tracking-wide text-slate-500 uppercase">{label}</p>
      <p className={`mt-1 text-xl font-semibold tabular-nums ${tones[tone]}`}>
        {formatNumber(value)}
      </p>
    </Card>
  );
}

function UploadZone({ onUpload, uploading }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  function handleFiles(files) {
    const file = files?.[0];
    if (file) onUpload(file);
  }

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        handleFiles(event.dataTransfer.files);
      }}
      className={`flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
        dragging
          ? "border-slate-400 bg-slate-50 dark:border-slate-500 dark:bg-slate-800/50"
          : "border-slate-300 dark:border-slate-700"
      }`}
    >
      <span className="flex h-11 w-11 items-center justify-center rounded-full bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400">
        <UploadIcon className="h-5 w-5" />
      </span>
      <div className="flex flex-col gap-1">
        <p className="text-sm font-medium">Drop a document here, or browse</p>
        <p className="text-xs text-slate-500">
          PDF, TXT, MD, CSV, DOCX, JSON or HTML · up to 25 MB
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = "";
        }}
      />
      <Button
        variant="secondary"
        loading={uploading}
        onClick={() => inputRef.current?.click()}
        disabled={uploading}
      >
        {uploading ? "Uploading…" : "Choose file"}
      </Button>
    </div>
  );
}



function ChunkViewer({ document, onClose }) {
  const [chunks, setChunks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!document) return undefined;
    let cancelled = false;
    setLoading(true);
    setError("");
    api
      .documentChunks(document.id, { page_size: 50 })
      .then((page) => {
        if (!cancelled) setChunks(page.items || []);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [document]);

  return (
    <Modal
      open={Boolean(document)}
      onClose={onClose}
      title={document?.original_filename || "Chunks"}
      description={
        document ? `${document.chunk_count} chunks · ${formatBytes(document.size_bytes)}` : ""
      }
      footer={
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      }
    >
      <div className="flex max-h-[50vh] flex-col gap-3 overflow-y-auto">
        {loading ? (
          <>
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </>
        ) : error ? (
          <Alert>{error}</Alert>
        ) : chunks.length === 0 ? (
          <p className="text-sm text-slate-500">No chunks stored for this document.</p>
        ) : (
          chunks.map((chunk) => (
            <div
              key={chunk.id}
              className="rounded-lg border border-slate-200 p-3 text-xs dark:border-slate-800"
            >
              <div className="mb-1.5 flex items-center justify-between">
                <Badge tone="slate">Chunk {chunk.chunk_index + 1}</Badge>
                <span className="text-slate-400">
                  {chunk.char_count} chars · {chunk.token_count} tok
                </span>
              </div>
              <p className="leading-5 whitespace-pre-wrap text-slate-600 dark:text-slate-400">
                {chunk.content}
              </p>
            </div>
          ))
        )}
      </div>
    </Modal>
  );
}


export default function DocumentsPage() {
  const toast = useToast();
  const [documents, setDocuments] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState({});
  const [chunkDoc, setChunkDoc] = useState(null);
  const [renameDoc, setRenameDoc] = useState(null);
  const [renameValue, setRenameValue] = useState("");
  const [pendingDelete, setPendingDelete] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [listing, counters] = await Promise.all([
        api.listDocuments({ page, page_size: PAGE_SIZE, search, status: statusFilter }),
        api.documentStats(),
      ]);
      setDocuments(listing.items || []);
      setTotal(listing.total || 0);
      setStats(counters);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [page, search, statusFilter]);

  useEffect(() => {
    load();
  }, [load]);

  // Poll ingestion progress while any document is still being processed.
  useEffect(() => {
    const pending = documents.filter((doc) =>
      ["pending", "processing"].includes(doc.status),
    );
    if (pending.length === 0) return undefined;

    const timer = setInterval(async () => {
      const updates = {};
      await Promise.all(
        pending.map(async (doc) => {
          try {
            updates[doc.id] = await api.documentStatus(doc.id);
          } catch {
            /* ignore transient polling errors */
          }
        }),
      );
      setProgress((prev) => ({ ...prev, ...updates }));

      const settled = Object.values(updates).some(
        (entry) => entry.status === "ready" || entry.status === "failed",
      );
      if (settled) load();
    }, 2500);

    return () => clearInterval(timer);
  }, [documents, load]);

  const totalPages = Math.max(Math.ceil(total / PAGE_SIZE), 1);

  async function handleUpload(file) {
    setError("");
    setUploading(true);
    try {
      const result = await api.uploadDocument(file, { processNow: true });
      toast.success(`“${result.document.original_filename}” accepted for ingestion.`);
      await load();
    } catch (err) {
      setError(err.message);
      toast.error(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleReindex(id) {
    try {
      await api.reindexDocument(id);
      toast.success("Re-indexing started.");
      await load();
    } catch (err) {
      toast.error(err.message);
    }
  }

  async function confirmDelete() {
    if (!pendingDelete) return;
    try {
      await api.deleteDocument(pendingDelete.id);
      toast.success("Document deleted.");
      setPendingDelete(null);
      await load();
    } catch (err) {
      toast.error(err.message);
    }
  }

  async function submitRename(event) {
    event.preventDefault();
    if (!renameDoc) return;
    try {
      await api.updateDocument(renameDoc.id, { original_filename: renameValue });
      toast.success("Document renamed.");
      setRenameDoc(null);
      await load();
    } catch (err) {
      toast.error(err.message);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
          <p className="text-sm text-slate-500">
            Upload domain documents and monitor their ingestion into the vector store.
          </p>
        </div>
        <Button variant="secondary" onClick={load} loading={loading}>
          <RefreshIcon className="h-4 w-4" />
          Refresh
        </Button>
      </header>

      <Alert tone="error" onDismiss={() => setError("")}>
        {error}
      </Alert>

      <section className="grid gap-3 sm:grid-cols-3 lg:grid-cols-5">
        <StatTile label="Total" value={stats?.total ?? 0} />
        <StatTile label="Ready" value={stats?.ready ?? 0} tone="green" />
        <StatTile label="Processing" value={stats?.processing ?? 0} tone="blue" />
        <StatTile label="Pending" value={stats?.pending ?? 0} tone="amber" />
        <StatTile label="Failed" value={stats?.failed ?? 0} tone="red" />
      </section>

      <Card>
        <CardHeader
          title="Add a document"
          description="Ingestion runs asynchronously — you can keep working while it processes."
        />
        <div className="p-5">
          <UploadZone onUpload={handleUpload} uploading={uploading} />
        </div>
      </Card>

      <Card className="overflow-hidden">
        <CardHeader
          title="Knowledge base"
          description={`${formatNumber(total)} document${total === 1 ? "" : "s"}`}
        />
        <div className="flex flex-wrap items-center gap-3 border-b border-slate-100 p-4 dark:border-slate-800">
          <SearchInput
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setPage(1);
            }}
            placeholder="Search by name or tag…"
            className="min-w-[12rem] flex-1"
          />
          <Select
            value={statusFilter}
            onChange={(event) => {
              setStatusFilter(event.target.value);
              setPage(1);
            }}
            className="w-40"
          >
            {STATUS_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </Select>
        </div>

        {loading ? (
          <div className="flex flex-col gap-2 p-5">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
          </div>
        ) : documents.length === 0 ? (
          <EmptyState
            icon={DocumentsIcon}
            title="No documents yet"
            description="Upload a PDF, TXT, Markdown or DOCX file to start building your knowledge base."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-100 text-xs tracking-wide text-slate-500 uppercase dark:border-slate-800">
                <tr>
                  <th className="px-5 py-3 font-medium">Document</th>
                  <th className="px-3 py-3 font-medium">Status</th>
                  <th className="px-3 py-3 font-medium">Progress</th>
                  <th className="px-3 py-3 font-medium">Chunks</th>
                  <th className="px-3 py-3 font-medium">Size</th>
                  <th className="px-3 py-3 font-medium">Uploaded</th>
                  <th className="px-5 py-3 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {documents.map((doc) => {
                  const live = progress[doc.id];
                  const settled = doc.status === "ready" || doc.status === "failed";
                  const percent = live?.progress ?? (settled ? 100 : 0);
                  return (
                    <tr key={doc.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                      <td className="max-w-[16rem] px-5 py-3">
                        <button
                          type="button"
                          onClick={() => setChunkDoc(doc)}
                          className="flex flex-col items-start text-left"
                          title="View chunks"
                        >
                          <span className="truncate font-medium hover:underline">
                            {doc.original_filename}
                          </span>
                          <span className="truncate text-xs text-slate-500">
                            {doc.extension} · {doc.collection}
                          </span>
                          {doc.error_message ? (
                            <span className="truncate text-xs text-red-600 dark:text-red-400">
                              {doc.error_message}
                            </span>
                          ) : null}
                        </button>
                      </td>
                      <td className="px-3 py-3">
                        <StatusBadge status={live?.status || doc.status} />
                      </td>
                      <td className="w-40 px-3 py-3">
                        <ProgressBar value={percent} />
                      </td>
                      <td className="px-3 py-3 tabular-nums">{formatNumber(doc.chunk_count)}</td>
                      <td className="px-3 py-3 tabular-nums">{formatBytes(doc.size_bytes)}</td>
                      <td className="px-3 py-3 text-xs whitespace-nowrap text-slate-500">
                        {formatDate(doc.created_at)}
                      </td>
                      <td className="px-5 py-3">
                        <div className="flex justify-end gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => {
                              setRenameDoc(doc);
                              setRenameValue(doc.original_filename);
                            }}
                            title="Rename"
                          >
                            Rename
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleReindex(doc.id)}
                            title="Re-index"
                          >
                            <RefreshIcon className="h-3.5 w-3.5" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setPendingDelete(doc)}
                            className="text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950"
                            title="Delete"
                          >
                            <TrashIcon className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {total > PAGE_SIZE ? (
          <div className="flex items-center justify-between border-t border-slate-100 px-5 py-3 text-sm dark:border-slate-800">
            <span className="text-slate-500">
              Page {page} of {totalPages}
            </span>
            <div className="flex gap-2">
              <Button
                variant="secondary"
                size="sm"
                disabled={page <= 1}
                onClick={() => setPage((value) => Math.max(value - 1, 1))}
              >
                Previous
              </Button>
              <Button
                variant="secondary"
                size="sm"
                disabled={page >= totalPages}
                onClick={() => setPage((value) => Math.min(value + 1, totalPages))}
              >
                Next
              </Button>
            </div>
          </div>
        ) : null}
      </Card>

      <ChunkViewer document={chunkDoc} onClose={() => setChunkDoc(null)} />

      <Modal
        open={Boolean(renameDoc)}
        onClose={() => setRenameDoc(null)}
        title="Rename document"
        description="Only the display name changes; the stored file and vectors are untouched."
        footer={
          <>
            <Button variant="secondary" onClick={() => setRenameDoc(null)}>
              Cancel
            </Button>
            <Button type="submit" form="rename-form">
              Save
            </Button>
          </>
        }
      >
        <form id="rename-form" onSubmit={submitRename}>
          <Input
            label="Display name"
            value={renameValue}
            onChange={(event) => setRenameValue(event.target.value)}
          />
        </form>
      </Modal>

      <Modal
        open={Boolean(pendingDelete)}
        onClose={() => setPendingDelete(null)}
        title="Delete document"
        description="This removes the row, its chunks, the stored file and its vectors."
        footer={
          <>
            <Button variant="secondary" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button variant="danger" onClick={confirmDelete}>
              Delete
            </Button>
          </>
        }
      >
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Delete <strong>{pendingDelete?.original_filename}</strong>? This cannot be undone.
        </p>
      </Modal>
    </div>
  );
}
