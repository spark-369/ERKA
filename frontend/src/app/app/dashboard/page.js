"use client";

/**
 * Dashboard: knowledge-base counters, dependency health and the background-task
 * audit trail — the operational view of the RAG stack.
 */

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  AlertIcon,
  ArrowRightIcon,
  ChatIcon,
  CheckIcon,
  DashboardIcon,
  DocumentsIcon,
  HeartPulseIcon,
  RefreshIcon,
} from "@/components/icons";
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  PageHeader,
  ProgressBar,
  Skeleton,
  StatCard,
  TaskStatusBadge,
  formatRelative,
} from "@/components/ui";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useToast } from "@/lib/toast";

function CounterTile({ label, value, tone = "slate" }) {
  return <StatCard label={label} value={value} tone={tone} />;
}

function HealthRow({ name, value }) {
  const degraded = String(value).startsWith("degraded");
  const failed = String(value).startsWith("error");
  const tone = failed ? "red" : degraded ? "amber" : "green";
  const Icon = failed ? AlertIcon : CheckIcon;
  return (
    <div className="flex items-center justify-between gap-3 border-b border-slate-100 py-2.5 text-sm last:border-0 dark:border-slate-800">
      <span className="flex items-center gap-2 font-medium capitalize">
        <Icon
          className={`h-4 w-4 ${
            failed
              ? "text-red-500"
              : degraded
                ? "text-amber-500"
                : "text-emerald-500"
          }`}
        />
        {name.replace(/_/g, " ")}
      </span>
      <Badge tone={tone}>{value}</Badge>
    </div>
  );
}


export default function DashboardPage() {
  const { user } = useAuth();
  const toast = useToast();
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [probing, setProbing] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [userStats, readiness, taskPage] = await Promise.all([
        api.stats(),
        api.readiness().catch(() => null),
        api.listTasks({ page_size: 8 }),
      ]);
      setStats(userStats);
      setHealth(readiness);
      setTasks(taskPage.items || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function dispatchProbe() {
    setProbing(true);
    try {
      await api.dispatchWorkerHealthProbe();
      toast.success("Worker health probe dispatched.");
      await load();
    } catch (err) {
      toast.error(err.message);
    } finally {
      setProbing(false);
    }
  }

  const documents = stats?.documents || {};

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Dashboard"
        description={
          <>
            Signed in as {user?.email} · <span className="capitalize">{user?.role}</span>
          </>
        }
        actions={
          <>
            <Button variant="secondary" onClick={dispatchProbe} loading={probing}>
              <HeartPulseIcon className="h-4 w-4" />
              Run worker probe
            </Button>
            <Button variant="secondary" onClick={load} loading={loading}>
              <RefreshIcon className="h-4 w-4" />
              Refresh
            </Button>
          </>
        }
      />

      <Alert tone="error" onDismiss={() => setError("")}>
        {error}
      </Alert>

      {/* ------------------------------------------------------------ counters */}
      <section className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <CounterTile label="Documents" value={documents.total ?? 0} />
        <CounterTile label="Ready" value={documents.ready ?? 0} tone="green" />
        <CounterTile label="Processing" value={documents.processing ?? 0} tone="blue" />
        <CounterTile label="Pending" value={documents.pending ?? 0} tone="amber" />
        <CounterTile label="Failed" value={documents.failed ?? 0} tone="red" />
        <CounterTile label="Conversations" value={stats?.conversations ?? 0} tone="brand" />
      </section>

      {/* ---------------------------------------------------------- quick links */}
      <section className="grid gap-3 sm:grid-cols-2">
        <Link
          href="/app/documents"
          className="group flex items-center justify-between gap-4 rounded-xl border border-slate-200 bg-white px-4 py-3 transition-colors hover:border-brand-300 hover:bg-brand-50/40 dark:border-slate-800 dark:bg-slate-900 dark:hover:border-brand-800 dark:hover:bg-brand-950/20"
        >
          <span className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-50 text-brand-600 dark:bg-brand-950/50 dark:text-brand-400">
              <DocumentsIcon className="h-4 w-4" />
            </span>
            <span className="flex flex-col">
              <span className="text-sm font-medium">Manage documents</span>
              <span className="text-xs text-slate-500 dark:text-slate-400">
                Upload, tag and re-index your knowledge base
              </span>
            </span>
          </span>
          <ArrowRightIcon className="h-4 w-4 text-slate-400 transition-transform group-hover:translate-x-0.5" />
        </Link>
        <Link
          href="/app/chat"
          className="group flex items-center justify-between gap-4 rounded-xl border border-slate-200 bg-white px-4 py-3 transition-colors hover:border-brand-300 hover:bg-brand-50/40 dark:border-slate-800 dark:bg-slate-900 dark:hover:border-brand-800 dark:hover:bg-brand-950/20"
        >
          <span className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-50 text-brand-600 dark:bg-brand-950/50 dark:text-brand-400">
              <ChatIcon className="h-4 w-4" />
            </span>
            <span className="flex flex-col">
              <span className="text-sm font-medium">Start a chat</span>
              <span className="text-xs text-slate-500 dark:text-slate-400">
                Ask questions and get cited, streamed answers
              </span>
            </span>
          </span>
          <ArrowRightIcon className="h-4 w-4 text-slate-400 transition-transform group-hover:translate-x-0.5" />
        </Link>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* ------------------------------------------------------------ health */}
        <Card>
          <CardHeader
            title="Service health"
            description="Live readiness probe across every dependency."
            action={
              health ? (
                <Badge tone={health.status === "ok" ? "green" : "amber"}>
                  {health.status}
                </Badge>
              ) : null
            }
          />
          <div className="px-5 py-2">
            {loading && !health ? (
              <div className="flex flex-col gap-2 py-2">
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-8 w-full" />
              </div>
            ) : health ? (
              Object.entries(health.dependencies || {}).map(([name, value]) => (
                <HealthRow key={name} name={name} value={value} />
              ))
            ) : (
              <p className="py-4 text-sm text-slate-500">
                Readiness probe unavailable (the API may be starting up).
              </p>
            )}
          </div>
        </Card>

        {/* ------------------------------------------------------------- tasks */}
        <Card>
          <CardHeader
            title="Background tasks"
            description="Durable audit trail of dispatched background work."
          />
          {loading && tasks.length === 0 ? (
            <div className="flex flex-col gap-2 p-5">
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
            </div>
          ) : tasks.length === 0 ? (
            <EmptyState
              icon={DashboardIcon}
              title="No background tasks yet"
              description="Upload a document to trigger the ingestion pipeline."
            />
          ) : (
            <div className="flex flex-col gap-2 p-4">
              {tasks.map((task) => (
                <div
                  key={task.id}
                  className="flex flex-col gap-2 rounded-lg border border-slate-100 px-3 py-2.5 text-sm dark:border-slate-800"
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="truncate font-medium">{task.name}</span>
                    <TaskStatusBadge status={task.status} />
                  </div>
                  <div className="flex items-center justify-between gap-3 text-xs text-slate-500">
                    <span>
                      {formatRelative(task.created_at)}
                      {task.resource_type ? ` · ${task.resource_type}` : ""}
                    </span>
                    <span className="tabular-nums">{task.progress}%</span>
                  </div>
                  {task.status === "running" || task.status === "queued" ? (
                    <ProgressBar value={task.progress} />
                  ) : null}
                  {task.error_message ? (
                    <p className="truncate text-xs text-red-600 dark:text-red-400">
                      {task.error_message}
                    </p>
                  ) : null}
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
