"use client";

/**
 * Shared presentational primitives (Tailwind v4).
 *
 * Everything is a plain function component with no runtime dependency, so the
 * console stays light and there is a single source of truth for spacing, colour
 * and focus treatment.
 */

import { useEffect, useState } from "react";
import { CloseIcon, SearchIcon } from "@/components/icons";

// --------------------------------------------------------------------------- //
// Layout
// --------------------------------------------------------------------------- //
export function Card({ children, className = "", as: Tag = "div", ...props }) {
  return (
    <Tag
      className={`rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900 ${className}`}
      {...props}
    >
      {children}
    </Tag>
  );
}

/** Brand lockup reused in the sidebar, auth panel and landing header. */
export function BrandMark({ label = "RAG Assistant", size = "md", className = "" }) {
  const box = size === "lg" ? "h-8 w-8 text-base" : "h-7 w-7 text-xs";
  return (
    <span className={`flex items-center gap-2 ${className}`}>
      <span
        className={`flex ${box} items-center justify-center rounded-lg bg-brand-600 font-bold text-white shadow-sm`}
      >
        R
      </span>
      <span className="text-sm font-semibold tracking-tight">{label}</span>
    </span>
  );
}

/** Consistent page title row for the console routes. */
export function PageHeader({ title, description, actions }) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {description ? (
          <p className="text-sm text-slate-500 dark:text-slate-400">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </header>
  );
}

const STAT_TONES = {
  slate: "text-slate-900 dark:text-slate-100",
  green: "text-emerald-600 dark:text-emerald-400",
  amber: "text-amber-600 dark:text-amber-400",
  blue: "text-blue-600 dark:text-blue-400",
  red: "text-red-600 dark:text-red-400",
  brand: "text-brand-600 dark:text-brand-400",
};

/** Numeric tile used across the dashboard and documents consoles. */
export function StatCard({ label, value, tone = "slate", hint, icon: Icon }) {
  return (
    <Card className="px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-medium tracking-wide text-slate-500 uppercase">{label}</p>
        {Icon ? <Icon className="h-4 w-4 text-slate-400" /> : null}
      </div>
      <p className={`mt-1 text-2xl font-semibold tabular-nums ${STAT_TONES[tone] || STAT_TONES.slate}`}>
        {formatNumber(value ?? 0)}
      </p>
      {hint ? <p className="mt-0.5 text-xs text-slate-400">{hint}</p> : null}
    </Card>
  );
}

export function CardHeader({ title, description, action }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-slate-100 px-5 py-4 dark:border-slate-800">
      <div className="flex flex-col gap-0.5">
        <h2 className="font-semibold tracking-tight">{title}</h2>
        {description ? (
          <p className="text-sm text-slate-500 dark:text-slate-400">{description}</p>
        ) : null}
      </div>
      {action}
    </div>
  );
}

export function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-12 text-center">
      {Icon ? (
        <span className="flex h-11 w-11 items-center justify-center rounded-full bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400">
          <Icon className="h-5 w-5" />
        </span>
      ) : null}
      <div className="flex flex-col gap-1">
        <p className="font-medium">{title}</p>
        {description ? (
          <p className="mx-auto max-w-sm text-sm text-slate-500 dark:text-slate-400">
            {description}
          </p>
        ) : null}
      </div>
      {action}
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Controls
// --------------------------------------------------------------------------- //
export function Button({
  children,
  variant = "primary",
  size = "md",
  loading = false,
  className = "",
  type = "button",
  disabled,
  ...props
}) {
  const variants = {
    primary:
      "bg-brand-600 text-white shadow-sm hover:bg-brand-700 disabled:bg-brand-300 dark:bg-brand-500 dark:hover:bg-brand-400 dark:disabled:bg-brand-900",
    secondary:
      "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800",
    ghost:
      "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800",
    danger:
      "bg-red-600 text-white hover:bg-red-500 disabled:bg-red-300 dark:disabled:bg-red-900",
  };
  const sizes = {
    sm: "px-2.5 py-1.5 text-xs",
    md: "px-3.5 py-2 text-sm",
    lg: "px-5 py-2.5 text-sm",
  };
  return (
    <button
      type={type}
      disabled={disabled || loading}
      className={`inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-1 focus-visible:ring-offset-background focus-visible:outline-none disabled:cursor-not-allowed ${variants[variant]} ${sizes[size]} ${className}`}
      {...props}
    >
      {loading ? <Spinner className="h-3.5 w-3.5" /> : null}
      {children}
    </button>
  );
}


export function Input({ label, hint, error, className = "", id, ...props }) {
  const inputId = id || props.name;
  return (
    <label className="flex flex-col gap-1.5 text-sm" htmlFor={inputId}>
      {label ? (
        <span className="font-medium text-slate-700 dark:text-slate-300">{label}</span>
      ) : null}
      <input
        id={inputId}
        aria-invalid={error ? "true" : undefined}
        className={`rounded-lg border bg-white px-3 py-2 text-sm text-slate-900 outline-none transition-colors placeholder:text-slate-400 focus:ring-2 disabled:bg-slate-100 disabled:text-slate-500 dark:bg-slate-950 dark:text-slate-100 dark:disabled:bg-slate-900 ${
          error
            ? "border-red-300 focus:border-red-500 focus:ring-red-100 dark:border-red-900 dark:focus:ring-red-950"
            : "border-slate-300 focus:border-brand-500 focus:ring-brand-100 dark:border-slate-700 dark:focus:ring-brand-950"
        } ${className}`}
        {...props}
      />
      {error ? (
        <span className="text-xs text-red-600 dark:text-red-400">{error}</span>
      ) : hint ? (
        <span className="text-xs text-slate-500 dark:text-slate-400">{hint}</span>
      ) : null}
    </label>
  );
}

export function SearchInput({ value, onChange, placeholder = "Search…", className = "" }) {
  return (
    <div className={`relative ${className}`}>
      <SearchIcon className="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-slate-400" />
      <input
        type="search"
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        className="w-full rounded-lg border border-slate-300 bg-white py-2 pr-3 pl-9 text-sm text-slate-900 outline-none transition-colors placeholder:text-slate-400 focus:border-brand-500 focus:ring-2 focus:ring-brand-100 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:focus:ring-brand-950"
      />
    </div>
  );
}

export function Select({ label, className = "", id, children, ...props }) {
  const inputId = id || props.name;
  return (
    <label className="flex flex-col gap-1.5 text-sm" htmlFor={inputId}>
      {label ? (
        <span className="font-medium text-slate-700 dark:text-slate-300">{label}</span>
      ) : null}
      <select
        id={inputId}
        className={`rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition-colors focus:border-brand-500 focus:ring-2 focus:ring-brand-100 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:focus:ring-brand-950 ${className}`}
        {...props}
      >
        {children}
      </select>
    </label>
  );
}

export function Checkbox({ label, className = "", id, ...props }) {
  const inputId = id || props.name;
  return (
    <label
      htmlFor={inputId}
      className={`flex cursor-pointer items-center gap-2 text-sm text-slate-600 select-none dark:text-slate-300 ${className}`}
    >
      <input
        id={inputId}
        type="checkbox"
        className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500 dark:border-slate-600 dark:bg-slate-800"
        {...props}
      />
      {label}
    </label>
  );
}


// --------------------------------------------------------------------------- //
// Feedback
// --------------------------------------------------------------------------- //
export function Badge({ children, tone = "slate", className = "" }) {
  const tones = {
    slate: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    green: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
    amber: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
    red: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
    blue: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
    brand: "bg-brand-100 text-brand-700 dark:bg-brand-950 dark:text-brand-300",
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

export function Alert({ children, tone = "error", onDismiss }) {
  if (!children) return null;
  const tones = {
    error:
      "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300",
    info: "border-blue-200 bg-blue-50 text-blue-700 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-300",
    success:
      "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-300",
    warning:
      "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300",
  };
  return (
    <div
      className={`flex items-start justify-between gap-3 rounded-lg border px-3 py-2 text-sm ${tones[tone]}`}
      role="alert"
    >
      <span className="flex-1">{children}</span>
      {onDismiss ? (
        <button
          type="button"
          onClick={onDismiss}
          className="shrink-0 opacity-60 hover:opacity-100"
          aria-label="Dismiss"
        >
          <CloseIcon className="h-3.5 w-3.5" />
        </button>
      ) : null}
    </div>
  );
}

export function Spinner({ className = "h-4 w-4" }) {
  return (
    <span
      className={`inline-block animate-spin rounded-full border-2 border-current border-t-transparent ${className}`}
      aria-hidden="true"
    />
  );
}

export function Skeleton({ className = "" }) {
  return <div className={`skeleton rounded-md ${className}`} aria-hidden="true" />;
}

/** Status pill for document ingestion states. */
export function StatusBadge({ status }) {
  const map = {
    ready: { tone: "green", label: "Ready" },
    processing: { tone: "blue", label: "Processing" },
    pending: { tone: "amber", label: "Pending" },
    failed: { tone: "red", label: "Failed" },
  };
  const entry = map[status] || { tone: "slate", label: status || "Unknown" };
  return <Badge tone={entry.tone}>{entry.label}</Badge>;
}

/** Status pill for background-task states. */
export function TaskStatusBadge({ status }) {
  const map = {
    succeeded: "green",
    running: "blue",
    queued: "amber",
    failed: "red",
  };
  return <Badge tone={map[status] || "slate"}>{status}</Badge>;
}

export function ProgressBar({ value = 0, className = "" }) {
  const percent = Math.max(0, Math.min(100, Math.round(value)));
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div
        className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800"
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className="h-full rounded-full bg-brand-600 transition-all duration-300 dark:bg-brand-500"
          style={{ width: `${percent}%` }}
        />
      </div>
      <span className="w-9 text-right text-xs tabular-nums text-slate-500">{percent}%</span>
    </div>
  );
}


// --------------------------------------------------------------------------- //
// Modal
// --------------------------------------------------------------------------- //
export function Modal({ open, onClose, title, description, children, footer }) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (event) => {
      if (event.key === "Escape") onClose?.();
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="animate-fade-in relative z-10 w-full max-w-lg rounded-xl border border-slate-200 bg-white shadow-xl dark:border-slate-800 dark:bg-slate-900"
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-100 px-5 py-4 dark:border-slate-800">
          <div className="flex flex-col gap-0.5">
            <h2 className="font-semibold tracking-tight">{title}</h2>
            {description ? (
              <p className="text-sm text-slate-500 dark:text-slate-400">{description}</p>
            ) : null}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
            aria-label="Close dialog"
          >
            <CloseIcon className="h-4 w-4" />
          </button>
        </div>
        <div className="px-5 py-4">{children}</div>
        {footer ? (
          <div className="flex justify-end gap-2 border-t border-slate-100 px-5 py-4 dark:border-slate-800">
            {footer}
          </div>
        ) : null}
      </div>
    </div>
  );
}

/** Copy-to-clipboard button with a transient "Copied" state. */
export function CopyButton({ value, className = "", label = "Copy" }) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(value ?? "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (insecure context) */
    }
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={`inline-flex items-center gap-1 text-xs text-slate-400 transition-colors hover:text-slate-600 dark:hover:text-slate-200 ${className}`}
      title={label}
    >
      {copied ? "Copied" : label}
    </button>
  );
}

// --------------------------------------------------------------------------- //
// Formatters
// --------------------------------------------------------------------------- //
export function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function formatDate(value) {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleString();
  } catch {
    return String(value);
  }
}

export function formatRelative(value) {
  if (!value) return "—";
  const date = new Date(value);
  const diff = Date.now() - date.getTime();
  const minutes = Math.round(diff / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days}d ago`;
  return date.toLocaleDateString();
}

export function formatNumber(value) {
  if (value === null || value === undefined) return "0";
  return new Intl.NumberFormat().format(value);
}
