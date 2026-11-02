"use client";

/**
 * Minimal toast system (no dependency).
 *
 * `ToastProvider` renders a fixed stack of notifications; `useToast()` returns
 * `{ success, error, info, dismiss }` so any component can surface transient
 * feedback (upload accepted, delete failed, ...) without prop drilling.
 */

import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";
import { AlertIcon, CheckIcon } from "@/components/icons";

const ToastContext = createContext(null);

const TONES = {
  success: {
    icon: CheckIcon,
    ring: "border-emerald-200 dark:border-emerald-900",
    accent: "text-emerald-600 dark:text-emerald-400",
  },
  error: {
    icon: AlertIcon,
    ring: "border-red-200 dark:border-red-900",
    accent: "text-red-600 dark:text-red-400",
  },
  info: {
    icon: AlertIcon,
    ring: "border-slate-200 dark:border-slate-700",
    accent: "text-slate-500 dark:text-slate-400",
  },
};

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const counter = useRef(0);

  const dismiss = useCallback((id) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    (message, tone = "info", duration = 4000) => {
      counter.current += 1;
      const id = counter.current;
      setToasts((prev) => [...prev, { id, message, tone }]);
      if (duration > 0) {
        setTimeout(() => dismiss(id), duration);
      }
      return id;
    },
    [dismiss],
  );

  const value = useMemo(
    () => ({
      success: (message) => push(message, "success"),
      error: (message) => push(message, "error", 6000),
      info: (message) => push(message, "info"),
      dismiss,
    }),
    [push, dismiss],
  );

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        className="pointer-events-none fixed inset-x-0 bottom-0 z-50 flex flex-col items-center gap-2 p-4 sm:items-end"
        role="region"
        aria-label="Notifications"
      >
        {toasts.map((toast) => {
          const tone = TONES[toast.tone] || TONES.info;
          const ToneIcon = tone.icon;
          return (
            <div
              key={toast.id}
              role="status"
              className={`animate-slide-in pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-xl border bg-white px-4 py-3 text-sm shadow-lg dark:bg-slate-900 ${tone.ring}`}
            >
              <ToneIcon className={`mt-0.5 h-4 w-4 shrink-0 ${tone.accent}`} />
              <p className="flex-1 leading-5">{toast.message}</p>
              <button
                type="button"
                onClick={() => dismiss(toast.id)}
                className="shrink-0 text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                aria-label="Dismiss notification"
              >
                ✕
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) throw new Error("useToast must be used inside a <ToastProvider>.");
  return context;
}
