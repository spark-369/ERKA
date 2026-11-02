"use client";

/**
 * Auth shell: a two-column marketing panel + the form card.
 */

import Link from "next/link";
import { CheckIcon } from "@/components/icons";
import { ThemeSwitcher } from "@/components/shell";

const HIGHLIGHTS = [
  "Answers grounded in your own documents",
  "Citations back to the exact source chunks",
  "Runs fully on your own infrastructure",
];

export default function AuthShell({ title, subtitle, children, footer }) {
  return (
    <div className="flex min-h-screen">
      {/* ------------------------------------------------------ brand panel */}
      <div className="relative hidden w-1/2 flex-col justify-between bg-slate-900 p-12 text-white lg:flex dark:bg-slate-950">
        <Link href="/" className="flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-white text-sm font-bold text-slate-900">
            R
          </span>
          <span className="text-sm font-semibold tracking-tight">RAG Assistant</span>
        </Link>

        <div className="flex flex-col gap-6">
          <h2 className="max-w-md text-3xl font-semibold tracking-tight text-balance">
            A private knowledge assistant for your domain documents.
          </h2>
          <ul className="flex flex-col gap-3">
            {HIGHLIGHTS.map((item) => (
              <li key={item} className="flex items-center gap-3 text-sm text-slate-300">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-400">
                  <CheckIcon className="h-3 w-3" />
                </span>
                {item}
              </li>
            ))}
          </ul>
        </div>

        <p className="text-xs text-slate-500">
          FastAPI · Celery · ChromaDB · Qwen2.5
        </p>
      </div>

      {/* ------------------------------------------------------------ form */}
      <div className="flex flex-1 flex-col">
        <div className="flex items-center justify-between px-6 py-4 lg:justify-end">
          <Link
            href="/"
            className="text-sm font-semibold tracking-tight lg:hidden"
          >
            RAG Assistant
          </Link>
          <ThemeSwitcher compact />
        </div>

        <div className="flex flex-1 items-center justify-center px-6 pb-12">
          <div className="w-full max-w-sm">
            <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
            <p className="mt-1 mb-6 text-sm text-slate-500">{subtitle}</p>
            {children}
            {footer ? <div className="mt-6 text-center text-sm text-slate-500">{footer}</div> : null}
          </div>
        </div>
      </div>
    </div>
  );
}
