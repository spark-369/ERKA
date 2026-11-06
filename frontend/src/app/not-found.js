"use client";

/** Global 404 for unmatched routes. */

import Link from "next/link";
import { ArrowRightIcon } from "@/components/icons";
import { BrandMark, Button } from "@/components/ui";

export default function NotFound() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-5 px-6 text-center">
      <BrandMark className="mb-2" />
      <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-50 text-2xl font-bold text-brand-600 dark:bg-brand-950/50 dark:text-brand-400">
        404
      </span>
      <h1 className="text-2xl font-semibold tracking-tight">Page not found</h1>
      <p className="max-w-sm text-sm text-slate-500 dark:text-slate-400">
        The page you are looking for does not exist or has been moved.
      </p>
      <Link href="/">
        <Button>
          Back to home
          <ArrowRightIcon className="h-4 w-4" />
        </Button>
      </Link>
    </main>
  );
}
