"use client";

/**
 * Route-level error boundary for the console. Next.js renders this when a
 * segment throws during rendering, instead of a blank screen.
 */

import { useEffect } from "react";
import Link from "next/link";
import { AlertIcon, RefreshIcon } from "@/components/icons";
import { Button, Card } from "@/components/ui";

export default function AppError({ error, reset }) {
  useEffect(() => {
    // Surface the error in the browser console for debugging.
    console.error(error);
  }, [error]);

  return (
    <div className="flex flex-1 items-center justify-center py-16">
      <Card className="flex w-full max-w-md flex-col items-center gap-4 p-8 text-center">
        <span className="flex h-11 w-11 items-center justify-center rounded-full bg-red-50 text-red-600 dark:bg-red-950 dark:text-red-400">
          <AlertIcon className="h-5 w-5" />
        </span>
        <div className="flex flex-col gap-1">
          <h1 className="text-lg font-semibold tracking-tight">Something went wrong</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            {error?.message || "An unexpected error occurred while rendering this page."}
          </p>
        </div>
        <div className="flex flex-wrap justify-center gap-2">
          <Button onClick={reset}>
            <RefreshIcon className="h-4 w-4" />
            Try again
          </Button>
          <Link href="/app/chat">
            <Button variant="secondary">Back to chat</Button>
          </Link>
        </div>
      </Card>
    </div>
  );
}
