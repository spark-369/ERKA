"use client";

/**
 * Protected console shell: auth gate + sidebar chrome.
 *
 * The auth gate is a wrapper *around* `{children}` rather than an early return
 * from the layout itself. A layout that returns early without rendering
 * `{children}` drops every segment in its subtree, which Next.js reports as
 * "Could not validate that a segment in your UI has instant navigation"
 * (see https://nextjs.org/docs/messages/instant-unrendered-segment).
 */

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Sidebar, TopBar } from "@/components/shell";
import { Spinner } from "@/components/ui";
import { useAuth } from "@/lib/auth";

/** Client-side guard: the API is the real authority, this is just UX. */
function AuthGate({ children }) {
  const router = useRouter();
  const { loading, isAuthenticated } = useAuth();

  useEffect(() => {
    if (!loading && !isAuthenticated) router.replace("/login");
  }, [loading, isAuthenticated, router]);

  // `{children}` is always rendered so the segment stays in the render tree -
  // a guard that returns early (or `null`) during SSR drops every segment below
  // it and trips the instant-navigation validation. The blocker is drawn *over*
  // the subtree instead of replacing it.
  const blocked = loading || !isAuthenticated;

  return (
    <>
      {children}
      {blocked ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center gap-2 bg-slate-50 text-sm text-slate-500 dark:bg-slate-950">
          <Spinner />
          {loading ? "Restoring your session…" : "Redirecting to sign in…"}
        </div>
      ) : null}
    </>
  );
}

export default function AppLayout({ children }) {
  const { isAuthenticated } = useAuth();
  const [navOpen, setNavOpen] = useState(false);

  return (
    <div className="flex min-h-screen">
      {isAuthenticated ? <Sidebar open={navOpen} onClose={() => setNavOpen(false)} /> : null}

      <div className="flex min-w-0 flex-1 flex-col">
        {isAuthenticated ? <TopBar onMenu={() => setNavOpen(true)} /> : null}
        <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col px-4 py-6 sm:px-6">
          <AuthGate>{children}</AuthGate>
        </main>
      </div>
    </div>
  );
}
