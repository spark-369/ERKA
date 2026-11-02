"use client";

/** `/app` has no content of its own — send the user to the chat console. */

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { Spinner } from "@/components/ui";

export default function AppIndexPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/app/chat");
  }, [router]);

  return (
    <div className="flex flex-1 items-center justify-center gap-2 text-sm text-slate-500">
      <Spinner /> Opening the console…
    </div>
  );
}
