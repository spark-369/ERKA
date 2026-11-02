"use client";

/**
 * Console chrome: sidebar navigation, top bar, theme switcher and user menu.
 *
 * Kept separate from the route layout so the layout itself stays a thin
 * composition and the pieces are individually reusable (e.g. the settings page
 * mounts `ThemeSwitcher` on its own).
 */

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import {
  ChatIcon,
  CloseIcon,
  DashboardIcon,
  DocumentsIcon,
  LogoutIcon,
  MoonIcon,
  SettingsIcon,
  ShieldIcon,
  SunIcon,
} from "@/components/icons";
import { Spinner } from "@/components/ui";
import { useAuth } from "@/lib/auth";
import { useTheme } from "@/lib/theme";

export const NAV = [
  { href: "/app/chat", label: "Chat", icon: ChatIcon },
  { href: "/app/documents", label: "Documents", icon: DocumentsIcon },
  { href: "/app/dashboard", label: "Dashboard", icon: DashboardIcon },
  { href: "/app/settings", label: "Settings", icon: SettingsIcon },
];

// --------------------------------------------------------------------------- //
// Theme switcher
// --------------------------------------------------------------------------- //
export function ThemeSwitcher({ compact = false }) {
  const { theme, setTheme } = useTheme();

  if (compact) {
    const isDark = theme === "dark";
    const Icon = isDark ? SunIcon : MoonIcon;
    return (
      <button
        type="button"
        onClick={() => setTheme(isDark ? "light" : "dark")}
        className="inline-flex h-9 w-9 items-center justify-center rounded-lg text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-700 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-200"
        title={`Switch to ${isDark ? "light" : "dark"} mode`}
        aria-label="Toggle theme"
      >
        <Icon className="h-4 w-4" />
      </button>
    );
  }

  const options = [
    { value: "light", label: "Light", icon: SunIcon },
    { value: "dark", label: "Dark", icon: MoonIcon },
  ];

  return (
    <div className="inline-flex rounded-lg border border-slate-200 p-0.5 dark:border-slate-800">
      {options.map((option) => {
        const active = theme === option.value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => setTheme(option.value)}
            className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
              active
                ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900"
                : "text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
            }`}
          >
            <option.icon className="h-3.5 w-3.5" />
            {option.label}
          </button>
        );
      })}
    </div>
  );
}


// --------------------------------------------------------------------------- //
// User menu
// --------------------------------------------------------------------------- //
function UserMenu() {
  const router = useRouter();
  const { user, logout } = useAuth();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onClick = (event) => {
      if (ref.current && !ref.current.contains(event.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [open]);

  async function handleLogout() {
    setBusy(true);
    try {
      await logout();
      router.replace("/login");
    } finally {
      setBusy(false);
    }
  }

  const initial = (user?.full_name || user?.email || "?").trim().charAt(0).toUpperCase();

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex items-center gap-2 rounded-lg p-1 pr-2 transition-colors hover:bg-slate-100 dark:hover:bg-slate-800"
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-slate-900 text-xs font-semibold text-white dark:bg-white dark:text-slate-900">
          {initial}
        </span>
        <span className="hidden max-w-[10rem] truncate text-sm text-slate-600 sm:inline dark:text-slate-300">
          {user?.full_name || user?.email}
        </span>
      </button>

      {open ? (
        <div
          role="menu"
          className="animate-fade-in absolute right-0 z-30 mt-2 w-60 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-lg dark:border-slate-800 dark:bg-slate-900"
        >
          <div className="border-b border-slate-100 px-4 py-3 dark:border-slate-800">
            <p className="truncate text-sm font-medium">{user?.full_name || "Signed in"}</p>
            <p className="truncate text-xs text-slate-500">{user?.email}</p>
            <span className="mt-1.5 inline-flex items-center gap-1 text-[11px] text-slate-400">
              <ShieldIcon className="h-3 w-3" />
              {user?.role}
            </span>
          </div>
          <div className="flex flex-col p-1">
            <Link
              href="/app/settings"
              onClick={() => setOpen(false)}
              className="flex items-center gap-2 rounded-md px-3 py-2 text-sm text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              <SettingsIcon className="h-4 w-4" />
              Settings
            </Link>
            <button
              type="button"
              onClick={handleLogout}
              disabled={busy}
              className="flex items-center gap-2 rounded-md px-3 py-2 text-left text-sm text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950"
            >
              {busy ? <Spinner className="h-4 w-4" /> : <LogoutIcon className="h-4 w-4" />}
              Sign out
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}


// --------------------------------------------------------------------------- //
// Sidebar
// --------------------------------------------------------------------------- //
export function Sidebar({ open, onClose }) {
  const pathname = usePathname();

  return (
    <>
      {open ? (
        <div
          className="fixed inset-0 z-30 bg-slate-900/40 backdrop-blur-sm lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      ) : null}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-slate-200 bg-white transition-transform duration-200 lg:static lg:translate-x-0 dark:border-slate-800 dark:bg-slate-900 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex h-14 items-center justify-between border-b border-slate-100 px-4 dark:border-slate-800">
          <Link href="/app/chat" className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-900 text-xs font-bold text-white dark:bg-white dark:text-slate-900">
              R
            </span>
            <span className="text-sm font-semibold tracking-tight">RAG Assistant</span>
          </Link>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 lg:hidden dark:hover:text-slate-200"
            aria-label="Close navigation"
          >
            <CloseIcon className="h-4 w-4" />
          </button>
        </div>

        <nav className="flex flex-1 flex-col gap-1 p-3">
          {NAV.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={onClose}
                aria-current={active ? "page" : undefined}
                className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                  active
                    ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900"
                    : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
                }`}
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex flex-col gap-3 border-t border-slate-100 p-3 dark:border-slate-800">
          <div className="flex items-center justify-between px-1">
            <span className="text-xs text-slate-400">Appearance</span>
            <ThemeSwitcher />
          </div>
        </div>
      </aside>
    </>
  );
}

// --------------------------------------------------------------------------- //
// Top bar
// --------------------------------------------------------------------------- //
export function TopBar({ onMenu }) {
  const pathname = usePathname();
  const current = NAV.find((item) => item.href === pathname);

  return (
    <header className="sticky top-0 z-20 flex h-14 items-center justify-between gap-3 border-b border-slate-200 bg-white/80 px-4 backdrop-blur dark:border-slate-800 dark:bg-slate-900/80">
      <div className="flex min-w-0 items-center gap-3">
        <button
          type="button"
          onClick={onMenu}
          className="inline-flex h-9 w-9 items-center justify-center rounded-lg text-slate-500 hover:bg-slate-100 lg:hidden dark:hover:bg-slate-800"
          aria-label="Open navigation"
        >
          <span className="flex flex-col gap-1">
            <span className="block h-0.5 w-4 bg-current" />
            <span className="block h-0.5 w-4 bg-current" />
            <span className="block h-0.5 w-4 bg-current" />
          </span>
        </button>
        <h1 className="truncate text-sm font-semibold tracking-tight">
          {current?.label || "Console"}
        </h1>
      </div>

      <div className="flex items-center gap-2">
        <div className="hidden sm:block">
          <ThemeSwitcher compact />
        </div>
        <UserMenu />
      </div>
    </header>
  );
}
