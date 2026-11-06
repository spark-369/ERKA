"use client";

/**
 * Settings: profile, appearance and account controls.
 *
 * Profile edits go through `PATCH /users/me`; the appearance panel writes the
 * theme preference locally (no server round-trip).
 */

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ShieldIcon } from "@/components/icons";
import { ThemeSwitcher } from "@/components/shell";
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  Input,
  Modal,
  PageHeader,
  formatDate,
} from "@/components/ui";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useToast } from "@/lib/toast";

export default function SettingsPage() {
  const router = useRouter();
  const toast = useToast();
  const { user, refreshUser, logout } = useAuth();

  const [fullName, setFullName] = useState(user?.full_name || "");
  const [password, setPassword] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [confirmDeactivate, setConfirmDeactivate] = useState(false);

  // The auth bootstrap resolves the user asynchronously, so adopt the profile
  // name once it arrives (and refresh it after a successful save).
  useEffect(() => {
    setFullName(user?.full_name || "");
  }, [user?.full_name]);

  async function saveProfile(event) {
    event.preventDefault();
    setError("");
    setSaving(true);
    try {
      const payload = {};
      if (fullName !== (user?.full_name || "")) payload.full_name = fullName;
      if (password) payload.password = password;
      if (Object.keys(payload).length === 0) {
        toast.info("Nothing to update.");
        return;
      }
      await api.updateProfile(payload);
      await refreshUser();
      setPassword("");
      toast.success("Profile updated.");
    } catch (err) {
      setError(err.message);
      toast.error(err.message);
    } finally {
      setSaving(false);
    }
  }

  async function deactivate() {
    try {
      await api.deactivateAccount();
      toast.success("Account deactivated.");
      await logout();
      router.replace("/login");
    } catch (err) {
      toast.error(err.message);
    }
  }

  const initial = (user?.full_name || user?.email || "?").trim().charAt(0).toUpperCase();

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Settings"
        description="Manage your profile, appearance and account."
      />

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Profile" description="Your display name and password." />
          <form onSubmit={saveProfile} className="flex flex-col gap-4 p-5">
            <div className="flex items-center gap-3">
              <span className="flex h-11 w-11 items-center justify-center rounded-full bg-brand-600 text-base font-semibold text-white">
                {initial}
              </span>
              <div className="flex flex-col">
                <span className="text-sm font-medium">{user?.full_name || "Unnamed user"}</span>
                <span className="flex items-center gap-2 text-sm text-slate-500">
                  {user?.email}
                  <Badge tone="brand">{user?.role}</Badge>
                </span>
              </div>
            </div>
            <Input
              label="Full name"
              value={fullName}
              onChange={(event) => setFullName(event.target.value)}
              placeholder="Ada Lovelace"
            />
            <Input
              label="New password"
              type="password"
              autoComplete="new-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Leave blank to keep the current password"
              hint="At least 8 characters, with letters and numbers."
            />
            <Alert tone="error" onDismiss={() => setError("")}>
              {error}
            </Alert>
            <div className="flex justify-end">
              <Button type="submit" loading={saving}>
                Save changes
              </Button>
            </div>
          </form>
        </Card>

        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader
              title="Appearance"
              description="Choose how the console looks on this device."
            />
            <div className="flex items-center justify-between gap-4 p-5">
              <span className="text-sm text-slate-500">Theme</span>
              <ThemeSwitcher />
            </div>
          </Card>

          <Card>
            <CardHeader title="Account" description="Session and account lifecycle controls." />
            <div className="flex flex-col gap-4 p-5">
              <div className="flex items-center justify-between gap-4 text-sm">
                <span className="text-slate-500">Member since</span>
                <span>{formatDate(user?.created_at)}</span>
              </div>
              <div className="flex items-center justify-between gap-4 text-sm">
                <span className="text-slate-500">Last sign-in</span>
                <span>{formatDate(user?.last_login_at)}</span>
              </div>
              <div className="flex flex-wrap gap-2 border-t border-slate-100 pt-4 dark:border-slate-800">
                <Button
                  variant="secondary"
                  onClick={async () => {
                    await logout();
                    router.replace("/login");
                  }}
                >
                  Sign out
                </Button>
                <Button variant="danger" onClick={() => setConfirmDeactivate(true)}>
                  Deactivate account
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>

      <Card>
        <CardHeader
          title="About this deployment"
          description="Runs entirely on your infrastructure."
          action={
            <Badge tone="green">
              <ShieldIcon className="h-3.5 w-3.5" /> Self-hosted
            </Badge>
          }
        />
        <div className="grid gap-3 p-5 sm:grid-cols-2 lg:grid-cols-4">
          {[
            ["Frontend", "Responsive web console"],
            ["API", "Secure REST + streaming"],
            ["Workers", "Background processing"],
            ["Model", "Runs locally"],
          ].map(([label, value]) => (
            <div
              key={label}
              className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-800 dark:bg-slate-900/50"
            >
              <p className="text-xs font-medium tracking-wide text-slate-500 uppercase">
                {label}
              </p>
              <p className="mt-1 text-sm font-semibold">{value}</p>
            </div>
          ))}
        </div>
      </Card>

      <Modal
        open={confirmDeactivate}
        onClose={() => setConfirmDeactivate(false)}
        title="Deactivate account"
        description="Your data is retained, but access is revoked immediately."
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmDeactivate(false)}>
              Cancel
            </Button>
            <Button variant="danger" onClick={deactivate}>
              Deactivate
            </Button>
          </>
        }
      >
        <p className="text-sm text-slate-600 dark:text-slate-400">
          You will be signed out and unable to sign in again until an administrator
          reactivates the account.
        </p>
      </Modal>
    </div>
  );
}

