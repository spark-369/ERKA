"use client";

/**
 * Authentication context.
 *
 * Holds the current user, exposes login/register/logout, and revalidates the
 * session against `GET /auth/me` on mount (which also exercises token refresh).
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, tokenStore } from "./api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Restore a session from localStorage, then confirm it with the backend.
  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      const cachedUser = tokenStore.getUser();
      const hasToken = Boolean(tokenStore.getAccessToken());

      if (!hasToken) {
        if (!cancelled) {
          setUser(null);
          setLoading(false);
        }
        return;
      }

      if (cachedUser && !cancelled) setUser(cachedUser);

      try {
        const fresh = await api.me();
        if (!cancelled) {
          setUser(fresh);
          tokenStore.setUser(fresh);
        }
      } catch {
        if (!cancelled) {
          tokenStore.clear();
          setUser(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (email, password) => {
    const tokens = await api.login({ email, password });
    tokenStore.set(tokens);
    setUser(tokens.user);
    return tokens.user;
  }, []);

  const register = useCallback(async ({ email, password, full_name }) => {
    await api.register({ email, password, full_name });
    const tokens = await api.login({ email, password });
    tokenStore.set(tokens);
    setUser(tokens.user);
    return tokens.user;
  }, []);

  const logout = useCallback(async () => {
    await api.logout();
    tokenStore.clear();
    setUser(null);
  }, []);

  const refreshUser = useCallback(async () => {
    const fresh = await api.me();
    setUser(fresh);
    tokenStore.setUser(fresh);
    return fresh;
  }, []);

  const value = useMemo(
    () => ({
      user,
      loading,
      isAuthenticated: Boolean(user),
      login,
      register,
      logout,
      refreshUser,
    }),
    [user, loading, login, register, logout, refreshUser],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside an <AuthProvider>.");
  }
  return context;
}
