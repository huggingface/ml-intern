/**
 * Authentication hook — simple server-side OAuth.
 *
 * - Hors iframe: /auth/login redirect (cookies work fine)
 * - Dans iframe: show "Open in full page" link
 *
 * Token is stored via HttpOnly cookie by the backend.
 * In dev mode (no OAUTH_CLIENT_ID), auth is bypassed.
 */

import { useEffect } from 'react';
import { useAgentStore } from '@/store/agentStore';
import { logger } from '@/utils/logger';

/** Check if we're running inside an iframe. */
export function isInIframe(): boolean {
  try {
    return window.top !== window.self;
  } catch {
    return true; // SecurityError = cross-origin iframe
  }
}

/** Redirect to the server-side OAuth login. */
export function triggerLogin(): void {
  window.location.href = '/auth/login';
}

/**
 * Hook: on mount, check if user is authenticated.
 * Sets user in the agent store.
 */
export function useAuth() {
  const setUser = useAgentStore((s) => s.setUser);
  const setAuthChecking = useAgentStore((s) => s.setAuthChecking);

  useEffect(() => {
    let cancelled = false;

    async function hydrateCurrentUser(): Promise<boolean> {
      const response = await fetch('/auth/me', { credentials: 'include' });
      if (!response.ok) return false;

      const data = await response.json();
      if (!data.authenticated) return false;
      if (!cancelled) {
        setUser({
          authenticated: true,
          username: data.username,
          name: data.name,
          picture: data.picture,
          plan: data.plan === 'pro' ? 'pro' : 'free',
        });
        logger.log('Authenticated as', data.username);
      }
      return true;
    }

    async function checkAuth() {
      try {
        // Check the cheap instance status first. Local development bypasses
        // OAuth, so mark it authenticated immediately instead of flashing a
        // misleading "Sign in" gate while /auth/me validates HF_TOKEN.
        const statusRes = await fetch('/auth/status', { credentials: 'include' });
        const statusData = statusRes.ok ? await statusRes.json() : null;
        if (statusData && !statusData.auth_enabled) {
          if (!cancelled) {
            setUser({ authenticated: true, username: 'dev', plan: 'pro' });
            setAuthChecking(false);
          }
          // Resolve the real HF identity in the background when HF_TOKEN is
          // configured. Failure is harmless because dev auth is already valid.
          try {
            await hydrateCurrentUser();
          } catch {
            // Keep the local dev identity.
          }
          return;
        }

        // Hosted/OAuth mode: restore the cookie-backed user when available.
        if (await hydrateCurrentUser()) return;

        // Auth enabled but not logged in — welcome screen will handle it
        if (!cancelled) setUser(null);
      } catch {
        // Backend unreachable — assume dev mode
        if (!cancelled) setUser({ authenticated: true, username: 'dev', plan: 'pro' });
      } finally {
        if (!cancelled) setAuthChecking(false);
      }
    }

    checkAuth();
    return () => { cancelled = true; };
  }, [setAuthChecking, setUser]);
}
