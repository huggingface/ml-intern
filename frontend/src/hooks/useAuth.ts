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

  useEffect(() => {
    let cancelled = false;

    async function checkAuth() {
      try {
        // Check if user is already authenticated (cookie-based)
        const response = await fetch('/auth/me', { credentials: 'include' });
        if (response.ok) {
          const data = await response.json();
          if (!cancelled && data.authenticated) {
            setUser({
              authenticated: true,
              username: data.username,
              name: data.name,
              picture: data.picture,
            });
            logger.log('Authenticated as', data.username);
            return;
          }
        }

        // Not authenticated — check if auth is enabled
        const statusRes = await fetch('/auth/status', { credentials: 'include' });
        const statusData = await statusRes.json();
        if (!statusData.auth_enabled) {
          // Dev mode — no OAuth configured
          if (!cancelled) setUser({ authenticated: true, username: 'dev' });
          return;
        }

        // Auth enabled but not logged in — welcome screen will handle it
        if (!cancelled) setUser(null);
      } catch {
        // Backend unreachable — assume dev mode
        if (!cancelled) setUser({ authenticated: true, username: 'dev' });
      }
    }

    checkAuth();
    return () => { cancelled = true; };
  }, [setUser]);
}
