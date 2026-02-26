/**
 * Lightweight localStorage persistence for UIMessage arrays,
 * keyed by session ID.
 *
 * Uses the same storage namespace (`hf-agent-messages`) that the
 * old Zustand-based store used, so existing data is compatible.
 */
import type { UIMessage } from 'ai';
import { logger } from '@/utils/logger';

const STORAGE_KEY = 'hf-agent-messages';
const MAX_SESSIONS = 50;

type MessagesMap = Record<string, UIMessage[]>;

function readAll(): MessagesMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    // Legacy format was { messagesBySession: {...} }
    if (parsed.messagesBySession) return parsed.messagesBySession;
    // New flat format
    if (typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
    return {};
  } catch {
    return {};
  }
}

function writeAll(map: MessagesMap): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch (e) {
    logger.warn('Failed to persist messages:', e);
  }
}

export function loadMessages(sessionId: string): UIMessage[] {
  const map = readAll();
  return map[sessionId] ?? [];
}

export function saveMessages(sessionId: string, messages: UIMessage[]): void {
  const map = readAll();
  map[sessionId] = messages;

  // Evict oldest sessions if we exceed the cap
  const keys = Object.keys(map);
  if (keys.length > MAX_SESSIONS) {
    const toRemove = keys.slice(0, keys.length - MAX_SESSIONS);
    for (const k of toRemove) delete map[k];
  }

  writeAll(map);
}

export function deleteMessages(sessionId: string): void {
  const map = readAll();
  delete map[sessionId];
  writeAll(map);
}
