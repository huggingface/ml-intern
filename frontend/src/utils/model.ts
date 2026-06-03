/**
 * Shared model-id constants used by session-create call sites and the model
 * picker.
 *
 * Keep in sync with DEFAULT_MODEL_OPTIONS in components/Chat/ChatInput.tsx and
 * AVAILABLE_MODELS in backend/routes/agent.py.
 */

export const KIMI_K26_MODEL_PATH = 'moonshotai/Kimi-K2.6';
export const CLAUDE_OPUS_48_MODEL_PATH = 'anthropic/claude-opus-4.8:fal-ai';
export const GPT_55_MODEL_PATH = 'openai/gpt-5.5:fal-ai';

const PRO_ONLY_MODEL_PATHS = new Set([
  CLAUDE_OPUS_48_MODEL_PATH,
  GPT_55_MODEL_PATH,
]);

export function isProOnlyPath(modelPath: string | undefined): boolean {
  return !!modelPath && PRO_ONLY_MODEL_PATHS.has(modelPath);
}
