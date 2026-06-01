/**
 * Shared model-id constants used by session-create call sites and the model
 * picker.
 *
 * Keep in sync with MODEL_OPTIONS in components/Chat/ChatInput.tsx and
 * AVAILABLE_MODELS in backend/routes/agent.py.
 */

export const CLAUDE_MODEL_PATH = 'huggingface/anthropic/claude-opus-4.6:fal-ai';
export const GPT_55_MODEL_PATH = 'huggingface/openai/gpt-5.5:fal-ai';

export function isClaudePath(modelPath: string | undefined): boolean {
  return !!modelPath && modelPath.includes('anthropic');
}

export function isPremiumPath(modelPath: string | undefined): boolean {
  return modelPath === CLAUDE_MODEL_PATH || modelPath === GPT_55_MODEL_PATH;
}
