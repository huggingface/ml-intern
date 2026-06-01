/**
 * Shared model-id constants used by session-create call sites and the model
 * picker.
 *
 * Keep in sync with MODEL_OPTIONS in components/Chat/ChatInput.tsx and
 * AVAILABLE_MODELS in backend/routes/agent.py.
 */

export const CLAUDE_OPUS_46_MODEL_PATH = 'bedrock/us.anthropic.claude-opus-4-6-v1';
export const CLAUDE_OPUS_48_MODEL_PATH = 'bedrock/us.anthropic.claude-opus-4-8';
export const CLAUDE_MODEL_PATH = CLAUDE_OPUS_46_MODEL_PATH;
export const GPT_55_MODEL_PATH = 'openai/gpt-5.5';
export const USER_BILLED_CLAUDE_OPUS_46_MODEL_PATH = 'huggingface/anthropic/claude-opus-4.6:fal-ai';
export const USER_BILLED_CLAUDE_OPUS_48_MODEL_PATH = 'huggingface/anthropic/claude-opus-4.8:fal-ai';
export const USER_BILLED_CLAUDE_MODEL_PATH = USER_BILLED_CLAUDE_OPUS_46_MODEL_PATH;
export const USER_BILLED_GPT_55_MODEL_PATH = 'huggingface/openai/gpt-5.5:fal-ai';

const PREMIUM_MODEL_PATHS = new Set([
  CLAUDE_OPUS_46_MODEL_PATH,
  CLAUDE_OPUS_48_MODEL_PATH,
  GPT_55_MODEL_PATH,
  USER_BILLED_CLAUDE_OPUS_46_MODEL_PATH,
  USER_BILLED_CLAUDE_OPUS_48_MODEL_PATH,
  USER_BILLED_GPT_55_MODEL_PATH,
]);

export function isClaudePath(modelPath: string | undefined): boolean {
  return !!modelPath && modelPath.includes('anthropic');
}

export function isPremiumPath(modelPath: string | undefined): boolean {
  return !!modelPath && PREMIUM_MODEL_PATHS.has(modelPath);
}
