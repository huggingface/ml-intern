/**
 * Shared model-id constants used by session-create call sites and the model
 * picker.
 *
 * Keep in sync with MODEL_OPTIONS in components/Chat/ChatInput.tsx and
 * AVAILABLE_MODELS in backend/routes/agent.py.
 */

export const CLAUDE_OPUS_48_MODEL_PATH = 'anthropic/claude-opus-4.8:fal-ai';
export const GPT_55_MODEL_PATH = 'openai/gpt-5.5:fal-ai';
export const KIMI_K26_MODEL_PATH = 'moonshotai/Kimi-K2.6:novita';
export const MINIMAX_M27_MODEL_PATH = 'MiniMaxAI/MiniMax-M2.7:novita';
export const GLM_51_MODEL_PATH = 'zai-org/GLM-5.1:novita';
export const DEEPSEEK_V4_PRO_MODEL_PATH = 'deepseek-ai/DeepSeek-V4-Pro:novita';

export function isClaudePath(modelPath: string | undefined): boolean {
  return !!modelPath && modelPath.includes('anthropic');
}
