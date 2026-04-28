import { useState, useCallback, useEffect, useRef, KeyboardEvent } from 'react';
import { Box, TextField, IconButton, CircularProgress, Typography, Menu, MenuItem, ListItemIcon, ListItemText, Chip, Divider, Button } from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import StopIcon from '@mui/icons-material/Stop';
import MemoryIcon from '@mui/icons-material/Memory';
import { apiFetch } from '@/utils/api';
import { useUserQuota } from '@/hooks/useUserQuota';
import ClaudeCapDialog from '@/components/ClaudeCapDialog';
import JobsUpgradeDialog from '@/components/JobsUpgradeDialog';
import { useAgentStore } from '@/store/agentStore';
import { CLAUDE_MODEL_PATH, FIRST_FREE_MODEL_PATH, isClaudePath } from '@/utils/model';

// Model configuration
interface ModelOption {
  id: string;
  name: string;
  description: string;
  modelPath: string;
  avatarUrl: string;
  provider?: string;
  recommended?: boolean;
}

interface BackendModel {
  id: string;
  label: string;
  provider: string;
  tier?: string;
  recommended?: boolean;
}

const getHfAvatarUrl = (modelId: string) => {
  const org = modelId.split('/')[0];
  return `https://huggingface.co/api/avatars/${org}`;
};

const DEFAULT_MODEL_OPTIONS: ModelOption[] = [
  {
    id: 'moonshotai/Kimi-K2.6',
    name: 'Kimi K2.6',
    description: 'Novita',
    modelPath: 'moonshotai/Kimi-K2.6',
    avatarUrl: getHfAvatarUrl('moonshotai/Kimi-K2.6'),
    provider: 'huggingface',
    recommended: true,
  },
  {
    id: 'bedrock/us.anthropic.claude-opus-4-6-v1',
    name: 'Claude Opus 4.6',
    description: 'Anthropic',
    modelPath: CLAUDE_MODEL_PATH,
    avatarUrl: 'https://huggingface.co/api/avatars/Anthropic',
    provider: 'anthropic',
    recommended: true,
  },
  {
    id: 'MiniMaxAI/MiniMax-M2.7',
    name: 'MiniMax M2.7',
    description: 'Novita',
    modelPath: 'MiniMaxAI/MiniMax-M2.7',
    avatarUrl: getHfAvatarUrl('MiniMaxAI/MiniMax-M2.7'),
    provider: 'huggingface',
  },
  {
    id: 'zai-org/GLM-5.1',
    name: 'GLM 5.1',
    description: 'Together',
    modelPath: 'zai-org/GLM-5.1',
    avatarUrl: getHfAvatarUrl('zai-org/GLM-5.1'),
    provider: 'huggingface',
  },
];

const providerDescription = (model: BackendModel): string => {
  if (model.provider === 'local') return 'Local';
  if (model.provider === 'anthropic') return 'Anthropic';
  return model.provider === 'huggingface' ? 'Hugging Face' : model.provider;
};

const modelOptionFromBackend = (model: BackendModel): ModelOption => ({
  id: model.id,
  name: model.label,
  description: providerDescription(model),
  modelPath: model.id,
  avatarUrl: model.provider === 'local' ? '' : getHfAvatarUrl(model.id),
  provider: model.provider,
  recommended: model.recommended,
});

const LOCAL_MODEL_PREFIXES = ['ollama/', 'vllm/', 'llamacpp/', 'local://'];

const findModelByPath = (options: ModelOption[], path: string): ModelOption | undefined => {
  return options.find(m => m.modelPath === path || m.id === path);
};

const isLocalModelPath = (path: string) => LOCAL_MODEL_PREFIXES.some(prefix => path.startsWith(prefix));

const labelFromLocalPath = (path: string) => {
  if (path.startsWith('local://')) return path.slice('local://'.length);
  const prefix = LOCAL_MODEL_PREFIXES.find(p => path.startsWith(p));
  return prefix ? path.slice(prefix.length) : path;
};

const customLocalModelOption = (path: string): ModelOption => ({
  id: path,
  name: labelFromLocalPath(path),
  description: 'Custom local path',
  modelPath: path,
  avatarUrl: '',
  provider: 'local',
});

interface ChatInputProps {
  sessionId?: string;
  onSend: (text: string) => void;
  onStop?: () => void;
  onDeclineBlockedJobs?: () => Promise<boolean>;
  onContinueBlockedJobsWithNamespace?: (namespace: string) => Promise<boolean>;
  isProcessing?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

const isClaudeModel = (m: ModelOption) => isClaudePath(m.modelPath);
const isLocalModel = (m: ModelOption) => m.provider === 'local' || isLocalModelPath(m.modelPath);

function ModelAvatar({ model, size }: { model: ModelOption; size: number }) {
  if (isLocalModel(model)) {
    return (
      <Box
        sx={{
          width: size,
          height: size,
          borderRadius: '4px',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'rgba(255,255,255,0.08)',
          color: 'var(--text)',
        }}
      >
        <MemoryIcon sx={{ fontSize: Math.max(14, size - 6) }} />
      </Box>
    );
  }
  return (
    <img
      src={model.avatarUrl}
      alt={model.name}
      style={{ height: size, width: size, objectFit: 'contain', borderRadius: '4px' }}
    />
  );
}

export default function ChatInput({ sessionId, onSend, onStop, onDeclineBlockedJobs, onContinueBlockedJobsWithNamespace, isProcessing = false, disabled = false, placeholder = 'Ask anything...' }: ChatInputProps) {
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>(DEFAULT_MODEL_OPTIONS);
  const [selectedModelId, setSelectedModelId] = useState<string>(DEFAULT_MODEL_OPTIONS[0].id);
  const [modelAnchorEl, setModelAnchorEl] = useState<null | HTMLElement>(null);
  const [customModelPath, setCustomModelPath] = useState('');
  const [customModelError, setCustomModelError] = useState('');
  const { quota, refresh: refreshQuota } = useUserQuota();
  // The daily-cap dialog is triggered from two places: (a) a 429 returned
  // from the chat transport when the user tries to send on Opus over cap —
  // surfaced via the agent-store flag — and (b) nothing else right now
  // (switching models is free). Keeping the open state in the store means
  // the hook layer can flip it without threading props through.
  const claudeQuotaExhausted = useAgentStore((s) => s.claudeQuotaExhausted);
  const setClaudeQuotaExhausted = useAgentStore((s) => s.setClaudeQuotaExhausted);
  const jobsUpgradeRequired = useAgentStore((s) => s.jobsUpgradeRequired);
  const setJobsUpgradeRequired = useAgentStore((s) => s.setJobsUpgradeRequired);
  const lastSentRef = useRef<string>('');

  useEffect(() => {
    let cancelled = false;
    apiFetch('/api/config/model')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled || !Array.isArray(data?.available)) return;
        const nextOptions = data.available.map(modelOptionFromBackend);
        if (nextOptions.length === 0) return;
        setModelOptions(nextOptions);
        const current = typeof data.current === 'string'
          ? findModelByPath(nextOptions, data.current)
          : undefined;
        setSelectedModelId((prev) => (
          current?.id ?? (nextOptions.some((model: ModelOption) => model.id === prev) ? prev : nextOptions[0].id)
        ));
      })
      .catch(() => { /* keep bundled defaults */ });
    return () => { cancelled = true; };
  }, []);

  // Model is per-session: fetch this tab's current model every time the
  // session changes. Other tabs keep their own selections independently.
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    apiFetch(`/api/session/${sessionId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled) return;
        if (data?.model) {
          let model = findModelByPath(modelOptions, data.model);
          if (!model && isLocalModelPath(data.model)) {
            model = customLocalModelOption(data.model);
            setModelOptions((prev) => (
              findModelByPath(prev, data.model) ? prev : [...prev, model as ModelOption]
            ));
          }
          if (model) setSelectedModelId(model.id);
        }
      })
      .catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, [sessionId, modelOptions]);

  const selectedModel = modelOptions.find(m => m.id === selectedModelId) || modelOptions[0];
  const customLocalEnabled = modelOptions.some(isLocalModel);
  const firstFreeModel = () => modelOptions.find(m => !isClaudeModel(m)) ?? modelOptions[0];

  // Auto-focus the textarea when the session becomes ready
  useEffect(() => {
    if (!disabled && !isProcessing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled, isProcessing]);

  const handleSend = useCallback(() => {
    if (input.trim() && !disabled) {
      lastSentRef.current = input;
      onSend(input);
      setInput('');
    }
  }, [input, disabled, onSend]);

  // When the chat transport reports a Claude-quota 429, restore the typed
  // text so the user doesn't lose their message.
  useEffect(() => {
    if (claudeQuotaExhausted && lastSentRef.current) {
      setInput(lastSentRef.current);
    }
  }, [claudeQuotaExhausted]);

  // Refresh the quota display whenever the session changes (user might
  // have started another tab that spent quota).
  useEffect(() => {
    if (sessionId) refreshQuota();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleModelClick = (event: React.MouseEvent<HTMLElement>) => {
    setModelAnchorEl(event.currentTarget);
  };

  const handleModelClose = () => {
    setModelAnchorEl(null);
  };

  const handleSelectModel = async (model: ModelOption) => {
    handleModelClose();
    if (!sessionId) return;
    try {
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: 'POST',
        body: JSON.stringify({ model: model.modelPath }),
      });
      if (res.ok) setSelectedModelId(model.id);
    } catch { /* ignore */ }
  };

  const handleUseCustomModel = async () => {
    const path = customModelPath.trim();
    if (!path || !sessionId) return;
    setCustomModelError('');
    const model = customLocalModelOption(path);
    try {
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: 'POST',
        body: JSON.stringify({ model: model.modelPath }),
      });
      if (!res.ok) {
        let message = `Unable to use ${path}`;
        try {
          const data = await res.json();
          if (typeof data?.detail === 'string') {
            message = data.detail;
          } else if (typeof data?.detail?.message === 'string') {
            message = data.detail.message;
          }
        } catch { /* keep fallback */ }
        setCustomModelError(message);
        return;
      }
      setModelOptions((prev) => (
        findModelByPath(prev, model.modelPath) ? prev : [...prev, model]
      ));
      setSelectedModelId(model.id);
      setCustomModelPath('');
      setCustomModelError('');
      handleModelClose();
    } catch {
      setCustomModelError('Unable to switch to that local model.');
    }
  };

  // Dialog close: just clear the flag. The typed text is already restored.
  const handleCapDialogClose = useCallback(() => {
    setClaudeQuotaExhausted(false);
  }, [setClaudeQuotaExhausted]);

  // "Use a free model" — switch the current session to Kimi (or the first
  // non-Anthropic option) and auto-retry the send that tripped the cap.
  const handleUseFreeModel = useCallback(async () => {
    setClaudeQuotaExhausted(false);
    if (!sessionId) return;
    const free = modelOptions.find(m => m.modelPath === FIRST_FREE_MODEL_PATH)
      ?? firstFreeModel();
    try {
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: 'POST',
        body: JSON.stringify({ model: free.modelPath }),
      });
      if (res.ok) {
        setSelectedModelId(free.id);
        const retryText = lastSentRef.current;
        if (retryText) {
          onSend(retryText);
          setInput('');
          lastSentRef.current = '';
        }
      }
    } catch { /* ignore */ }
  }, [sessionId, onSend, setClaudeQuotaExhausted, modelOptions]);

  const handleClaudeUpgradeClick = useCallback(async () => {
    if (!sessionId) return;
    try {
      await apiFetch(`/api/pro-click/${sessionId}`, {
        method: 'POST',
        body: JSON.stringify({ source: 'claude_cap_dialog', target: 'pro_pricing' }),
      });
    } catch {
      /* tracking is best-effort */
    }
  }, [sessionId]);

  const handleJobsUpgradeClose = useCallback(() => {
    setJobsUpgradeRequired(null);
  }, [setJobsUpgradeRequired]);

  const handleJobsUpgradeClick = useCallback(async () => {
    if (!sessionId || !jobsUpgradeRequired) return;
    try {
      await apiFetch(`/api/pro-click/${sessionId}`, {
        method: 'POST',
        body: JSON.stringify({ source: 'hf_jobs_upgrade_dialog', target: 'pro_pricing' }),
      });
    } catch {
      /* tracking is best-effort */
    }
  }, [sessionId, jobsUpgradeRequired]);

  const handleDeclineBlockedJobs = useCallback(async () => {
    if (!onDeclineBlockedJobs) return;
    await onDeclineBlockedJobs();
  }, [onDeclineBlockedJobs]);

  const handleContinueBlockedJobsWithNamespace = useCallback(async (namespace: string) => {
    if (!onContinueBlockedJobsWithNamespace) return;
    await onContinueBlockedJobsWithNamespace(namespace);
  }, [onContinueBlockedJobsWithNamespace]);

  // Hide the chip until the user has actually burned quota — an unused
  // Opus session shouldn't populate a counter.
  const claudeChip = (() => {
    if (!quota || quota.claudeUsedToday === 0) return null;
    if (quota.plan === 'free') {
      return quota.claudeRemaining > 0 ? 'Free today' : 'Pro only';
    }
    return `${quota.claudeUsedToday}/${quota.claudeDailyCap} today`;
  })();

  return (
    <Box
      sx={{
        pb: { xs: 2, md: 4 },
        pt: { xs: 1, md: 2 },
        position: 'relative',
        zIndex: 10,
      }}
    >
      <Box sx={{ maxWidth: '880px', mx: 'auto', width: '100%', px: { xs: 0, sm: 1, md: 2 } }}>
        <Box
          className="composer"
          sx={{
            display: 'flex',
            gap: '10px',
            alignItems: 'flex-start',
            bgcolor: 'var(--composer-bg)',
            borderRadius: 'var(--radius-md)',
            p: '12px',
            border: '1px solid var(--border)',
            transition: 'box-shadow 0.2s ease, border-color 0.2s ease',
            '&:focus-within': {
                borderColor: 'var(--accent-yellow)',
                boxShadow: 'var(--focus)',
            }
          }}
        >
          <TextField
            fullWidth
            multiline
            maxRows={6}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled || isProcessing}
            variant="standard"
            inputRef={inputRef}
            InputProps={{
                disableUnderline: true,
                sx: {
                    color: 'var(--text)',
                    fontSize: '15px',
                    fontFamily: 'inherit',
                    padding: 0,
                    lineHeight: 1.5,
                    minHeight: { xs: '44px', md: '56px' },
                    alignItems: 'flex-start',
                }
            }}
            sx={{
                flex: 1,
                '& .MuiInputBase-root': {
                    p: 0,
                    backgroundColor: 'transparent',
                },
                '& textarea': {
                    resize: 'none',
                    padding: '0 !important',
                }
            }}
          />
          {isProcessing ? (
            <IconButton
              onClick={onStop}
              sx={{
                mt: 1,
                p: 1.5,
                borderRadius: '10px',
                color: 'var(--muted-text)',
                transition: 'all 0.2s',
                position: 'relative',
                '&:hover': {
                  bgcolor: 'var(--hover-bg)',
                  color: 'var(--accent-red)',
                },
              }}
            >
              <Box sx={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <CircularProgress size={28} thickness={3} sx={{ color: 'inherit', position: 'absolute' }} />
                <StopIcon sx={{ fontSize: 16 }} />
              </Box>
            </IconButton>
          ) : (
            <IconButton
              onClick={handleSend}
              disabled={disabled || !input.trim()}
              sx={{
                mt: 1,
                p: 1,
                borderRadius: '10px',
                color: 'var(--muted-text)',
                transition: 'all 0.2s',
                '&:hover': {
                  color: 'var(--accent-yellow)',
                  bgcolor: 'var(--hover-bg)',
                },
                '&.Mui-disabled': {
                  opacity: 0.3,
                },
              }}
            >
              <ArrowUpwardIcon fontSize="small" />
            </IconButton>
          )}
        </Box>

        {/* Powered By Badge */}
        <Box
          onClick={handleModelClick}
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mt: 1.5,
            gap: 0.8,
            opacity: 0.6,
            cursor: 'pointer',
            transition: 'opacity 0.2s',
            '&:hover': {
              opacity: 1
            }
          }}
        >
          <Typography variant="caption" sx={{ fontSize: '10px', color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 500 }}>
            powered by
          </Typography>
          <ModelAvatar model={selectedModel} size={14} />
          <Typography variant="caption" sx={{ fontSize: '10px', color: 'var(--text)', fontWeight: 600, letterSpacing: '0.02em' }}>
            {selectedModel.name}
          </Typography>
          <ArrowDropDownIcon sx={{ fontSize: '14px', color: 'var(--muted-text)' }} />
        </Box>

        {/* Model Selection Menu */}
        <Menu
          anchorEl={modelAnchorEl}
          open={Boolean(modelAnchorEl)}
          onClose={handleModelClose}
          anchorOrigin={{
            vertical: 'top',
            horizontal: 'center',
          }}
          transformOrigin={{
            vertical: 'bottom',
            horizontal: 'center',
          }}
          slotProps={{
            paper: {
              sx: {
                bgcolor: 'var(--panel)',
                border: '1px solid var(--divider)',
                mb: 1,
                maxHeight: '400px',
              }
            }
          }}
        >
          {modelOptions.map((model) => (
            <MenuItem
              key={model.id}
              onClick={() => handleSelectModel(model)}
              selected={selectedModelId === model.id}
              sx={{
                py: 1.5,
                '&.Mui-selected': {
                  bgcolor: 'rgba(255,255,255,0.05)',
                }
              }}
            >
              <ListItemIcon>
                <ModelAvatar model={model} size={24} />
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {model.name}
                    {model.recommended && (
                      <Chip
                        label="Recommended"
                        size="small"
                        sx={{
                          height: '18px',
                          fontSize: '10px',
                          bgcolor: 'var(--accent-yellow)',
                          color: '#000',
                          fontWeight: 600,
                        }}
                      />
                    )}
                    {isClaudeModel(model) && claudeChip && (
                      <Chip
                        label={claudeChip}
                        size="small"
                        sx={{
                          height: '18px',
                          fontSize: '10px',
                          bgcolor: 'rgba(255,255,255,0.08)',
                          color: 'var(--muted-text)',
                          fontWeight: 600,
                        }}
                      />
                    )}
                  </Box>
                }
                secondary={model.description}
                secondaryTypographyProps={{
                  sx: { fontSize: '12px', color: 'var(--muted-text)' }
                }}
              />
            </MenuItem>
          ))}
          {customLocalEnabled && (
            <>
              <Divider sx={{ borderColor: 'var(--divider)', my: 0.5 }} />
              <Box
                sx={{ px: 1.5, py: 1.25, width: 360, maxWidth: 'calc(100vw - 32px)' }}
                onClick={(e) => e.stopPropagation()}
                onKeyDown={(e) => e.stopPropagation()}
              >
                <Typography
                  variant="caption"
                  sx={{
                    display: 'block',
                    mb: 0.75,
                    color: 'var(--muted-text)',
                    fontSize: '11px',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                  }}
                >
                  Custom local model path
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                  <TextField
                    size="small"
                    value={customModelPath}
                    onChange={(e) => {
                      setCustomModelPath(e.target.value);
                      setCustomModelError('');
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        handleUseCustomModel();
                      }
                    }}
                    placeholder="ollama/qwen2.5-coder"
                    fullWidth
                    variant="outlined"
                    error={!!customModelError}
                    helperText={customModelError || ' '}
                    FormHelperTextProps={{
                      sx: {
                        mx: 0,
                        color: customModelError ? 'var(--accent-red)' : 'transparent',
                        fontSize: '11px',
                      },
                    }}
                    sx={{
                      '& .MuiInputBase-root': {
                        color: 'var(--text)',
                        bgcolor: 'rgba(255,255,255,0.04)',
                      },
                      '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'var(--divider)',
                      },
                    }}
                  />
                  <Button
                    variant="contained"
                    size="small"
                    disabled={!customModelPath.trim()}
                    onClick={handleUseCustomModel}
                    sx={{
                      minWidth: 56,
                      bgcolor: 'var(--accent-yellow)',
                      color: '#000',
                      fontWeight: 700,
                      '&:hover': { bgcolor: 'var(--accent-yellow)' },
                    }}
                  >
                    Use
                  </Button>
                </Box>
              </Box>
            </>
          )}
        </Menu>

        <ClaudeCapDialog
          open={claudeQuotaExhausted}
          plan={quota?.plan ?? 'free'}
          cap={quota?.claudeDailyCap ?? 1}
          onClose={handleCapDialogClose}
          onUseFreeModel={handleUseFreeModel}
          onUpgrade={handleClaudeUpgradeClick}
        />
        <JobsUpgradeDialog
          open={!!jobsUpgradeRequired}
          mode={jobsUpgradeRequired?.mode || 'upgrade'}
          message={jobsUpgradeRequired?.message || ''}
          eligibleNamespaces={jobsUpgradeRequired?.eligibleNamespaces || []}
          onClose={handleJobsUpgradeClose}
          onUpgrade={handleJobsUpgradeClick}
          onDecline={handleDeclineBlockedJobs}
          onContinueWithNamespace={handleContinueBlockedJobsWithNamespace}
        />
      </Box>
    </Box>
  );
}
