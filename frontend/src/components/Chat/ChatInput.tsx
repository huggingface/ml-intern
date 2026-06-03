import { useState, useCallback, useEffect, useRef, KeyboardEvent } from 'react';
import {
  Alert,
  Box,
  TextField,
  IconButton,
  CircularProgress,
  Typography,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Chip,
  LinearProgress,
  Snackbar,
  Tooltip,
} from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import StopIcon from '@mui/icons-material/Stop';
import AddIcon from '@mui/icons-material/Add';
import { apiFetch, apiUpload } from '@/utils/api';
import type { PlanTier, UserQuota } from '@/hooks/useUserQuota';
import JobsUpgradeDialog from '@/components/JobsUpgradeDialog';
import { useAgentStore } from '@/store/agentStore';
import { useSessionStore } from '@/store/sessionStore';
import {
  CLAUDE_MODEL_PATH,
  CLAUDE_OPUS_48_MODEL_PATH,
  GPT_55_MODEL_PATH,
  isClaudePath,
  isPremiumPath,
  isProOnlyPath,
} from '@/utils/model';

// Model configuration
interface ModelOption {
  id: string;
  name: string;
  description: string;
  modelPath: string;
  avatarUrl: string;
  recommended?: boolean;
  minimumPlan?: 'free' | 'pro';
}

const getHfAvatarUrl = (modelId: string) => {
  const org = modelId.split('/')[0];
  return `https://huggingface.co/api/avatars/${org}`;
};

const DEFAULT_MODEL_OPTIONS: ModelOption[] = [
  {
    id: 'claude-sonnet-4-6',
    name: 'Claude Sonnet 4.6',
    description: 'Hugging Face',
    modelPath: CLAUDE_MODEL_PATH,
    avatarUrl: getHfAvatarUrl(CLAUDE_MODEL_PATH),
    recommended: true,
  },
  {
    id: 'claude-opus-4-8',
    name: 'Claude Opus 4.8',
    description: 'Hugging Face',
    modelPath: CLAUDE_OPUS_48_MODEL_PATH,
    avatarUrl: getHfAvatarUrl(CLAUDE_OPUS_48_MODEL_PATH),
    minimumPlan: 'pro',
  },
  {
    id: 'gpt-5.5',
    name: 'GPT-5.5',
    description: 'Hugging Face',
    modelPath: GPT_55_MODEL_PATH,
    avatarUrl: getHfAvatarUrl(GPT_55_MODEL_PATH),
    minimumPlan: 'pro',
  },
  {
    id: 'kimi-k2.6',
    name: 'Kimi K2.6',
    description: 'Hugging Face',
    modelPath: 'moonshotai/Kimi-K2.6',
    avatarUrl: getHfAvatarUrl('moonshotai/Kimi-K2.6'),
  },
  {
    id: 'minimax-m2.7',
    name: 'MiniMax M2.7',
    description: 'Hugging Face',
    modelPath: 'MiniMaxAI/MiniMax-M2.7',
    avatarUrl: getHfAvatarUrl('MiniMaxAI/MiniMax-M2.7'),
  },
  {
    id: 'glm-5.1',
    name: 'GLM 5.1',
    description: 'Hugging Face',
    modelPath: 'zai-org/GLM-5.1',
    avatarUrl: getHfAvatarUrl('zai-org/GLM-5.1'),
  },
  {
    id: 'deepseek-v4-pro',
    name: 'DeepSeek V4 Pro',
    description: 'Hugging Face',
    modelPath: 'deepseek-ai/DeepSeek-V4-Pro:deepinfra',
    avatarUrl: getHfAvatarUrl('deepseek-ai/DeepSeek-V4-Pro'),
  },
];

const normalizeModelPath = (path: string | undefined) => (
  (path ?? '')
    .toLowerCase()
    .replace(/^huggingface\//, '')
    .replace(/claude-opus-4\.(\d)/g, 'claude-opus-4-$1')
    .replace(/claude-sonnet-4\.(\d)/g, 'claude-sonnet-4-$1')
);

const findModelByPath = (path: string, options: ModelOption[]): ModelOption | undefined => {
  const normalizedPath = normalizeModelPath(path);
  const matched = options.find((m) => {
    const normalizedModelPath = normalizeModelPath(m.modelPath);
    const normalizedId = normalizeModelPath(m.id);
    return (
      m.modelPath === path ||
      normalizedModelPath === normalizedPath ||
      normalizedPath.includes(normalizedId)
    );
  });
  if (matched) return matched;
  if (isClaudePath(path)) {
    const claude = options.find(isClaudeModel);
    if (claude) return claude;
  }
  return undefined;
};

const modelOptionId = (modelPath: string) => (
  normalizeModelPath(modelPath)
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
);

const modelOptionFromApi = (model: {
  id?: string;
  label?: string;
  provider?: string;
  recommended?: boolean;
  minimum_plan?: string;
}): ModelOption | null => {
  if (!model.id) return null;
  return {
    id: modelOptionId(model.id),
    name: model.label ?? model.id,
    description: 'Hugging Face',
    modelPath: model.id,
    avatarUrl: getHfAvatarUrl(model.id.replace(/^huggingface\//, '')),
    recommended: Boolean(model.recommended),
    minimumPlan: model.minimum_plan === 'pro' ? 'pro' : 'free',
  };
};

const readApiErrorMessage = async (res: Response, fallback: string): Promise<string> => {
  try {
    const data = await res.json();
    const detail = data?.detail;
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail.message === 'string') return detail.message;
    if (detail && typeof detail.error === 'string') return detail.error;
  } catch {
    /* ignore malformed error bodies */
  }
  return fallback;
};

interface ChatInputProps {
  sessionId?: string;
  initialModelPath?: string | null;
  onSend: (text: string) => void;
  onStop?: () => void;
  onDatasetUploaded?: () => Promise<boolean> | boolean;
  isProcessing?: boolean;
  disabled?: boolean;
  placeholder?: string;
  quota: UserQuota | null;
  refreshQuota: () => Promise<void> | void;
}

interface DatasetUploadResponse {
  session_id: string;
  repo_id: string;
  repo_type: 'dataset';
  private: true;
  upload_id: string;
  config_name: string;
  filename: string;
  path_in_repo: string;
  size_bytes: number;
  format: 'csv' | 'json' | 'jsonl';
  hub_url: string;
  load_dataset_snippet: string;
}

const MAX_DATASET_UPLOAD_BYTES = 100 * 1024 * 1024;
const DATASET_UPLOAD_ACCEPT = '.csv,.json,.jsonl';
const DATASET_UPLOAD_EXTENSIONS = new Set(['csv', 'json', 'jsonl']);

const isClaudeModel = (m: ModelOption) => isClaudePath(m.modelPath);
const isPremiumModel = (m: ModelOption) => isPremiumPath(m.modelPath);
const isProOnlyModel = (m: ModelOption) => (
  m.minimumPlan === 'pro' || isProOnlyPath(m.modelPath)
);
const isModelAllowedForPlan = (m: ModelOption, plan: PlanTier) => (
  plan === 'pro' || !isProOnlyModel(m)
);

const formatBytes = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const datasetRepoUrl = (repoId: string) => (
  `https://huggingface.co/datasets/${repoId.split('/').map(encodeURIComponent).join('/')}`
);

export default function ChatInput({ sessionId, initialModelPath, onSend, onStop, onDatasetUploaded, isProcessing = false, disabled = false, placeholder = 'Ask anything...', quota, refreshQuota }: ChatInputProps) {
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>(DEFAULT_MODEL_OPTIONS);
  const modelOptionsRef = useRef<ModelOption[]>(DEFAULT_MODEL_OPTIONS);
  const sessionIdRef = useRef<string | undefined>(sessionId);
  const [selectedModelId, setSelectedModelId] = useState<string>(
    () => findModelByPath(initialModelPath ?? '', DEFAULT_MODEL_OPTIONS)?.id ?? DEFAULT_MODEL_OPTIONS[0].id,
  );
  const [modelAnchorEl, setModelAnchorEl] = useState<null | HTMLElement>(null);
  const jobsUpgradeRequired = useAgentStore((s) => s.jobsUpgradeRequired);
  const setJobsUpgradeRequired = useAgentStore((s) => s.setJobsUpgradeRequired);
  const updateSessionModel = useSessionStore((s) => s.updateSessionModel);
  const [awaitingTopUp, setAwaitingTopUp] = useState(false);
  const [modelSwitchError, setModelSwitchError] = useState<string | null>(null);
  const [datasetUploadError, setDatasetUploadError] = useState<string | null>(null);
  const [datasetUploadSuccess, setDatasetUploadSuccess] = useState<string | null>(null);
  const [uploadedDatasets, setUploadedDatasets] = useState<DatasetUploadResponse[]>([]);
  const [isUploadingDataset, setIsUploadingDataset] = useState(false);
  const [datasetUploadProgress, setDatasetUploadProgress] = useState<number | null>(null);

  useEffect(() => {
    modelOptionsRef.current = modelOptions;
  }, [modelOptions]);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    let cancelled = false;
    apiFetch('/api/config/model')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled || !data?.available) return;
        const next = data.available
          .map(modelOptionFromApi)
          .filter((model: ModelOption | null): model is ModelOption => model !== null);
        if (!next.length) return;
        modelOptionsRef.current = next;
        setModelOptions(next);
        if (!sessionIdRef.current) {
          const current = data.current ? findModelByPath(data.current, next) : null;
          if (current) setSelectedModelId(current.id);
        }
      })
      .catch(() => { /* ignore */ });
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
          const model = findModelByPath(data.model, modelOptionsRef.current);
          if (model) setSelectedModelId(model.id);
          updateSessionModel(sessionId, data.model);
        }
      })
      .catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, [sessionId, updateSessionModel]);

  const plan = quota?.plan ?? 'free';
  const visibleModelOptions = modelOptions.filter((model) => (
    isModelAllowedForPlan(model, plan)
  ));
  const selectedModel = (
    visibleModelOptions.find(m => m.id === selectedModelId)
    || modelOptions.find(m => m.id === selectedModelId)
    || visibleModelOptions[0]
    || modelOptions[0]
  );

  // Auto-focus the textarea when the session becomes ready
  useEffect(() => {
    if (!disabled && !isProcessing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled, isProcessing]);

  const handleSend = useCallback(() => {
    if (input.trim() && !disabled && !isUploadingDataset) {
      onSend(input);
      setInput('');
    }
  }, [input, disabled, isUploadingDataset, onSend]);

  const handleDatasetUploadClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleDatasetFileChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = '';
      if (!file) return;

      if (!sessionId) {
        setDatasetUploadError('Start a session before uploading a dataset.');
        return;
      }

      const extension = file.name.split('.').pop()?.toLowerCase() || '';
      if (!DATASET_UPLOAD_EXTENSIONS.has(extension)) {
        setDatasetUploadError('Only CSV, JSON, and JSONL dataset files are supported.');
        return;
      }
      if (file.size > MAX_DATASET_UPLOAD_BYTES) {
        setDatasetUploadError(
          `Dataset files must be 100 MB or smaller. ${file.name} is ${formatBytes(file.size)}.`
        );
        return;
      }
      if (file.size === 0) {
        setDatasetUploadError('Uploaded dataset file is empty.');
        return;
      }

      const formData = new FormData();
      formData.append('file', file);
      setIsUploadingDataset(true);
      setDatasetUploadProgress(0);
      setDatasetUploadError(null);
      setDatasetUploadSuccess(null);
      try {
        const res = await apiUpload(`/api/session/${sessionId}/datasets`, formData, {
          onProgress: ({ percent }) => {
            setDatasetUploadProgress(percent !== null && percent < 100 ? percent : null);
          },
        });
        if (!res.ok) {
          setDatasetUploadError(await readApiErrorMessage(res, 'Dataset upload failed.'));
          return;
        }
        const payload = await res.json() as DatasetUploadResponse;
        setUploadedDatasets((previous) => [payload, ...previous]);
        setDatasetUploadSuccess(`Uploaded ${payload.filename} to ${payload.repo_id}`);
        await onDatasetUploaded?.();
      } catch (error) {
        setDatasetUploadError(
          error instanceof Error ? error.message : 'Dataset upload failed.'
        );
      } finally {
        setIsUploadingDataset(false);
        setDatasetUploadProgress(null);
      }
    },
    [sessionId, onDatasetUploaded],
  );

  useEffect(() => {
    if (!datasetUploadError) return;
    const timeout = window.setTimeout(() => setDatasetUploadError(null), 7000);
    return () => window.clearTimeout(timeout);
  }, [datasetUploadError]);

  useEffect(() => {
    if (!datasetUploadSuccess) return;
    const timeout = window.setTimeout(() => setDatasetUploadSuccess(null), 5000);
    return () => window.clearTimeout(timeout);
  }, [datasetUploadSuccess]);

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
    void refreshQuota();
  };

  const handleModelClose = () => {
    setModelAnchorEl(null);
  };

  const handleSelectModel = async (model: ModelOption) => {
    handleModelClose();
    if (!sessionId) return;
    if (!isModelAllowedForPlan(model, plan)) {
      setModelSwitchError('Claude Opus 4.8 and GPT-5.5 daily sessions require HF Pro.');
      return;
    }
    try {
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: 'POST',
        body: JSON.stringify({ model: model.modelPath }),
      });
      if (res.ok) {
        setSelectedModelId(model.id);
        updateSessionModel(sessionId, model.modelPath);
        setModelSwitchError(null);
        return;
      }
      setModelSwitchError(await readApiErrorMessage(res, 'Could not switch model.'));
    } catch (error) {
      setModelSwitchError(error instanceof Error ? error.message : 'Could not switch model.');
    }
  };

  const handleJobsUpgradeClose = useCallback(() => {
    setJobsUpgradeRequired(null);
    setAwaitingTopUp(false);
  }, [setJobsUpgradeRequired]);

  const handleJobsUpgradeClick = useCallback(async () => {
    setAwaitingTopUp(true);
    if (!sessionId || !jobsUpgradeRequired) return;
    try {
      await apiFetch(`/api/pro-click/${sessionId}`, {
        method: 'POST',
        body: JSON.stringify({ source: 'hf_jobs_billing_dialog', target: 'hf_billing' }),
      });
    } catch {
      /* tracking is best-effort */
    }
  }, [sessionId, jobsUpgradeRequired]);

  const handleJobsRetry = useCallback(() => {
    const namespace = jobsUpgradeRequired?.namespace;
    setJobsUpgradeRequired(null);
    setAwaitingTopUp(false);
    const msg = namespace
      ? `I just added credits to the \`${namespace}\` namespace. Please retry the previous job.`
      : "I just added credits. Please retry the previous job.";
    onSend(msg);
  }, [jobsUpgradeRequired, setJobsUpgradeRequired, onSend]);

  // Auto-retry when the user comes back to this tab after clicking "Add credits".
  // Browsers fire visibilitychange when the tab regains focus from a sibling tab.
  useEffect(() => {
    if (!awaitingTopUp || !jobsUpgradeRequired) return;
    const onVisible = () => {
      if (document.visibilityState === 'visible') {
        handleJobsRetry();
      }
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => document.removeEventListener('visibilitychange', onVisible);
  }, [awaitingTopUp, jobsUpgradeRequired, handleJobsRetry]);

  // Show the remaining subsidized premium-session allowance for today.
  const premiumChip = (() => {
    if (!quota) return null;
    const remaining = Math.max(0, quota.premiumRemaining);
    if (remaining === 0) {
      return quota.plan === 'pro' ? '0 left – using HF billing' : '0 left – enable billing';
    }
    return `${remaining} left today`;
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
            display: 'grid',
            gridTemplateColumns: 'auto 1fr auto',
            gridTemplateRows: 'auto auto',
            columnGap: '10px',
            rowGap: '4px',
            alignItems: 'end',
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
                gridColumn: '1 / -1',
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
          <input
            ref={fileInputRef}
            type="file"
            accept={DATASET_UPLOAD_ACCEPT}
            onChange={handleDatasetFileChange}
            style={{ display: 'none' }}
          />
          <Box sx={{ gridColumn: '1', gridRow: '2', display: 'flex' }}>
            <Tooltip title="Upload dataset">
              <span>
                <IconButton
                  onClick={handleDatasetUploadClick}
                  disabled={disabled || isProcessing || isUploadingDataset || !sessionId}
                  sx={{
                    p: 1,
                    borderRadius: '50%',
                    color: uploadedDatasets.length ? 'var(--accent-yellow)' : 'var(--muted-text)',
                    transition: 'all 0.2s',
                    '&:hover': {
                      color: 'var(--accent-yellow)',
                      bgcolor: 'var(--hover-bg)',
                    },
                    '&.Mui-disabled': {
                      opacity: 0.3,
                    },
                  }}
                  aria-label="Upload dataset"
                >
                  <AddIcon fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
          </Box>
          {isProcessing ? (
            <IconButton
              onClick={onStop}
              sx={{
                gridColumn: '3',
                gridRow: '2',
                justifySelf: 'end',
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
              disabled={disabled || isUploadingDataset || !input.trim()}
              sx={{
                gridColumn: '3',
                gridRow: '2',
                justifySelf: 'end',
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
        {isUploadingDataset && (
          <Box sx={{ mt: 1, px: 0.5 }}>
            <LinearProgress
              variant={datasetUploadProgress === null ? 'indeterminate' : 'determinate'}
              value={datasetUploadProgress ?? 0}
              aria-label="Dataset upload progress"
              sx={{
                height: 4,
                borderRadius: 999,
                bgcolor: 'rgba(255,255,255,0.08)',
                '& .MuiLinearProgress-bar': {
                  borderRadius: 999,
                  bgcolor: 'var(--accent-yellow)',
                },
              }}
            />
          </Box>
        )}
        {(datasetUploadError || datasetUploadSuccess) && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
            <Alert
              severity={datasetUploadError ? 'error' : 'success'}
              variant="filled"
              onClose={() => {
                setDatasetUploadError(null);
                setDatasetUploadSuccess(null);
              }}
              sx={{ fontSize: '0.8rem', maxWidth: 520, width: '100%' }}
            >
              {datasetUploadError ?? datasetUploadSuccess}
            </Alert>
          </Box>
        )}
        {uploadedDatasets.length > 0 && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, justifyContent: 'center', mt: 1 }}>
            {uploadedDatasets.map((dataset) => (
              <Chip
                key={dataset.upload_id}
                size="small"
                label={`Dataset: ${dataset.filename}`}
                component="a"
                href={datasetRepoUrl(dataset.repo_id)}
                target="_blank"
                rel="noreferrer"
                clickable
                sx={{
                  maxWidth: '100%',
                  bgcolor: 'rgba(255,255,255,0.08)',
                  color: 'var(--text)',
                  border: '1px solid var(--divider)',
                  '& .MuiChip-label': {
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  },
                }}
              />
            ))}
          </Box>
        )}

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
          <img
            src={selectedModel.avatarUrl}
            alt={selectedModel.name}
            style={{ height: '14px', width: '14px', objectFit: 'contain', borderRadius: '2px' }}
          />
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
          {visibleModelOptions.map((model) => (
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
                <img
                  src={model.avatarUrl}
                  alt={model.name}
                  style={{ width: 24, height: 24, borderRadius: '4px', objectFit: 'cover' }}
                />
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
                    {isPremiumModel(model) && premiumChip && (
                      <Chip
                        label={premiumChip}
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
        </Menu>

        <JobsUpgradeDialog
          open={!!jobsUpgradeRequired}
          message={jobsUpgradeRequired?.message || ''}
          awaitingTopUp={awaitingTopUp}
          onClose={handleJobsUpgradeClose}
          onUpgrade={handleJobsUpgradeClick}
          onRetry={handleJobsRetry}
        />
        <Snackbar
          open={!!modelSwitchError}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
          onClose={() => setModelSwitchError(null)}
          autoHideDuration={6000}
        >
          <Alert
            severity="error"
            variant="filled"
            onClose={() => setModelSwitchError(null)}
            sx={{ fontSize: '0.8rem', maxWidth: 480 }}
          >
            {modelSwitchError}
          </Alert>
        </Snackbar>
      </Box>
    </Box>
  );
}
