import { useState, useCallback, useEffect, useRef, KeyboardEvent } from 'react';
import {
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
  Button,
} from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import StopIcon from '@mui/icons-material/Stop';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { apiFetch } from '@/utils/api';
import ExamplePromptsDialog from '@/components/Chat/ExamplePromptsDialog';

// Model configuration
interface ModelOption {
  id: string;
  name: string;
  description: string;
  modelPath: string;
  avatarUrl: string;
  recommended?: boolean;
}

const getHfAvatarUrl = (modelId: string) => {
  const org = modelId.split('/')[0];
  return `https://huggingface.co/api/avatars/${org}`;
};

const MODEL_OPTIONS: ModelOption[] = [
  {
    id: 'claude-opus',
    name: 'Claude Opus 4.6',
    description: 'Anthropic',
    modelPath: 'anthropic/claude-opus-4-6',
    avatarUrl: 'https://huggingface.co/api/avatars/Anthropic',
    recommended: true,
  },
  {
    id: 'minimax-m2.5',
    name: 'MiniMax M2.5',
    description: 'Via Fireworks',
    modelPath: 'huggingface/fireworks-ai/MiniMaxAI/MiniMax-M2.5',
    avatarUrl: getHfAvatarUrl('MiniMaxAI/MiniMax-M2.5'),
    recommended: true,
  },
  {
    id: 'kimi-k2.5',
    name: 'Kimi K2.5',
    description: 'Via Novita',
    modelPath: 'huggingface/novita/moonshotai/kimi-k2.5',
    avatarUrl: getHfAvatarUrl('moonshotai/Kimi-K2.5'),
  },
  {
    id: 'glm-5',
    name: 'GLM 5',
    description: 'Via Novita',
    modelPath: 'huggingface/novita/zai-org/glm-5',
    avatarUrl: getHfAvatarUrl('zai-org/GLM-5'),
  },
];

const findModelByPath = (path: string): ModelOption | undefined => {
  return MODEL_OPTIONS.find(m => m.modelPath === path || path?.includes(m.id));
};

interface ChatInputProps {
  onSend: (text: string) => void;
  onStop?: () => void;
  isProcessing?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({ onSend, onStop, isProcessing = false, disabled = false, placeholder = 'Describe what you want to research, build, or ship....' }: ChatInputProps) {
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [selectedModelId, setSelectedModelId] = useState<string>(() => {
    try {
      const stored = localStorage.getItem('hf-agent-model');
      if (stored && MODEL_OPTIONS.some(m => m.id === stored)) return stored;
    } catch { /* localStorage unavailable */ }
    return MODEL_OPTIONS[0].id;
  });
  const [modelAnchorEl, setModelAnchorEl] = useState<null | HTMLElement>(null);
  const [promptsOpen, setPromptsOpen] = useState(false);

  // Sync with backend on mount (backend is source of truth, localStorage is just a cache)
  useEffect(() => {
    fetch('/api/config/model')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data?.current) {
          const model = findModelByPath(data.current);
          if (model) {
            setSelectedModelId(model.id);
            try { localStorage.setItem('hf-agent-model', model.id); } catch { /* ignore */ }
          }
        }
      })
      .catch(() => { /* ignore */ });
  }, []);

  const selectedModel = MODEL_OPTIONS.find(m => m.id === selectedModelId) || MODEL_OPTIONS[0];

  // Auto-focus the textarea when the session becomes ready
  useEffect(() => {
    if (!disabled && !isProcessing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled, isProcessing]);

  const handleSend = useCallback(() => {
    if (input.trim() && !disabled) {
      onSend(input);
      setInput('');
    }
  }, [input, disabled, onSend]);

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
    try {
      const res = await apiFetch('/api/config/model', {
        method: 'POST',
        body: JSON.stringify({ model: model.modelPath }),
      });
      if (res.ok) {
        setSelectedModelId(model.id);
        try { localStorage.setItem('hf-agent-model', model.id); } catch { /* ignore */ }
      }
    } catch { /* ignore */ }
  };

  const handleUsePrompt = useCallback((prompt: string) => {
    setInput(prompt);
    setPromptsOpen(false);
    setTimeout(() => inputRef.current?.focus(), 0);
  }, []);

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
            gap: '12px',
            alignItems: 'flex-start',
            bgcolor: 'var(--composer-bg)',
            borderRadius: 'var(--radius-md)',
            p: '14px',
            border: '1px solid var(--border)',
            transition: 'box-shadow 0.2s ease, border-color 0.2s ease, transform 0.18s ease',
            '&:focus-within': {
                borderColor: 'var(--accent-yellow)',
                boxShadow: 'var(--focus)',
                transform: 'translateY(-1px)',
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
                    fontSize: '0.96rem',
                    fontFamily: 'inherit',
                    padding: 0,
                    lineHeight: 1.5,
                    minHeight: { xs: '46px', md: '58px' },
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

        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            mt: 1.25,
            gap: 1,
            flexWrap: 'wrap',
          }}
        >
          <Button
            variant="outlined"
            onClick={() => setPromptsOpen(true)}
            startIcon={<AutoAwesomeIcon sx={{ fontSize: 16 }} />}
            sx={{
              borderColor: 'var(--border)',
              color: 'var(--text)',
              borderRadius: '999px',
              px: 1.5,
              py: 0.55,
              fontSize: '0.78rem',
              fontWeight: 700,
              '&:hover': {
                borderColor: 'var(--accent-yellow)',
                bgcolor: 'var(--hover-bg)',
              },
            }}
          >
            Example Prompts
          </Button>

          {/* Model Badge */}
          <Box
            onClick={handleModelClick}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.8,
              opacity: 0.72,
              cursor: 'pointer',
              transition: 'opacity 0.2s, transform 0.2s',
              '&:hover': {
                opacity: 1,
                transform: 'translateY(-1px)',
              },
            }}
          >
            <Typography variant="caption" sx={{ fontSize: '0.62rem', color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600 }}>
              Model
            </Typography>
            <img
              src={selectedModel.avatarUrl}
              alt={selectedModel.name}
              style={{ height: '14px', width: '14px', objectFit: 'contain', borderRadius: '2px' }}
            />
            <Typography variant="caption" sx={{ fontSize: '0.7rem', color: 'var(--text)', fontWeight: 700, letterSpacing: '0.02em' }}>
              {selectedModel.name}
            </Typography>
            <ArrowDropDownIcon sx={{ fontSize: '14px', color: 'var(--muted-text)' }} />
          </Box>
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
          {MODEL_OPTIONS.map((model) => (
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

        <ExamplePromptsDialog
          open={promptsOpen}
          onClose={() => setPromptsOpen(false)}
          onSelectPrompt={handleUsePrompt}
        />
      </Box>
    </Box>
  );
}
