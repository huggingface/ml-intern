import { useState, useCallback, useEffect, useRef, KeyboardEvent } from 'react';
import { Box, TextField, IconButton, CircularProgress, Typography, Menu, MenuItem, ListItemIcon, ListItemText, Chip } from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import { apiFetch } from '@/utils/api';

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
    id: 'minimax-m2.1',
    name: 'MiniMax M2.1',
    description: 'Via Novita',
    modelPath: 'huggingface/novita/minimax/minimax-m2.1',
    avatarUrl: getHfAvatarUrl('MiniMaxAI/MiniMax-M2.1'),
    recommended: true,
  },
  {
    id: 'claude-opus',
    name: 'Claude Opus 4.5',
    description: 'Anthropic',
    modelPath: 'anthropic/claude-opus-4-5-20251101',
    avatarUrl: 'https://huggingface.co/api/avatars/Anthropic',
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
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({ onSend, disabled = false, placeholder = 'Ask anything...' }: ChatInputProps) {
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

  // Auto-focus the textarea when the session becomes ready (disabled -> false)
  useEffect(() => {
    if (!disabled && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled]);

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
            disabled={disabled}
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
            {disabled ? <CircularProgress size={20} color="inherit" /> : <ArrowUpwardIcon fontSize="small" />}
          </IconButton>
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
      </Box>
    </Box>
  );
}
