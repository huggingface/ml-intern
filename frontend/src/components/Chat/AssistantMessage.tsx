import { useMemo, useState } from 'react';
import { Box, IconButton, Stack, Tooltip, Typography } from '@mui/material';
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded';
import CheckRoundedIcon from '@mui/icons-material/CheckRounded';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import MarkdownContent from './MarkdownContent';
import ToolCallGroup from './ToolCallGroup';
import type { UIMessage } from 'ai';
import type { MessageMeta } from '@/types/agent';

interface AssistantMessageProps {
  message: UIMessage;
  isStreaming?: boolean;
  canRegenerate?: boolean;
  onRegenerate?: (assistantMessageId: string) => void | Promise<void>;
  approveTools: (approvals: Array<{ tool_call_id: string; approved: boolean; feedback?: string | null }>) => Promise<boolean>;
}

/**
 * Groups consecutive tool parts together so they render as a single
 * ToolCallGroup (visually identical to the old segments approach).
 */
type DynamicToolPart = Extract<UIMessage['parts'][number], { type: 'dynamic-tool' }>;

function groupParts(parts: UIMessage['parts']) {
  const groups: Array<
    | { kind: 'text'; text: string; idx: number }
    | { kind: 'tools'; tools: DynamicToolPart[]; idx: number }
  > = [];

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];

    if (part.type === 'text') {
      groups.push({ kind: 'text', text: part.text, idx: i });
    } else if (part.type === 'dynamic-tool') {
      const toolPart = part as DynamicToolPart;
      const last = groups[groups.length - 1];
      if (last?.kind === 'tools') {
        last.tools.push(toolPart);
      } else {
        groups.push({ kind: 'tools', tools: [toolPart], idx: i });
      }
    }
    // step-start, step-end, etc. are ignored visually
  }

  return groups;
}

export default function AssistantMessage({
  message,
  isStreaming = false,
  canRegenerate = false,
  onRegenerate,
  approveTools,
}: AssistantMessageProps) {
  const groups = useMemo(() => groupParts(message.parts), [message.parts]);
  const [copied, setCopied] = useState(false);

  const copyText = useMemo(() => {
    return message.parts
      .filter((p): p is Extract<UIMessage['parts'][number], { type: 'text' }> => p.type === 'text')
      .map((p) => p.text)
      .join('\n\n')
      .trim();
  }, [message.parts]);

  const hasText = copyText.length > 0;
  const showActions = !isStreaming && hasText;

  const handleCopy = async () => {
    if (!copyText) return;
    try {
      await navigator.clipboard.writeText(copyText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard API may be unavailable; fail silently
    }
  };

  const handleRegenerate = () => {
    if (!onRegenerate) return;
    onRegenerate(message.id);
  };

  // Find the last text group index for streaming cursor
  let lastTextIdx = -1;
  for (let i = groups.length - 1; i >= 0; i--) {
    if (groups[i].kind === 'text') { lastTextIdx = i; break; }
  }

  const meta = message.metadata as MessageMeta | undefined;
  const timeStr = meta?.createdAt
    ? new Date(meta.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null;

  if (groups.length === 0) return null;

  return (
    <Box sx={{ minWidth: 0 }}>
      <Stack direction="row" alignItems="baseline" spacing={1} sx={{ mb: 0.5 }}>
        <Typography
          variant="caption"
          sx={{
            fontWeight: 700,
            fontSize: '0.72rem',
            color: 'var(--muted-text)',
            textTransform: 'uppercase',
            letterSpacing: '0.04em',
          }}
        >
          Assistant
        </Typography>
        {timeStr && (
          <Typography variant="caption" sx={{ color: 'var(--muted-text)', fontSize: '0.7rem' }}>
            {timeStr}
          </Typography>
        )}
      </Stack>

      <Box
        sx={{
          maxWidth: { xs: '95%', md: '85%' },
          bgcolor: 'var(--surface)',
          borderRadius: 1.5,
          borderTopLeftRadius: 4,
          px: { xs: 1.5, md: 2.5 },
          py: 1.5,
          border: '1px solid var(--border)',
        }}
      >
        {groups.map((group, i) => {
          if (group.kind === 'text' && group.text) {
            return (
              <MarkdownContent
                key={group.idx}
                content={group.text}
                isStreaming={isStreaming && i === lastTextIdx}
              />
            );
          }
          if (group.kind === 'tools' && group.tools.length > 0) {
            return (
              <ToolCallGroup
                key={group.idx}
                tools={group.tools}
                approveTools={approveTools}
              />
            );
          }
          return null;
        })}
      </Box>

      {showActions && (
        <Stack
          direction="row"
          spacing={0.25}
          sx={{
            mt: 0.75,
            ml: 0.25,
            opacity: 0.75,
            transition: 'opacity 0.15s ease',
            '&:hover': { opacity: 1 },
          }}
        >
          <Tooltip title={copied ? 'Copied' : 'Copy message'} placement="bottom">
            <IconButton
              size="small"
              onClick={handleCopy}
              aria-label="Copy message"
              sx={{
                color: copied ? 'var(--accent-yellow)' : 'var(--muted-text)',
                width: 28,
                height: 28,
                borderRadius: 1,
                '&:hover': {
                  color: 'var(--text)',
                  bgcolor: 'var(--hover-bg)',
                },
              }}
            >
              {copied ? (
                <CheckRoundedIcon sx={{ fontSize: 15 }} />
              ) : (
                <ContentCopyRoundedIcon sx={{ fontSize: 15 }} />
              )}
            </IconButton>
          </Tooltip>

          {canRegenerate && onRegenerate && (
            <Tooltip title="Regenerate response" placement="bottom">
              <IconButton
                size="small"
                onClick={handleRegenerate}
                aria-label="Regenerate response"
                sx={{
                  color: 'var(--muted-text)',
                  width: 28,
                  height: 28,
                  borderRadius: 1,
                  '&:hover': {
                    color: 'var(--text)',
                    bgcolor: 'var(--hover-bg)',
                  },
                }}
              >
                <RefreshRoundedIcon sx={{ fontSize: 16 }} />
              </IconButton>
            </Tooltip>
          )}
        </Stack>
      )}
    </Box>
  );
}
