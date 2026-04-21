import { useMemo, useState } from 'react';
import {
  Box,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  InputAdornment,
  TextField,
  Typography,
  Chip,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import CloseIcon from '@mui/icons-material/Close';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import TuneIcon from '@mui/icons-material/Tune';
import DatasetIcon from '@mui/icons-material/Dataset';
import CleaningServicesIcon from '@mui/icons-material/CleaningServices';
import PsychologyIcon from '@mui/icons-material/Psychology';
import AssessmentIcon from '@mui/icons-material/Assessment';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import ScienceIcon from '@mui/icons-material/Science';
import TerminalIcon from '@mui/icons-material/Terminal';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import BugReportIcon from '@mui/icons-material/BugReport';
import CloudSyncIcon from '@mui/icons-material/CloudSync';

interface PromptCategory {
  id: string;
  label: string;
  description: string;
  icon: React.ReactNode;
  prompts: string[];
}

// Prompts derived from agent/prompts/system_prompt_v3.yaml and system_prompt_v2.yaml.
// They align with the agent's real workflow: research -> validate -> implement -> verify.
const EXAMPLE_PROMPT_CATEGORIES: PromptCategory[] = [
  {
    id: 'all',
    label: 'All prompts',
    description: 'Browse every example prompt in one place.',
    icon: <AutoAwesomeIcon sx={{ fontSize: 18 }} />,
    prompts: [],
  },
  {
    id: 'sft',
    label: 'SFT Fine-tuning',
    description: 'Supervised fine-tuning with TRL, Trackio, and push_to_hub.',
    icon: <ModelTrainingIcon sx={{ fontSize: 18 }} />,
    prompts: [
      'Research current TRL SFTTrainer usage, then fine-tune Qwen/Qwen2.5-1.5B-Instruct on HuggingFaceH4/ultrachat_200k with Trackio monitoring, push_to_hub enabled, and an appropriate timeout.',
      'Fine-tune a 3B instruct model for chat completion. Validate the dataset columns match SFT (messages/text or prompt/completion) before submitting the job.',
      'Create a full SFT training job with disable_tqdm=True, logging_strategy="steps", logging_first_step=True so loss values are greppable from logs.',
      'Plan and run a small 1B SFT smoke test first (short timeout) to verify the pipeline, then scale to a longer full training run on a10g-large.',
      'Fine-tune Llama-3.2-1B on an instruction dataset and push the resulting model to my Hub namespace, including a proper hub_model_id.',
      'Prepare an SFT training script that includes Trackio project/run_name/config, then submit it with dependencies [transformers, trl, torch, datasets, trackio].',
    ],
  },
  {
    id: 'dpo-grpo',
    label: 'DPO & GRPO',
    description: 'Preference optimization workflows with validated dataset formats.',
    icon: <TuneIcon sx={{ fontSize: 18 }} />,
    prompts: [
      'Research current DPO training best practices, verify my dataset has prompt/chosen/rejected columns, and submit a DPO run with proper monitoring.',
      'Set up a GRPO pipeline on a prompt-only dataset. Confirm dataset format, pick an appropriate base model, and configure hardware based on model size.',
      'Compare DPO vs SFT for my preference-style dataset and recommend which method fits based on dataset columns and expected outcomes.',
      'Submit a DPO run for a 7B instruct base model on a10g-large with Trackio dashboard and push_to_hub enabled.',
      'Create a reproducible GRPO baseline script and evaluate it after training using a simple benchmark config.',
    ],
  },
  {
    id: 'dataset-discovery',
    label: 'Dataset Discovery',
    description: 'Find, inspect, and validate datasets before training.',
    icon: <DatasetIcon sx={{ fontSize: 18 }} />,
    prompts: [
      'Find 3-5 conversational datasets suitable for instruction tuning, verify their columns, and recommend the best fit for SFT.',
      'Use hf_inspect_dataset to audit a dataset for schema, missing values, class imbalance, and duplicates. Surface notable risks before training.',
      'Search for reasoning-style datasets with clear prompt/completion format and verify them with hub_repo_details.',
      'Discover domain-specific datasets (medical, legal, code) and report license, size, and exact columns for each.',
      'Pick the best dataset for DPO training from candidates, ensuring they have prompt/chosen/rejected columns.',
    ],
  },
  {
    id: 'dataset-processing',
    label: 'Data Processing',
    description: 'Clean, filter, and publish datasets back to Hub.',
    icon: <CleaningServicesIcon sx={{ fontSize: 18 }} />,
    prompts: [
      'Load a dataset, filter rows where context length is greater than a threshold, and push the processed version to my private dataset repo.',
      'Standardize a noisy dataset into the SFT messages format and publish it to the Hub with a proper dataset card.',
      'Deduplicate a dataset using simple near-duplicate heuristics and push the clean version back to Hub.',
      'Convert a raw dataset into prompt/chosen/rejected format suitable for DPO training and push it to my namespace.',
      'Split a dataset into train/validation/test with a reproducible seed and upload it to Hub.',
      'Process a large dataset on cpu-upgrade hardware with an appropriate timeout and store results via push_to_hub.',
    ],
  },
  {
    id: 'model-discovery',
    label: 'Model Discovery',
    description: 'Search, compare, and pick optimal base models.',
    icon: <PsychologyIcon sx={{ fontSize: 18 }} />,
    prompts: [
      'Compare small instruct models (1B-3B) for fine-tuning quality, inference cost, and memory needs. Recommend one with reasoning.',
      'Find the best coding-focused base model under 7B and validate its tokenizer, architecture, and license.',
      'Suggest an optimal base model for domain-specific SFT given my dataset size and budget constraints.',
      'Compare Qwen vs Llama vs Mistral at the same parameter count for instruction following.',
      'Help me pick between LoRA and full SFT for my use case without silently changing the user-requested approach.',
    ],
  },
  {
    id: 'inference',
    label: 'Inference',
    description: 'Build reliable inference scripts and run them as jobs.',
    icon: <RocketLaunchIcon sx={{ fontSize: 18 }} />,
    prompts: [
      'Build an inference script that uses pipeline() with explicit generation parameters and run it on a10g-small via hf_jobs.',
      'Run batched inference on a dataset and store outputs in a private dataset repo for later analysis.',
      'Create a reproducible inference job that writes structured JSONL results and pushes them to my Hub namespace.',
      'Run inference for my fine-tuned model and save both prompts and completions to a dataset for qualitative review.',
    ],
  },
  {
    id: 'evaluation',
    label: 'Evaluation',
    description: 'Evaluate models with reproducible benchmark setups.',
    icon: <AssessmentIcon sx={{ fontSize: 18 }} />,
    prompts: [
      'Evaluate my fine-tuned model using lighteval on a small benchmark set and store the metrics in a private dataset repo.',
      'Compare baseline vs fine-tuned model on a held-out eval set and produce a concise metrics summary.',
      'Design a repeatable evaluation workflow that runs automatically after every new training run.',
      'Set up an lm-evaluation-harness job for a chosen model with appropriate hardware and results persistence.',
      'Create an eval script that outputs a markdown report with scores, sample generations, and failure cases.',
    ],
  },
  {
    id: 'research',
    label: 'Research & Papers',
    description: 'Literature crawl, citation graphs, and implementation recipes.',
    icon: <MenuBookIcon sx={{ fontSize: 18 }} />,
    prompts: [
      'Run a literature crawl for instruction tuning. Start from landmark papers, crawl the citation graph, and extract concrete dataset + training method combinations that produced strong results.',
      'Find recent DPO-style alignment papers, read methodology sections, and propose an implementation plan grounded in published recipes.',
      'Search for papers that cite a given baseline method and summarize which ones improved it and how.',
      'Extract an end-to-end training recipe for reasoning models (dataset, method, hyperparameters) from recent literature.',
      'Research best practices for mixture-of-experts fine-tuning from current papers and propose a practical plan.',
    ],
  },
  {
    id: 'sweeps',
    label: 'Hyperparameter Sweeps',
    description: 'Automated sweep scripts instead of manual one-at-a-time tuning.',
    icon: <ScienceIcon sx={{ fontSize: 18 }} />,
    prompts: [
      'Write a sweep script that launches a grid over learning rate, batch size, and epochs, then evaluates each run automatically.',
      'Set up a small LR sweep (3-5 values) on a smoke-tested training job and pick the best run by eval metric.',
      'Design a DPO sweep across beta and learning rate. Submit one smoke job first before launching the remaining runs.',
      'Plan a staged sweep: small smoke runs first to validate the config, then full runs only on promising hyperparameters.',
    ],
  },
  {
    id: 'sandbox',
    label: 'Sandbox Development',
    description: 'Iterate fast in a sandbox before launching full jobs.',
    icon: <TerminalIcon sx={{ fontSize: 18 }} />,
    prompts: [
      'Create a GPU sandbox (t4-small), install TRL and dependencies, write an SFT script, test it with a small run, then scale via hf_jobs.',
      'Use a sandbox to test a data processing script on a small slice of the dataset, then run the full job.',
      'Debug a CUDA-based training script inside a sandbox before submitting it as a long-running job.',
      'Prototype a new evaluation script in a sandbox, verify outputs, then run it at scale via hf_jobs.',
    ],
  },
  {
    id: 'spaces',
    label: 'Spaces & Deployment',
    description: 'Deploy, debug, and publish Gradio/Streamlit Spaces.',
    icon: <CloudSyncIcon sx={{ fontSize: 18 }} />,
    prompts: [
      'Debug a Space that crashes on startup by inspecting requirements.txt, app.py, and likely import/dependency mismatches.',
      'Publish a Gradio demo for my fine-tuned model with a simple UI and proper Space dependencies.',
      'Create a reproducible Space deployment plan: app.py, requirements.txt, README, and health checks.',
      'Convert my inference script into a Gradio Space that accepts a prompt and returns the model completion.',
    ],
  },
  {
    id: 'debug',
    label: 'Debugging & Reliability',
    description: 'Diagnose failures with minimal, scope-preserving fixes.',
    icon: <BugReportIcon sx={{ fontSize: 18 }} />,
    prompts: [
      'My training job keeps failing with CUDA OOM. Diagnose logs, then apply minimal fixes (grad accumulation, gradient checkpointing, bigger GPU) without changing training method or max_length.',
      'Review my hf_jobs configuration for timeout, hardware mismatch, missing push_to_hub, and missing HF_TOKEN before submission.',
      'My job crashed with a KeyError on dataset columns. Inspect dataset schema and fix the script to match SFT/DPO/GRPO format.',
      'I have ImportError in my training script. Find correct current imports in TRL/Transformers and fix the script without changing scope.',
      'Create a recovery plan for a failed training run: diagnose root cause, apply a minimal fix, and re-run with monitoring.',
      'My Space logs show a startup error. Identify the missing package, update requirements.txt, and redeploy.',
    ],
  },
];

interface ExamplePromptsDialogProps {
  open: boolean;
  onClose: () => void;
  onSelectPrompt: (prompt: string) => void;
}

export default function ExamplePromptsDialog({
  open,
  onClose,
  onSelectPrompt,
}: ExamplePromptsDialogProps) {
  const [promptSearch, setPromptSearch] = useState('');
  const [selectedCategoryId, setSelectedCategoryId] = useState<string>('all');

  const allPrompts = useMemo(() => {
    return EXAMPLE_PROMPT_CATEGORIES.filter((c) => c.id !== 'all').flatMap((c) =>
      c.prompts.map((p) => ({ prompt: p, categoryId: c.id, categoryLabel: c.label }))
    );
  }, []);

  const resultPrompts = useMemo(() => {
    const query = promptSearch.trim().toLowerCase();

    // Global search overrides category selection
    if (query) {
      return allPrompts.filter(
        (entry) =>
          entry.prompt.toLowerCase().includes(query) ||
          entry.categoryLabel.toLowerCase().includes(query)
      );
    }

    if (selectedCategoryId === 'all') return allPrompts;
    return allPrompts.filter((entry) => entry.categoryId === selectedCategoryId);
  }, [allPrompts, selectedCategoryId, promptSearch]);

  const handleSelectPrompt = (prompt: string) => {
    onSelectPrompt(prompt);
    onClose();
    setPromptSearch('');
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          bgcolor: 'var(--panel)',
          border: '1px solid var(--border)',
          borderRadius: 3,
          height: { xs: '85vh', md: '78vh' },
          maxHeight: { xs: '85vh', md: '78vh' },
          overflow: 'hidden',
          boxShadow: 'var(--shadow-1)',
          display: 'flex',
          flexDirection: 'column',
        },
      }}
    >
      {/* ── Header: title · centered search · close ─────────────────────── */}
      <DialogTitle
        sx={{
          display: 'grid',
          gridTemplateColumns: '1fr minmax(0, 520px) 1fr',
          alignItems: 'center',
          gap: 2,
          borderBottom: '1px solid var(--border)',
          py: 1.5,
          px: { xs: 2, md: 3 },
          background:
            'radial-gradient(500px 160px at 50% -20%, color-mix(in oklch, var(--accent-yellow) 10%, transparent), transparent 60%)',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
          <AutoAwesomeIcon sx={{ fontSize: 18, color: 'var(--accent-yellow)' }} />
          <Typography
            sx={{
              fontWeight: 800,
              fontSize: '1.1rem',
              letterSpacing: '-0.01em',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            Example Prompts
          </Typography>
        </Box>

        <TextField
          size="small"
          value={promptSearch}
          onChange={(e) => setPromptSearch(e.target.value)}
          placeholder="Search every prompt and category…"
          sx={{
            justifySelf: 'center',
            width: '100%',
            '& .MuiOutlinedInput-root': {
              bgcolor: 'var(--surface)',
              borderRadius: 999,
              paddingLeft: 1,
            },
          }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon sx={{ color: 'var(--muted-text)', fontSize: 18 }} />
              </InputAdornment>
            ),
          }}
        />

        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
          <IconButton onClick={onClose} size="small">
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent
        sx={{
          p: 0,
          display: 'flex',
          flex: 1,
          minHeight: 0,
          overflow: 'hidden',
        }}
      >
        {/* ── Left: categories ─────────────────────────────────────────── */}
        <Box
          sx={{
            width: { xs: 220, md: 260 },
            flexShrink: 0,
            borderRight: '1px solid var(--border)',
            p: 1.2,
            display: 'flex',
            flexDirection: 'column',
            gap: 0.4,
            overflowY: 'auto',
            scrollbarWidth: 'none',
            msOverflowStyle: 'none',
            '&::-webkit-scrollbar': {
              display: 'none',
            },
          }}
        >
          {EXAMPLE_PROMPT_CATEGORIES.map((category) => {
            const active =
              category.id === selectedCategoryId &&
              promptSearch.trim().length === 0;
            return (
              <Box
                key={category.id}
                onClick={() => {
                  setSelectedCategoryId(category.id);
                  setPromptSearch('');
                }}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1.1,
                  px: 1.25,
                  py: 1,
                  borderRadius: 2,
                  cursor: 'pointer',
                  color: active ? 'var(--text)' : 'var(--muted-text)',
                  bgcolor: active ? 'var(--hover-bg)' : 'transparent',
                  transition: 'background-color 0.15s ease, color 0.15s ease',
                  '&:hover': {
                    bgcolor: 'var(--hover-bg)',
                    color: 'var(--text)',
                  },
                }}
              >
                <Box
                  sx={{
                    color: active ? 'var(--accent-yellow)' : 'var(--muted-text)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: 22,
                  }}
                >
                  {category.icon}
                </Box>
                <Typography
                  sx={{
                    flex: 1,
                    fontWeight: active ? 700 : 600,
                    fontSize: '0.84rem',
                  }}
                >
                  {category.label}
                </Typography>
              </Box>
            );
          })}
        </Box>

        {/* ── Right: prompt grid ───────────────────────────────────────── */}
        <Box
          sx={{
            flex: 1,
            minWidth: 0,
            minHeight: 0,
            p: { xs: 1.5, md: 2.25 },
            display: 'flex',
            flexDirection: 'column',
            gap: 1.25,
          }}
        >
          <Box
            sx={{
              display: 'flex',
              alignItems: 'baseline',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 1,
            }}
          >
            <Typography sx={{ fontWeight: 700, fontSize: '0.98rem' }}>
              {promptSearch.trim()
                ? `Search results`
                : EXAMPLE_PROMPT_CATEGORIES.find((c) => c.id === selectedCategoryId)?.label}
            </Typography>
            <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.78rem' }}>
              {resultPrompts.length} result{resultPrompts.length === 1 ? '' : 's'}
            </Typography>
          </Box>

          {!promptSearch.trim() && (
            <Typography sx={{ color: 'var(--muted-text)', fontSize: '0.82rem', mt: -0.5 }}>
              {EXAMPLE_PROMPT_CATEGORIES.find((c) => c.id === selectedCategoryId)?.description}
            </Typography>
          )}

          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' },
              gap: 1.25,
              alignContent: 'start',
              flex: 1,
              minHeight: 0,
              overflowY: 'auto',
              pr: 0.5,
              pt: 1,
              mt: 0.25,
              scrollbarWidth: 'none',
              msOverflowStyle: 'none',
              '&::-webkit-scrollbar': {
                display: 'none',
              },
            }}
          >
            {resultPrompts.length === 0 ? (
              <Box
                sx={{
                  gridColumn: '1 / -1',
                  textAlign: 'center',
                  color: 'var(--muted-text)',
                  fontSize: '0.88rem',
                  py: 6,
                }}
              >
                No prompts matched your search. Try a different keyword.
              </Box>
            ) : (
              resultPrompts.map(({ prompt, categoryLabel, categoryId }) => (
                <Box
                  key={`${categoryId}-${prompt}`}
                  onClick={() => handleSelectPrompt(prompt)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleSelectPrompt(prompt);
                    }
                  }}
                  sx={{
                    position: 'relative',
                    border: '1px solid var(--border)',
                    borderRadius: 2,
                    p: 1.4,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 0.9,
                    background: 'var(--surface)',
                    cursor: 'pointer',
                    transition:
                      'border-color 0.18s ease, transform 0.18s ease, background 0.18s ease, box-shadow 0.18s ease',
                    outline: 'none',
                    '&:hover, &:focus-visible': {
                      borderColor: 'var(--accent-yellow)',
                      background:
                        'color-mix(in oklch, var(--surface) 88%, var(--accent-yellow) 12%)',
                      transform: 'translateY(-2px)',
                      boxShadow: '0 10px 24px -18px rgba(0,0,0,0.6)',
                    },
                  }}
                >
                  <Chip
                    size="small"
                    label={categoryLabel}
                    sx={{
                      alignSelf: 'flex-start',
                      height: 20,
                      fontSize: '0.66rem',
                      fontWeight: 700,
                      bgcolor: 'var(--panel)',
                      color: 'var(--muted-text)',
                      border: '1px solid var(--border)',
                    }}
                  />
                  <Typography
                    sx={{
                      color: 'var(--text)',
                      fontSize: '0.86rem',
                      lineHeight: 1.5,
                    }}
                  >
                    {prompt}
                  </Typography>
                </Box>
              ))
            )}
          </Box>
        </Box>
      </DialogContent>
    </Dialog>
  );
}
