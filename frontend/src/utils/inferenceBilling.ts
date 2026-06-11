export const INFERENCE_PROVIDERS_PRICING_URL = 'https://huggingface.co/docs/inference-providers/pricing';
export const HF_PRO_SUBSCRIBE_URL = 'https://huggingface.co/subscribe/pro';

export type PlanTier = 'free' | 'pro';

export function isInferenceCreditError(error: string | undefined, errorType?: string): boolean {
  if (errorType === 'credits') return true;
  const value = (error ?? '').toLowerCase();
  return (
    value.includes('402') ||
    (value.includes('credit') && (
      value.includes('insufficient') ||
      value.includes('exhausted') ||
      value.includes('out of') ||
      value.includes('billing')
    ))
  );
}

export function inferenceCreditCta(plan: PlanTier | undefined) {
  if (plan === 'pro') {
    return {
      title: 'Inference credits exhausted',
      message: 'Your HF account needs more Inference Providers credits before this model can continue.',
      primaryLabel: 'View pricing',
      primaryHref: INFERENCE_PROVIDERS_PRICING_URL,
      secondaryLabel: null,
      secondaryHref: null,
    };
  }

  return {
    title: 'Inference credits exhausted',
    message: 'Upgrade to HF PRO for more monthly Inference Providers usage, or review pay-as-you-go pricing.',
    primaryLabel: 'Upgrade to PRO',
    primaryHref: HF_PRO_SUBSCRIBE_URL,
    secondaryLabel: 'View pricing',
    secondaryHref: INFERENCE_PROVIDERS_PRICING_URL,
  };
}
