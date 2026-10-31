/** Finding.numbersVerified: true = every number matched SEC/yfinance (±2%), false = mismatch, null/undefined = no numbers. */
export function NumbersBadge({ verified }: { verified?: boolean | null }) {
  if (verified === true)
    return (
      <span data-testid="numbers-verified" className="ml-1 rounded-full bg-edge-bg px-1.5 py-px text-[10px] font-bold tracking-normal text-edge" title="Every number matched SEC / Yahoo Finance data within ±2%">
        ✓ numbers verified
      </span>
    );
  if (verified === false)
    return (
      <span data-testid="numbers-unverified" className="ml-1 rounded-full bg-inconclusive-bg px-1.5 py-px text-[10px] font-bold tracking-normal text-inconclusive" title="A number in this claim did not match SEC / Yahoo Finance data">
        ⚠ numbers don&apos;t match data
      </span>
    );
  return null;
}
