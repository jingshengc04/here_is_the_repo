"use client";

import { ChevronRightIcon } from "lucide-react";
import { cn } from "@/lib/utils";

/** Closed-by-default detail section. Controlled so other parts of the page can open it. */
export function Fold({
  id,
  title,
  hint,
  open,
  onToggle,
  children,
}: {
  id: string;
  title: string;
  hint?: string;
  open: boolean;
  onToggle: (open: boolean) => void;
  children: React.ReactNode;
}) {
  return (
    <details id={id} data-testid={`fold-${id}`} open={open} onToggle={(e) => onToggle((e.currentTarget as HTMLDetailsElement).open)} className="group scroll-mt-20 border-t border-hairline">
      <summary className="flex cursor-pointer list-none items-baseline gap-3 py-4 outline-none focus-visible:ring-2 focus-visible:ring-ink/30 [&::-webkit-details-marker]:hidden">
        <ChevronRightIcon className={cn("size-4 shrink-0 translate-y-0.5 text-slate transition-transform", open && "rotate-90")} />
        <span className="text-lg font-semibold">{title}</span>
        {hint && <span className="hidden truncate text-sm text-slate sm:inline">{hint}</span>}
      </summary>
      {open && <div className="pb-6 pl-7">{children}</div>}
    </details>
  );
}
