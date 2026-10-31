"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2Icon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { horizonLabel, usd } from "@/lib/format";
import { Md } from "@/lib/markdown";
import type { ChatMessage, IntakeTurn } from "@/lib/types";

export const GREETING = "Hi! How much are you looking to invest, and what are you hoping for?";
export const MAX_MESSAGES = 20;

const RISK = { CONSERVATIVE: "Conservative", MODERATE: "Moderate", AGGRESSIVE: "Aggressive" } as const;

function DraftChips({ draft }: { draft: IntakeTurn["intake"] | null }) {
  const items = [
    ["Amount", draft?.amountUsd != null ? usd(draft.amountUsd) : null],
    ["Horizon", draft?.horizonYears != null ? horizonLabel(draft.horizonYears) : null],
    ["Risk", draft?.riskTolerance ? RISK[draft.riskTolerance] : null],
    ["Theme", draft?.theme ?? null],
  ] as const;
  const p = draft?.preferences;
  const prefs = [
    ...(p?.mustInclude ?? []).map((x) => `+ ${x}`),
    ...(p?.exclude ?? []).map((x) => `no ${x}`),
    ...(p?.onlyInclude ?? []).map((x) => `only ${x}`),
    ...(p?.styles ?? []).map((x) => x.charAt(0) + x.slice(1).toLowerCase()),
  ];
  return (
    <div data-testid="intake-draft" className="flex flex-wrap items-center gap-2 border-b pb-3">
      <span className="mr-1 text-sm text-muted-foreground">So far</span>
      {items.map(([label, value]) =>
        value ? (
          <span key={label} className="rounded-full border border-edge bg-edge-bg px-2.5 py-0.5 text-xs font-semibold text-edge">
            ✓ {value}
          </span>
        ) : (
          <span key={label} className="rounded-full border border-dashed border-neutral-300 px-2.5 py-0.5 text-xs text-neutral-400">
            {label}
          </span>
        ),
      )}
      {prefs.map((x) => (
        <span key={x} className="rounded-full border border-[#2f5bea]/40 bg-[#2f5bea]/[0.06] px-2.5 py-0.5 text-xs font-semibold text-[#2f5bea]">
          {x}
        </span>
      ))}
    </div>
  );
}

/** Splits off the backend's authoritative "Saved: a · b · c" confirmation and shows it as chips. */
function AssistantText({ text }: { text: string }) {
  const i = text.search(/(?:^|\s)Saved:\s/);
  if (i < 0) return <Md text={text} />;
  const body = text.slice(0, i).trim();
  const items = text
    .slice(i)
    .replace(/^\s*Saved:\s*/, "")
    .split(/\s*·\s*/)
    .filter(Boolean);
  return (
    <>
      {body && <Md text={body} />}
      <div data-testid="saved-line" className={body ? "mt-2 border-t border-ink/10 pt-2" : undefined}>
        <span className="mr-1.5 text-[13px] font-semibold text-slate">Saved</span>
        {items.map((x) => (
          <span key={x} className="mr-1 mb-1 inline-block rounded-full bg-card px-2 py-0.5 text-[13px] font-medium ring-1 ring-ink/10">
            {x}
          </span>
        ))}
      </div>
    </>
  );
}

interface Props {
  messages: ChatMessage[]; // full visible history, greeting included
  draft: IntakeTurn["intake"] | null;
  pending: boolean;
  onSend: (text: string) => void;
  onSkip: () => void;
}

export function IntakeChat({ messages, draft, pending, onSend, onSkip }: Props) {
  const [text, setText] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  const full = messages.length >= MAX_MESSAGES;

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages.length, pending]);

  const send = () => {
    const t = text.trim();
    if (!t || pending || full) return;
    onSend(t.slice(0, 2000));
    setText("");
  };

  return (
    <section aria-label="Onboarding chat" className="flex h-[560px] flex-col rounded-xl border bg-card p-4 shadow-xs">
      <DraftChips draft={draft} />
      <div ref={listRef} data-testid="chat-log" className="flex-1 space-y-2.5 overflow-y-auto py-4 pr-1">
        {messages.map((m, i) => (
          <div
            key={i}
            data-role={m.role}
            className={
              m.role === "user"
                ? "ml-auto max-w-[80%] rounded-2xl rounded-br-md bg-primary px-3.5 py-2 text-[15px] leading-snug text-primary-foreground"
                : "max-w-[80%] rounded-2xl rounded-bl-md bg-neutral-100 px-3.5 py-2 text-[15px] leading-snug"
            }
          >
            {m.role === "assistant" ? <AssistantText text={m.content} /> : m.content}
          </div>
        ))}
        {pending && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2Icon className="size-3.5 animate-spin" /> Intake agent is typing…
          </div>
        )}
      </div>
      <div className="flex items-end gap-2 border-t pt-3">
        <Textarea
          aria-label="Message"
          rows={1}
          value={text}
          maxLength={2000}
          disabled={full}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send();
            }
          }}
          placeholder={full ? "Chat limit reached. Use the form." : "Type a message…"}
          className="max-h-32 min-h-10 resize-none bg-card text-[15px]"
        />
        <Button onClick={send} disabled={!text.trim() || pending || full} className="h-10! px-4! font-bold">
          Send
        </Button>
      </div>
      <button type="button" onClick={onSkip} className="mt-2 self-start text-xs text-muted-foreground underline-offset-2 hover:underline">
        Prefer a form? Skip the chat
      </button>
    </section>
  );
}
