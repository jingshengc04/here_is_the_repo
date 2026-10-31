import type { Metadata } from "next";
import { Geist_Mono, IBM_Plex_Sans, IBM_Plex_Sans_Condensed } from "next/font/google";
import { Toaster } from "@/components/ui/sonner";
import "./globals.css";

const plex = IBM_Plex_Sans({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const plexCondensed = IBM_Plex_Sans_Condensed({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["500", "600"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "VibeTrader",
  description: "A team of AI agents builds — and audits — your portfolio.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${plex.variable} ${plexCondensed.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <header className="sticky top-0 z-20 border-b bg-background/85 backdrop-blur">
          <div className="mx-auto flex h-14 w-full max-w-[1240px] items-center gap-3 px-6">
            <span className="flex size-7 items-center justify-center rounded-md bg-primary text-sm font-black text-primary-foreground">
              V
            </span>
            <span className="text-xl font-extrabold tracking-tight">VibeTrader</span>
            <span className="hidden text-sm text-muted-foreground sm:inline">
              A team of AI agents builds — and audits — your portfolio
            </span>
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1240px] flex-1 px-6 py-6">{children}</main>
        <footer className="py-5 text-center text-xs text-muted-foreground">
          Not financial advice · Data: Yahoo Finance · OKX · SEC EDGAR · Google Search
        </footer>
        <Toaster theme="light" position="top-center" richColors />
      </body>
    </html>
  );
}
