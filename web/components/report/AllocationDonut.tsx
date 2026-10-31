"use client";

import { Component, lazy, Suspense, useSyncExternalStore, type ReactNode } from "react";
import type { Allocation } from "@/lib/types";
import { Donut3D } from "./Donut3D";

// WebGL donut (three.js) when the browser supports it; the SVG 2.5D donut otherwise, while three.js
// loads, if WebGL crashes, or with ?three=0 (e.g. a weak projector laptop).
const Donut3DGL = lazy(() => import("./Donut3DGL").then((m) => ({ default: m.Donut3DGL })));

function canUseWebGL(): boolean {
  if (new URLSearchParams(window.location.search).get("three") === "0") return false;
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}
let webgl: boolean | null = null;
const webglSnapshot = () => (webgl ??= canUseWebGL());
const noopSubscribe = () => () => {};

class Fallback extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

interface Props {
  allocation: Allocation;
  selected: string | null;
  onSelect: (symbol: string) => void;
}

export function AllocationDonut(props: Props) {
  const gl = useSyncExternalStore(noopSubscribe, webglSnapshot, () => false);
  const svg = <Donut3D {...props} />;
  if (!gl) return svg;
  return (
    <Fallback fallback={svg}>
      <Suspense fallback={svg}>
        <Donut3DGL {...props} />
      </Suspense>
    </Fallback>
  );
}
