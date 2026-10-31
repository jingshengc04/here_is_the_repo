"use client";

import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { animated, useSpring } from "@react-spring/three";
import { Environment, Lightformer, PresentationControls } from "@react-three/drei";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { EffectComposer, N8AO } from "@react-three/postprocessing";
import * as THREE from "three";
import { CLASS_COLOR, CLASS_LABEL, shade, short, usd, weight } from "@/lib/format";
import type { Allocation, AssetClass } from "@/lib/types";

// WebGL allocation donut: extruded ring segments with physical materials, studio lighting built from
// Lightformers (no HDR download), contact shadow and ambient occlusion. Inner taller ring = asset
// classes, outer ring = holdings. Drag to turn; click a holding to select it.

const TAU = Math.PI * 2;
const OUTER = { r0: 2.0, r1: 3.05, depth: 0.32, gap: 0.03 };
const INNER = { r0: 0.82, r1: 1.86, depth: 0.62, gap: 0.03 };
const BG = "#f6f7f9";

interface SliceSpec {
  key: string;
  symbol?: string;
  a0: number; // radians, a0 < a1; world position = (r cos θ, y, -r sin θ)
  a1: number;
  color: string;
  label: string | null;
}

function ringSegment(r0: number, r1: number, a0: number, a1: number, depth: number, gap: number) {
  // Same linear gap on both edges: a smaller angle on the outer arc than on the inner one.
  const shape = new THREE.Shape();
  shape.absarc(0, 0, r1, a0 + gap / r1, a1 - gap / r1, false);
  shape.absarc(0, 0, r0, a1 - gap / r0, a0 + gap / r0, true);
  shape.closePath();
  const geom = new THREE.ExtrudeGeometry(shape, {
    depth,
    bevelEnabled: true,
    bevelThickness: 0.035,
    bevelSize: 0.025,
    bevelSegments: 4,
    curveSegments: Math.max(6, Math.ceil(((a1 - a0) / TAU) * 120)),
  });
  geom.rotateX(-Math.PI / 2); // extrude upwards (+Y); shape (x, y) → world (x, -y)
  return geom;
}

function Slice({
  spec,
  ring,
  index,
  active,
  reduced,
  labelAt,
  onHover,
  onSelect,
  anchors,
}: {
  anchors: RefObject<Map<string, THREE.Object3D>>;
  spec: SliceSpec;
  ring: typeof OUTER;
  index: number;
  active: boolean;
  reduced: boolean;
  labelAt: number;
  onHover: (symbol: string | null) => void;
  onSelect: (symbol: string) => void;
}) {
  const geom = useMemo(() => ringSegment(ring.r0, ring.r1, spec.a0, spec.a1, ring.depth, ring.gap), [ring, spec.a0, spec.a1]);
  useEffect(() => () => geom.dispose(), [geom]);

  // One orchestrated entrance (slices rise in sequence) and a lift that answers hover/selection.
  const grow = useSpring({ from: { s: 0.001 }, to: { s: 1 }, delay: reduced ? 0 : 250 + index * 55, immediate: reduced, config: { tension: 170, friction: 18 } });
  const lift = useSpring({ y: active ? 0.28 : 0, immediate: reduced, config: { tension: 260, friction: 20 } });

  const m = (spec.a0 + spec.a1) / 2;
  const r = ring.r0 + (ring.r1 - ring.r0) * labelAt;
  const interactive = !!spec.symbol;

  return (
    <animated.group position-y={lift.y} scale-y={grow.s}>
      <mesh
        geometry={geom}
        onPointerOver={(e: ThreeEvent<PointerEvent>) => {
          if (!interactive) return;
          e.stopPropagation();
          onHover(spec.symbol!);
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          if (!interactive) return;
          onHover(null);
          document.body.style.cursor = "";
        }}
        onClick={(e: ThreeEvent<MouseEvent>) => {
          if (!interactive || e.delta > 4) return; // a drag is not a click
          e.stopPropagation();
          onSelect(spec.symbol!);
        }}
      >
        <meshPhysicalMaterial
          color={spec.color}
          roughness={0.34}
          metalness={0.04}
          clearcoat={0.85}
          clearcoatRoughness={0.22}
          emissive={spec.color}
          emissiveIntensity={active ? 0.22 : 0}
        />
      </mesh>
      {spec.label && (
        <object3D
          position={[r * Math.cos(m), ring.depth + 0.08, -r * Math.sin(m)]}
          ref={(o) => {
            if (o) anchors.current.set(spec.key, o);
            else anchors.current.delete(spec.key);
          }}
        />
      )}
    </animated.group>
  );
}

/** Relative luminance of "rgb(r, g, b)" or "#rrggbb", 0–1. */
function luminance(c: string) {
  const [r, g, b] = c.startsWith("#") ? [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)) : c.match(/\d+/g)!.map(Number);
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
}

/** Projects each label anchor to screen space every rendered frame and moves its DOM label there. */
function LabelProjector({ anchors, labels }: { anchors: RefObject<Map<string, THREE.Object3D>>; labels: RefObject<Map<string, HTMLSpanElement>> }) {
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    anchors.current.forEach((obj, key) => {
      const el = labels.current.get(key);
      if (!el) return;
      obj.getWorldPosition(v).project(camera);
      el.style.transform = `translate(${((v.x + 1) / 2) * size.width}px, ${((1 - v.y) / 2) * size.height}px) translate(-50%, -50%)`;
      el.style.opacity = obj.parent && obj.parent.scale.y > 0.6 ? "1" : "0"; // appear once the slice has risen
    });
  });
  return null;
}

/** Soft floor shadow: a radial-gradient texture that fades to nothing, so no edge shows on the page. */
function SoftShadow() {
  const texture = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    const g = c.getContext("2d")!;
    const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
    grad.addColorStop(0, "rgba(20,22,26,0.38)");
    grad.addColorStop(0.55, "rgba(20,22,26,0.22)");
    grad.addColorStop(1, "rgba(20,22,26,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, 256, 256);
    return new THREE.CanvasTexture(c);
  }, []);
  useEffect(() => () => texture.dispose(), [texture]);
  return (
    <mesh rotation-x={-Math.PI / 2} position={[0, -0.03, -0.2]}>
      <planeGeometry args={[7.8, 7.8]} />
      <meshBasicMaterial map={texture} transparent depthWrite={false} />
    </mesh>
  );
}

function Studio() {
  return (
    <Environment resolution={256} environmentIntensity={0.9}>
      <group rotation={[-Math.PI / 3, 0, 0]}>
        <Lightformer form="rect" intensity={3} position={[0, 6, -6]} scale={[12, 4, 1]} />
        <Lightformer form="rect" intensity={1.2} rotation-x={Math.PI / 2} position={[0, 9, 0]} scale={[10, 10, 1]} />
        <Lightformer form="circle" intensity={2.2} rotation-y={Math.PI / 2} position={[-6, 2, 0]} scale={4} />
        <Lightformer form="circle" intensity={1.4} rotation-y={-Math.PI / 2} position={[6, 2, 2]} scale={3} />
        <Lightformer form="ring" color="#b9c8ff" intensity={2} position={[0, 3, 7]} scale={6} />
      </group>
    </Environment>
  );
}

export function Donut3DGL({
  allocation,
  selected,
  onSelect,
}: {
  allocation: Allocation;
  selected: string | null;
  onSelect: (symbol: string) => void;
}) {
  const [hover, setHover] = useState<string | null>(null);
  const [reduced] = useState(() => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  useEffect(() => () => void (document.body.style.cursor = ""), []);

  const { outer, inner, holdings } = useMemo(() => {
    const holdings = allocation.slices.flatMap((s) => s.holdings.map((h, i) => ({ ...h, assetClass: s.assetClass as AssetClass, i })));
    // Start at the back (θ = π/2 → world −z) and run clockwise as seen from above.
    const arcs = (weights: number[]) => {
      const out: [number, number][] = [];
      let end = Math.PI / 2;
      for (const w of weights) {
        const span = (w / 100) * TAU;
        out.push([end - span, end]);
        end -= span;
      }
      return out;
    };
    const oa = arcs(holdings.map((h) => h.weightPct));
    const outer: SliceSpec[] = holdings.map((h, k) => ({
      key: h.symbol,
      symbol: h.symbol,
      a0: oa[k][0],
      a1: oa[k][1],
      color: shade(CLASS_COLOR[h.assetClass], Math.min(0.45, h.i * 0.11)),
      label: h.weightPct >= 4 ? short(h.symbol) : null,
    }));
    const ia = arcs(allocation.slices.map((s) => s.weightPct));
    const inner: SliceSpec[] = allocation.slices.map((s, k) => ({
      key: s.assetClass,
      a0: ia[k][0],
      a1: ia[k][1],
      color: CLASS_COLOR[s.assetClass],
      label: s.weightPct >= 5 ? weight(s.weightPct) : null,
    }));
    return { outer, inner, holdings };
  }, [allocation]);

  const active = hover ?? selected;
  const shown = holdings.find((h) => h.symbol === active);
  const anchors = useRef(new Map<string, THREE.Object3D>());
  const labelEls = useRef(new Map<string, HTMLSpanElement>());

  return (
    <figure data-testid="donut-3d" data-renderer="webgl" className="relative m-0">
      <div
        role="img"
        aria-label={`Allocation: ${allocation.slices.map((s) => `${CLASS_LABEL[s.assetClass]} ${s.weightPct}%`).join(", ")}`}
        className="relative aspect-[6/4.1] w-full cursor-grab active:cursor-grabbing"
      >
        <div aria-hidden className="pointer-events-none absolute inset-0 z-10 overflow-hidden">
          {[...outer, ...inner]
            .filter((s) => s.label)
            .map((s) => (
              <span
                key={s.key}
                ref={(el) => {
                  if (el) labelEls.current.set(s.key, el);
                  else labelEls.current.delete(s.key);
                }}
                className="absolute top-0 left-0 whitespace-nowrap font-bold opacity-0 transition-opacity duration-300"
                style={
                  luminance(s.color) > 0.62
                    ? { fontSize: s.symbol ? 13 : 15, color: "#14161a" }
                    : { fontSize: s.symbol ? 13 : 15, color: "#fff", textShadow: "0 1px 2px rgba(20,22,26,.55)" }
                }
              >
                {s.label}
              </span>
            ))}
        </div>
        <Canvas flat frameloop="demand" dpr={[1, 2]} camera={{ position: [0, 5.2, 7.4], fov: 34 }} onCreated={({ camera }) => camera.lookAt(0, -0.35, 0)} gl={{ antialias: false, powerPreference: "high-performance" }}>
          <color attach="background" args={[BG]} />
          <ambientLight intensity={0.45} />
          <directionalLight position={[2, 10, 4]} intensity={1.15} />
          <Studio />
          <PresentationControls global cursor={false} snap={false} rotation={[0, 0.35, 0]} polar={[-0.25, 0.3]} azimuth={[-Infinity, Infinity]} damping={0.3}>
            <group>
              {outer.map((s, i) => (
                <Slice key={s.key} spec={s} ring={OUTER} index={i} active={s.symbol === active} reduced={reduced} labelAt={0.62} onHover={setHover} onSelect={onSelect} anchors={anchors} />
              ))}
              {inner.map((s, i) => (
                <Slice key={s.key} spec={s} ring={INNER} index={outer.length + i} active={false} reduced={reduced} labelAt={0.55} onHover={setHover} onSelect={onSelect} anchors={anchors} />
              ))}
            </group>
          </PresentationControls>
          <LabelProjector anchors={anchors} labels={labelEls} />
          <SoftShadow />
          <EffectComposer multisampling={4}>
            <N8AO aoRadius={0.55} intensity={1.6} distanceFalloff={0.6} color="#14161a" halfRes />
          </EffectComposer>
        </Canvas>
      </div>
      {/* Text alternative for screen readers (and a stable hook for tests). */}
      <ul className="sr-only">
        {holdings.map((h) => (
          <li key={h.symbol} data-symbol={h.symbol}>
            {h.symbol} {h.name} {weight(h.weightPct)}
          </li>
        ))}
      </ul>
      <figcaption className="mt-1 min-h-6 text-center text-[15px] text-slate">
        {shown ? (
          <>
            <b className="text-ink">{shown.symbol}</b> {shown.name}, {weight(shown.weightPct)}, {usd(shown.amountUsd)}
          </>
        ) : (
          "Drag to turn. Select a holding to see why it's there."
        )}
      </figcaption>
    </figure>
  );
}
