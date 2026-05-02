"use client";

import { useEffect, useMemo, useState } from "react";
import Lottie from "lottie-react";

type MoveToken = "U" | "U'" | "R" | "R'" | "L" | "L'" | "B" | "B'" | "D" | "D'" | "F" | "F'";

const MOVE_TO_FILE: Record<MoveToken, string> = {
  // NOTE: These LottieFiles directions are opposite of move notation for some faces.
  // We map tokens to the animation that visually matches the expected turn direction.
  U: "Rubik's upper left rotation Lottie JSON animation.json",
  "U'": "Rubik's upper right rotation Lottie JSON animation.json",
  R: "Rubik's right up rotation Lottie JSON animation.json",
  "R'": "Rubik's right down rotation Lottie JSON animation.json",
  L: "Rubik's left down rotation Lottie JSON animation.json",
  "L'": "Rubik's left up rotation Lottie JSON animation.json",
  B: "Rubik's Back left side rotation Lottie JSON animation.json",
  "B'": "Rubik's Back right side rotation Lottie JSON animation.json",
  D: "Rubik's lower right rotation Lottie JSON animation.json",
  "D'": "Rubik's lower left rotation Lottie JSON animation.json",
  F: "Rubik's Front right side rotation Lottie JSON animation.json",
  "F'": "Rubik's Front left side rotation Lottie JSON animation.json",
};

function parseMoves(solution: string): MoveToken[] {
  const tokens = solution.trim().split(/\s+/).filter(Boolean);
  const out: MoveToken[] = [];

  for (const t of tokens) {
    const m = t.match(/^([URFDLB])([2']?)$/);
    if (!m) continue;
    const base = m[1] as "U" | "R" | "F" | "D" | "L" | "B";
    const suffix = m[2];
    if (suffix === "2") {
      out.push(base, base);
      continue;
    }
    out.push((suffix === "'" ? `${base}'` : base) as MoveToken);
  }
  return out;
}

async function loadAnimation(fileName: string): Promise<object> {
  const encoded = fileName
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
  const res = await fetch(`/rubik_animation/${encoded}`);
  if (!res.ok) throw new Error(`Failed to load animation: ${fileName}`);
  return (await res.json()) as object;
}

export function SolutionMoveAnimator({ solution }: { solution: string }) {
  const moves = useMemo(() => parseMoves(solution), [solution]);
  const [step, setStep] = useState(0);
  const [animData, setAnimData] = useState<object | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const currentMove = step < moves.length ? moves[step] : null;

  useEffect(() => {
    let cancelled = false;
    setAnimData(null);
    setLoadError(null);
    if (!currentMove) return;

    void loadAnimation(MOVE_TO_FILE[currentMove])
      .then((json) => {
        if (!cancelled) setAnimData(json);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setLoadError(e instanceof Error ? e.message : "Could not load animation");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [currentMove]);

  return (
    <div className="mt-4 grid gap-4 lg:grid-cols-[280px_1fr]">
      <div className="rounded-lg border border-emerald-300/60 bg-white/50 p-3 dark:border-emerald-700/70 dark:bg-emerald-950/30">
        <p className="text-xs font-semibold">Step by step</p>
        <p className="mt-1 text-xs opacity-80">
          Step {Math.min(step + 1, Math.max(1, moves.length))} / {moves.length}
        </p>
        <p className="mt-2 text-sm">
          Move:{" "}
          <span className="rounded bg-emerald-200/70 px-2 py-0.5 font-mono text-sm dark:bg-emerald-800/70">
            {currentMove ?? "Done"}
          </span>
        </p>

        <div className="mt-3 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => setStep((s) => Math.max(0, s - 1))}
            disabled={step === 0}
            className="rounded border border-emerald-400/70 px-3 py-1 text-xs disabled:cursor-not-allowed disabled:opacity-50"
          >
            Prev
          </button>
          <button
            type="button"
            onClick={() => setStep((s) => Math.min(moves.length, s + 1))}
            disabled={step >= moves.length}
            className="rounded bg-emerald-600 px-3 py-1 text-xs font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
          >
            Next
          </button>
          <button
            type="button"
            onClick={() => setStep(0)}
            className="rounded border border-emerald-400/70 px-3 py-1 text-xs"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="rounded-lg border border-emerald-300/60 bg-white/50 p-3 dark:border-emerald-700/70 dark:bg-emerald-950/30">
        {loadError ? (
          <p className="text-sm text-red-700 dark:text-red-300">{loadError}</p>
        ) : currentMove && animData ? (
          <div className="mx-auto w-full max-w-md">
            <Lottie animationData={animData} loop autoplay />
          </div>
        ) : (
          <p className="text-sm opacity-80">
            {currentMove ? "Loading animation..." : "All moves completed."}
          </p>
        )}
      </div>
    </div>
  );
}

