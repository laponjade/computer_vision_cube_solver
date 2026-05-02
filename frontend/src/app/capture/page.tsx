"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  analyzeFrame,
  apiBaseUrl,
  getHealth,
  type FaceGridResult,
  type StickerCell,
  solveCube,
} from "@/lib/api";
import { SolutionMoveAnimator } from "@/components/SolutionMoveAnimator";

type FaceName = "U" | "R" | "F" | "D" | "L" | "B";

const FACE_ORDER: Array<{ id: FaceName; title: string; hint: string }> = [
  { id: "U", title: "Up", hint: "Hold the cube so the top face is centered." },
  { id: "R", title: "Right", hint: "Rotate cube to show the right face." },
  { id: "F", title: "Front", hint: "Show the front face straight-on." },
  { id: "D", title: "Down", hint: "Show the bottom face (flip the cube)." },
  { id: "L", title: "Left", hint: "Rotate cube to show the left face." },
  { id: "B", title: "Back", hint: "Rotate cube to show the back face." },
];

const colorClass: Record<string, string> = {
  white: "bg-zinc-100 text-zinc-900 ring-zinc-300",
  yellow: "bg-yellow-300 text-yellow-950 ring-yellow-500",
  red: "bg-red-600 text-white ring-red-800",
  orange: "bg-orange-500 text-white ring-orange-700",
  blue: "bg-blue-700 text-white ring-blue-900",
  green: "bg-green-600 text-white ring-green-800",
};

const DEFAULT_CENTER_COLOR: Record<
  FaceName,
  Exclude<StickerCell["color"], null>
> = {
  U: "yellow",
  F: "red",
  R: "green",
  D: "white",
  L: "blue",
  B: "orange",
};

function emptyCell(color: StickerCell["color"]): StickerCell {
  return { color, confidence: color ? 1 : 0, bgr_mean: [0, 0, 0] };
}

function emptyGridWithCenter(
  center: Exclude<StickerCell["color"], null>,
): StickerCell[][] {
  return [
    [emptyCell(null), emptyCell(null), emptyCell(null)],
    [emptyCell(null), emptyCell(center), emptyCell(null)],
    [emptyCell(null), emptyCell(null), emptyCell(null)],
  ];
}

async function openBestCamera(): Promise<MediaStream> {
  const attempts: MediaStreamConstraints[] = [
    {
      video: {
        facingMode: "environment",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    },
    {
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    },
    { video: { width: { ideal: 960 }, height: { ideal: 540 } }, audio: false },
  ];
  let last: unknown;
  for (const c of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(c);
    } catch (e) {
      last = e;
    }
  }
  throw last;
}

function centerColor(grid: StickerCell[][]): StickerCell["color"] {
  return grid?.[1]?.[1]?.color ?? null;
}

function rotateGridCW<T>(grid: T[][]): T[][] {
  // 3x3 rotation (clockwise)
  return [
    [grid[2][0], grid[1][0], grid[0][0]],
    [grid[2][1], grid[1][1], grid[0][1]],
    [grid[2][2], grid[1][2], grid[0][2]],
  ];
}

function parseApiDetail(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err);
  // FastAPI often returns JSON text like {"detail":"..."} when thrown as Error(text).
  try {
    const j = JSON.parse(msg) as { detail?: unknown };
    if (j && typeof j === "object" && "detail" in j) {
      const d = (j as { detail?: unknown }).detail;
      if (typeof d === "string") return d;
    }
  } catch {
    // ignore
  }
  return msg;
}

function isGoodFace(r: FaceGridResult | null): { ok: boolean; reason: string } {
  if (!r) return { ok: false, reason: "No capture yet" };
  if (!r.face_detected) return { ok: false, reason: "No face detected" };
  if (!r.grid?.length) return { ok: false, reason: "No 3×3 grid returned" };
  const cells = r.grid.flat();
  const unknown = cells.filter((c) => !c.color).length;
  if (unknown > 0) return { ok: false, reason: `${unknown} unknown stickers` };
  const cc = centerColor(r.grid);
  if (!cc) return { ok: false, reason: "Center sticker unknown" };
  if ((r.face_confidence ?? 0) < 0.15)
    return { ok: false, reason: "Low face confidence" };
  return { ok: true, reason: "OK" };
}

function FaceMiniGrid({
  title,
  grid,
}: {
  title: string;
  grid: StickerCell[][];
}) {
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="text-[11px] font-semibold text-zinc-700 dark:text-zinc-200">
        {title}
      </div>
      <div className="grid grid-cols-3 gap-1 rounded-md bg-zinc-300/60 p-1 dark:bg-zinc-800/60">
        {grid.flatMap((row, ri) =>
          row.map((cell, ci) => (
            <div
              key={`${ri}-${ci}`}
              className={`h-10 w-10 rounded-[4px] ring-1 ring-inset ${
                cell.color
                  ? colorClass[cell.color]
                  : "bg-zinc-200 text-zinc-900 ring-zinc-400/60"
              }`}
              title={cell.color ?? "unknown"}
            />
          )),
        )}
      </div>
    </div>
  );
}

export default function CapturePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const visionBusyRef = useRef(false);

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [visionPending, setVisionPending] = useState(false);
  const [solvePending, setSolvePending] = useState(false);
  const [activeFaceIdx, setActiveFaceIdx] = useState(0);
  const [lastResult, setLastResult] = useState<FaceGridResult | null>(null);
  const [previewTurnsCW, setPreviewTurnsCW] = useState(0);
  const [faces, setFaces] = useState<Partial<Record<FaceName, FaceGridResult>>>(
    {},
  );
  const [solution, setSolution] = useState<string | null>(null);
  const [faceletState, setFaceletState] = useState<string | null>(null);

  const active = FACE_ORDER[Math.min(5, Math.max(0, activeFaceIdx))];

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setRunning(false);
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    setSolution(null);
    setFaceletState(null);
    try {
      await getHealth();
    } catch {
      setError(
        `Cannot reach the API at ${apiBaseUrl()}. Start the backend (uvicorn on port 8000) and refresh.`,
      );
      return;
    }
    try {
      const stream = await openBestCamera();
      streamRef.current = stream;
      const v = videoRef.current;
      if (!v) return;
      v.srcObject = stream;
      await v.play();
      setRunning(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not open camera");
    }
  }, []);

  const captureAndAnalyze = useCallback(async () => {
    if (visionBusyRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;

    visionBusyRef.current = true;
    setVisionPending(true);
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      visionBusyRef.current = false;
      setVisionPending(false);
      return;
    }
    ctx.drawImage(video, 0, 0, w, h);

    try {
      await new Promise<void>((resolve, reject) => {
        canvas.toBlob(
          async (blob) => {
            if (!blob) {
              reject(new Error("toBlob failed"));
              return;
            }
            try {
              const json = await analyzeFrame(blob);
              setLastResult(json);
              setPreviewTurnsCW(0);
              setError(null);
              resolve();
            } catch (err) {
              reject(err instanceof Error ? err : new Error(String(err)));
            }
          },
          "image/jpeg",
          0.85,
        );
      });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      visionBusyRef.current = false;
      setVisionPending(false);
    }
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const savedForActive = faces[active.id] ?? null;
  const draft = lastResult ?? savedForActive;
  const isEditingSaved = Boolean(!lastResult && savedForActive);
  const rotatedPreview = useMemo(() => {
    if (!draft?.grid?.length) return draft;
    const t = ((previewTurnsCW % 4) + 4) % 4;
    if (t === 0) return draft;
    let g = draft.grid;
    for (let i = 0; i < t; i += 1) g = rotateGridCW(g);
    return { ...draft, grid: g };
  }, [draft, previewTurnsCW]);
  const rotatedQuality = useMemo(
    () => isGoodFace(rotatedPreview),
    [rotatedPreview],
  );
  const capturedCount = Object.keys(faces).length;
  const allCaptured = capturedCount === 6;

  const cubeNet = useMemo(() => {
    const result: Record<FaceName, StickerCell[][]> = {
      U: emptyGridWithCenter(DEFAULT_CENTER_COLOR.U),
      R: emptyGridWithCenter(DEFAULT_CENTER_COLOR.R),
      F: emptyGridWithCenter(DEFAULT_CENTER_COLOR.F),
      D: emptyGridWithCenter(DEFAULT_CENTER_COLOR.D),
      L: emptyGridWithCenter(DEFAULT_CENTER_COLOR.L),
      B: emptyGridWithCenter(DEFAULT_CENTER_COLOR.B),
    };

    for (const f of FACE_ORDER) {
      const saved = faces[f.id];
      if (saved?.grid?.length) result[f.id] = saved.grid;
    }

    // While predicting/capturing, fill ONLY the active face with the predicted stickers.
    if (rotatedPreview?.grid?.length) {
      result[active.id] = rotatedPreview.grid;
    }

    return result;
  }, [active.id, faces, rotatedPreview]);

  const acceptFace = useCallback(() => {
    if (!rotatedPreview || !rotatedQuality.ok) return;
    const wasCaptured = Boolean(faces[active.id]);
    setFaces((prev) => ({ ...prev, [active.id]: rotatedPreview }));
    setLastResult(null);
    setPreviewTurnsCW(0);
    setSolution(null);
    setFaceletState(null);
    // Only auto-advance when capturing a new face.
    if (!wasCaptured) setActiveFaceIdx((i) => Math.min(5, i + 1));
  }, [active.id, faces, rotatedPreview, rotatedQuality.ok]);

  const selectFace = useCallback((idx: number) => {
    setActiveFaceIdx(idx);
    setLastResult(null);
    setPreviewTurnsCW(0);
    setSolution(null);
    setFaceletState(null);
  }, []);

  const clearPreview = useCallback(() => {
    setLastResult(null);
    setPreviewTurnsCW(0);
    setSolution(null);
    setFaceletState(null);
  }, []);

  const resetAll = useCallback(() => {
    setFaces({});
    setLastResult(null);
    setPreviewTurnsCW(0);
    setSolution(null);
    setFaceletState(null);
    setActiveFaceIdx(0);
  }, []);

  const solve = useCallback(async () => {
    setError(null);
    setSolution(null);
    setFaceletState(null);
    if (!allCaptured) {
      setError("Capture all 6 faces first.");
      return;
    }
    setSolvePending(true);
    try {
      const json = await solveCube({ faces });
      setFaceletState(json.state);
      setSolution(json.solution);
    } catch (e) {
      setError(parseApiDetail(e));
    } finally {
      setSolvePending(false);
    }
  }, [allCaptured, faces]);

  const faceBadges = useMemo(() => {
    const byId: Partial<Record<FaceName, { center: string | null }>> = {};
    for (const f of FACE_ORDER) {
      const r = faces[f.id];
      byId[f.id] = {
        center: r?.grid?.length ? (centerColor(r.grid) as string | null) : null,
      };
    }
    return byId;
  }, [faces]);

  return (
    <div className="mx-auto flex min-h-full max-w-5xl flex-col gap-6 px-4 py-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">
            6-face capture
          </h1>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            Capture each face once (U → R → F → D → L → B), then solve.
          </p>
          <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
            Debug model output?{" "}
            <Link
              href="/capture/single"
              className="text-blue-600 hover:underline dark:text-blue-400"
            >
              Single-face page
            </Link>
            .
          </p>
        </div>
        <Link
          href="/"
          className="text-sm font-medium text-blue-600 hover:underline dark:text-blue-400"
        >
          Back home
        </Link>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          {!running ? (
            <button
              type="button"
              onClick={() => void startCamera()}
              className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
            >
              Start camera
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={() => void captureAndAnalyze()}
                disabled={visionPending}
                className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200 dark:disabled:opacity-50"
              >
                {visionPending ? "Processing…" : "Capture"}
              </button>
              <button
                type="button"
                onClick={stopCamera}
                className="rounded-lg border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-800 hover:bg-zinc-50 dark:border-zinc-600 dark:text-zinc-100 dark:hover:bg-zinc-900"
              >
                Stop
              </button>
            </>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={resetAll}
            className="rounded-lg border border-zinc-300 px-3 py-2 text-sm font-medium text-zinc-800 hover:bg-zinc-50 dark:border-zinc-600 dark:text-zinc-100 dark:hover:bg-zinc-900"
          >
            Reset
          </button>
          <button
            type="button"
            onClick={() => void solve()}
            disabled={!allCaptured || solvePending}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {solvePending ? "Solving…" : "Solve cube"}
          </button>
        </div>
      </div>

      {error ? (
        <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          {error}
        </p>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-700">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                Step {activeFaceIdx + 1} / 6
              </p>
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
                {active.title} face ({active.id})
              </h2>
              <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
                {active.hint}
              </p>
              {faces[active.id] ? (
                <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                  This face is already saved — rotate it if needed, then{" "}
                  <span className="font-semibold">Accept face</span> to update
                  it.
                </p>
              ) : null}
            </div>
            <div className="text-right">
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Captured
              </p>
              <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                {capturedCount} / 6
              </p>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-6 gap-2">
            {FACE_ORDER.map((f, i) => {
              const done = Boolean(faces[f.id]);
              const isActive = i === activeFaceIdx;
              const center = faceBadges[f.id]?.center;
              return (
                <button
                  key={f.id}
                  type="button"
                  onClick={() => selectFace(i)}
                  className={`rounded-lg border px-2 py-2 text-xs font-medium ${
                    isActive
                      ? "border-zinc-900 bg-zinc-900 text-white dark:border-zinc-100 dark:bg-zinc-100 dark:text-zinc-900"
                      : "border-zinc-200 bg-white text-zinc-800 hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100 dark:hover:bg-zinc-900"
                  }`}
                  title={
                    done
                      ? `Captured (center: ${center ?? "?"})`
                      : "Not captured yet"
                  }
                >
                  <div className="flex items-center justify-center gap-2">
                    <span>{f.id}</span>
                    {done ? (
                      <span
                        className={`inline-block h-2.5 w-2.5 rounded-full ring-1 ring-inset ${
                          center
                            ? colorClass[center]
                            : "bg-zinc-300 text-zinc-900 ring-zinc-400"
                        }`}
                      />
                    ) : null}
                  </div>
                </button>
              );
            })}
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={acceptFace}
              disabled={!rotatedQuality.ok}
              className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-semibold text-white hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
            >
              {isEditingSaved ? "Update face" : "Accept face"}
            </button>
            <button
              type="button"
              onClick={() => setPreviewTurnsCW((t) => (t + 3) % 4)}
              disabled={!draft?.grid?.length}
              className="rounded-lg border border-zinc-300 px-3 py-2 text-sm font-medium text-zinc-800 hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-600 dark:text-zinc-100 dark:hover:bg-zinc-900"
            >
              Rotate ⟲
            </button>
            <button
              type="button"
              onClick={() => setPreviewTurnsCW((t) => (t + 1) % 4)}
              disabled={!draft?.grid?.length}
              className="rounded-lg border border-zinc-300 px-3 py-2 text-sm font-medium text-zinc-800 hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-600 dark:text-zinc-100 dark:hover:bg-zinc-900"
            >
              Rotate ⟳
            </button>
            <button
              type="button"
              onClick={clearPreview}
              className="rounded-lg border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-800 hover:bg-zinc-50 dark:border-zinc-600 dark:text-zinc-100 dark:hover:bg-zinc-900"
            >
              {lastResult ? "Discard capture" : "Clear selection"}
            </button>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">
              Quality:{" "}
              <span
                className={
                  rotatedQuality.ok
                    ? "text-emerald-700 dark:text-emerald-300"
                    : "text-amber-700 dark:text-amber-300"
                }
              >
                {rotatedQuality.reason}
              </span>
            </p>
          </div>
        </div>

        <div className="relative overflow-hidden rounded-xl border border-zinc-200 bg-black/5 dark:border-zinc-700">
          <video
            ref={videoRef}
            className="aspect-video w-full object-contain"
            playsInline
            muted
          />
          {running ? (
            <>
              <div
                className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center"
                aria-hidden
              >
                <div
                  className="relative aspect-square rounded-md border-2 border-white/85 shadow-[0_0_0_1px_rgba(0,0,0,0.45),0_2px_14px_rgba(0,0,0,0.35)]"
                  style={{ width: "clamp(96px, 18vmin, 140px)" }}
                >
                  <div
                    className="absolute inset-0.5 rounded-sm"
                    style={{
                      backgroundImage: [
                        "linear-gradient(to right, rgba(255,255,255,0.7) 1px, transparent 1px)",
                        "linear-gradient(to bottom, rgba(255,255,255,0.7) 1px, transparent 1px)",
                        "linear-gradient(to right, rgba(0,0,0,0.4) 1px, transparent 1px)",
                        "linear-gradient(to bottom, rgba(0,0,0,0.4) 1px, transparent 1px)",
                      ].join(","),
                      backgroundSize:
                        "33.333333% 100%, 100% 33.333333%, 33.333333% 100%, 100% 33.333333%",
                      backgroundPosition: "0 0, 0 0, 1px 1px, 1px 1px",
                    }}
                  />
                </div>
              </div>
              <p className="pointer-events-none absolute bottom-2 left-2 right-2 z-10 rounded bg-black/55 px-2 py-1 text-center text-xs font-medium text-white">
                {active.title} ({active.id}) — keep the face flat and centered
              </p>
            </>
          ) : (
            <p className="absolute inset-0 flex items-center justify-center text-sm text-zinc-500">
              Start camera to begin
            </p>
          )}
          <canvas ref={canvasRef} className="hidden" />
        </div>
      </div>

      <div className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-700">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Cube preview
            </h2>
            <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
              When no face is scanned, only centers are shown (U yellow, F red,
              R green, D white, L blue, B orange). While predicting, only the
              active face fills in.
            </p>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            API: <span className="font-mono">{apiBaseUrl()}</span>
          </p>
        </div>

        <div className="mt-4 flex items-center justify-center">
          <div className="grid grid-cols-4 gap-3">
            <div />
            <FaceMiniGrid title="U" grid={cubeNet.U} />
            <div />
            <div />

            <FaceMiniGrid title="L" grid={cubeNet.L} />
            <FaceMiniGrid title="F" grid={cubeNet.F} />
            <FaceMiniGrid title="R" grid={cubeNet.R} />
            <FaceMiniGrid title="B" grid={cubeNet.B} />

            <div />
            <FaceMiniGrid title="D" grid={cubeNet.D} />
            <div />
            <div />
          </div>
        </div>

        {rotatedPreview?.grid?.length ? (
          <p className="mt-4 text-xs text-zinc-500 dark:text-zinc-400">
            Active face {active.id}: center{" "}
            <span className="font-mono">
              {centerColor(rotatedPreview.grid) ?? "unknown"}
            </span>{" "}
            · Face conf{" "}
            <span className="font-mono">{rotatedPreview.face_confidence}</span>{" "}
            · Inference{" "}
            <span className="font-mono">{rotatedPreview.inference_ms} ms</span>{" "}
            · Rot{" "}
            <span className="font-mono">
              {((previewTurnsCW % 4) + 4) % 4}×90°
            </span>
          </p>
        ) : (
          <p className="mt-4 text-xs text-zinc-500 dark:text-zinc-400">
            No prediction yet for active face {active.id}.
          </p>
        )}
      </div>

      {solution ? (
        <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-emerald-900 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-100">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold">Solution</h2>
              <p className="mt-2 font-mono text-sm">{solution}</p>
              {faceletState ? (
                <p className="mt-2 text-xs">
                  State: <span className="font-mono">{faceletState}</span>
                </p>
              ) : null}
            </div>
          </div>
          <SolutionMoveAnimator solution={solution} />

        </div>
      ) : null}
    </div>
  );
}
