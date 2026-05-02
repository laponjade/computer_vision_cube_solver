"use client";

import Link from "next/link";
import type { CSSProperties } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  analyzeFrame,
  apiBaseUrl,
  getHealth,
  type FaceGridResult,
} from "@/lib/api";

/** Map bbox in video pixel space to overlay coords when `<video>` uses `object-contain`. */
function bboxOverlayStyle(
  bbox: [number, number, number, number],
  vw: number,
  vh: number,
  cw: number,
  ch: number,
): CSSProperties | null {
  if (!vw || !vh || !cw || !ch) return null;
  const scale = Math.min(cw / vw, ch / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;
  return {
    left: ox + bbox[0] * scale,
    top: oy + bbox[1] * scale,
    width: (bbox[2] - bbox[0]) * scale,
    height: (bbox[3] - bbox[1]) * scale,
  };
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

const colorClass: Record<string, string> = {
  white: "bg-zinc-100 text-zinc-900 ring-zinc-300",
  yellow: "bg-yellow-300 text-yellow-950 ring-yellow-500",
  red: "bg-red-600 text-white ring-red-800",
  orange: "bg-orange-500 text-white ring-orange-700",
  blue: "bg-blue-700 text-white ring-blue-900",
  green: "bg-green-600 text-white ring-green-800",
};

export default function CaptureSinglePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const visionBusyRef = useRef(false);

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FaceGridResult | null>(null);
  const [visionPending, setVisionPending] = useState(false);
  const [videoLayout, setVideoLayout] = useState({
    vw: 0,
    vh: 0,
    cw: 0,
    ch: 0,
  });

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setRunning(false);
  }, []);

  const refreshVideoLayout = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    setVideoLayout({
      vw: v.videoWidth,
      vh: v.videoHeight,
      cw: v.clientWidth,
      ch: v.clientHeight,
    });
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    setResult(null);
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
      requestAnimationFrame(() => refreshVideoLayout());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not open camera");
    }
  }, [refreshVideoLayout]);

  const captureAndSend = useCallback(async () => {
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
              setResult(json);
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

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !running) return;
    const ro = new ResizeObserver(() => refreshVideoLayout());
    ro.observe(v);
    return () => ro.disconnect();
  }, [running, refreshVideoLayout]);

  const bboxStyle =
    result?.face_detected && result.bbox_xyxy
      ? bboxOverlayStyle(
          result.bbox_xyxy,
          videoLayout.vw,
          videoLayout.vh,
          videoLayout.cw,
          videoLayout.ch,
        )
      : null;

  return (
    <div className="mx-auto flex min-h-full max-w-4xl flex-col gap-6 px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">
            Single-face capture (debug)
          </h1>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            Snapshot-based camera capture for debugging model output and crops.
          </p>
        </div>
        <Link
          href="/capture"
          className="text-sm font-medium text-blue-600 hover:underline dark:text-blue-400"
        >
          Back to 6-face capture
        </Link>
      </div>

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
              onClick={() => void captureAndSend()}
              disabled={visionPending}
              className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200 dark:disabled:opacity-50"
            >
              {visionPending ? "Processing…" : "Capture & analyze"}
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

      {error ? (
        <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          {error}
        </p>
      ) : null}

      <div className="grid gap-8 lg:grid-cols-2">
        <div className="relative overflow-hidden rounded-xl border border-zinc-200 bg-black/5 dark:border-zinc-700">
          <video
            ref={videoRef}
            className="aspect-video w-full object-contain"
            playsInline
            muted
            onLoadedMetadata={refreshVideoLayout}
            onLoadedData={refreshVideoLayout}
            onPlaying={refreshVideoLayout}
            onResize={refreshVideoLayout}
          />
          {running ? (
            <>
              <div
                className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center"
                aria-hidden
              >
                <div
                  className="relative aspect-square rounded-md border-2 border-white/85 shadow-[0_0_0_1px_rgba(0,0,0,0.45),0_2px_14px_rgba(0,0,0,0.35)]"
                  style={{ width: "clamp(68px, 12vmin, 94px)" }}
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
              <p className="pointer-events-none absolute bottom-2 left-2 right-2 z-10 rounded bg-black/50 px-2 py-1 text-center text-xs font-medium text-white">
                Align one cube face inside the square, then Capture & analyze
              </p>
            </>
          ) : null}
          {bboxStyle ? (
            <div
              className="pointer-events-none absolute z-20 border-2 border-emerald-400 shadow-[0_0_0_1px_rgba(0,0,0,0.3)]"
              style={bboxStyle}
            />
          ) : null}
          <canvas ref={canvasRef} className="hidden" />
        </div>

        <div className="flex flex-col gap-4">
          <div className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-700">
            <h2 className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
              Status
            </h2>
            {running && visionPending ? (
              <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
                Sending snapshot to API…
              </p>
            ) : null}
            {result ? (
              <dl className="mt-2 space-y-1 text-sm text-zinc-800 dark:text-zinc-200">
                <div className="flex justify-between gap-4">
                  <dt>API</dt>
                  <dd
                    className="max-w-48 truncate font-mono text-xs"
                    title={apiBaseUrl()}
                  >
                    {apiBaseUrl()}
                  </dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt>Face detected</dt>
                  <dd className="font-mono">
                    {result.face_detected ? "yes" : "no"}
                  </dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt>Face confidence</dt>
                  <dd className="font-mono">{result.face_confidence}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt>Method</dt>
                  <dd className="font-mono">{result.detection_method}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt>Inference</dt>
                  <dd className="font-mono">{result.inference_ms} ms</dd>
                </div>
              </dl>
            ) : (
              <p className="mt-2 text-sm text-zinc-500">
                {running
                  ? "Click Capture & analyze to process the current video frame."
                  : "Start the camera, then capture one frame at a time."}
              </p>
            )}
            {result?.message ? (
              <p className="mt-2 text-xs text-amber-700 dark:text-amber-300">
                {result.message}
              </p>
            ) : null}
          </div>

          {result?.face_detected && result.grid.length > 0 ? (
            <div className="flex flex-wrap gap-6">
              <div>
                <h2 className="mb-2 text-sm font-medium text-zinc-500 dark:text-zinc-400">
                  Sticker crops
                </h2>
                <div className="inline-grid grid-cols-3 gap-2">
                  {result.grid.flatMap((row, ri) =>
                    row.map((cell, ci) => (
                      <div
                        key={`crop-${ri}-${ci}`}
                        className="flex h-16 w-16 items-center justify-center overflow-hidden rounded-lg ring-1 ring-inset ring-zinc-300 dark:ring-zinc-700"
                      >
                        {cell.crop_jpeg_b64 ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img
                            alt={`sticker crop ${ri}-${ci}`}
                            src={`data:image/jpeg;base64,${cell.crop_jpeg_b64}`}
                            className="h-full w-full object-cover"
                          />
                        ) : (
                          <span className="text-xs text-zinc-500">—</span>
                        )}
                      </div>
                    )),
                  )}
                </div>
              </div>

              <div>
                <h2 className="mb-2 text-sm font-medium text-zinc-500 dark:text-zinc-400">
                  {result.dl_sticker_color_source === "cls"
                    ? "YOLO-cls (confidence)"
                    : "HSV prediction (confidence)"}
                </h2>
                <div className="inline-grid grid-cols-3 gap-2">
                  {result.grid.flatMap((row, ri) =>
                    row.map((cell, ci) => (
                      <div
                        key={`pred-${ri}-${ci}`}
                        className={`flex h-16 w-16 flex-col items-center justify-center rounded-lg text-xs font-medium ring-1 ring-inset ${
                          cell.color
                            ? colorClass[cell.color]
                            : "bg-zinc-200 text-zinc-900"
                        }`}
                        title={
                          result.dl_sticker_color_source === "cls"
                            ? (cell.cls_label_raw ?? undefined)
                            : undefined
                        }
                      >
                        <span className="capitalize">
                          {cell.color ?? "unknown"}
                        </span>
                        <span className="opacity-80">
                          {cell.confidence.toFixed(2)}
                        </span>
                      </div>
                    )),
                  )}
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

