/**
 * Browser calls same-origin `/cube-api/*` in dev (rewritten to FastAPI) so LAN/mobile
 * works without setting NEXT_PUBLIC_API_URL. Override with NEXT_PUBLIC_API_URL when needed.
 */
export function apiBaseUrl(): string {
  const fromEnv = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (fromEnv) return fromEnv.replace(/\/$/, "");
  if (process.env.NODE_ENV === "development") return "/cube-api";
  return "http://127.0.0.1:8000";
}

export async function getHealth(): Promise<{ status: string }> {
  const res = await fetch(`${apiBaseUrl()}/health`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`);
  }
  return res.json() as Promise<{ status: string }>;
}

export type StickerCell = {
  color: "white" | "yellow" | "red" | "orange" | "blue" | "green" | null;
  confidence: number;
  bgr_mean: [number, number, number];
  crop_jpeg_b64?: string | null;
  /** YOLO-cls test head when VISION_DL_CLS_WEIGHTS is set (DL mode) */
  cls_color?: "white" | "yellow" | "red" | "orange" | "blue" | "green" | null;
  cls_confidence?: number;
  cls_label_raw?: string | null;
};

export type FaceGridResult = {
  face_detected: boolean;
  face_confidence: number;
  detection_method: "yolo" | "yolo_obb" | "contour_fallback";
  bbox_xyxy: [number, number, number, number] | null;
  grid: StickerCell[][];
  inference_ms: number;
  message: string | null;
  /** DL mode: "hsv" = middle grid is HSV; "cls" = classifier-only (VISION_MODEL_CLASSIFY_STRICT) */
  dl_sticker_color_source?: "hsv" | "cls" | null;
};

export async function analyzeFrame(image: Blob): Promise<FaceGridResult> {
  const form = new FormData();
  form.append("file", image, "frame.jpg");
  const res = await fetch(`${apiBaseUrl()}/vision/analyze-frame`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Vision request failed: ${res.status}`);
  }
  return res.json() as Promise<FaceGridResult>;
}

export type FaceName = "U" | "R" | "F" | "D" | "L" | "B";

export type SolveCubeRequest = {
  faces: Partial<Record<FaceName, FaceGridResult>>;
};

export type SolveCubeResponse = {
  state: string;
  solution: string;
  color_to_face: Record<string, FaceName>;
};

export async function solveCube(req: SolveCubeRequest): Promise<SolveCubeResponse> {
  const res = await fetch(`${apiBaseUrl()}/session/solve`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Solve request failed: ${res.status}`);
  }
  return res.json() as Promise<SolveCubeResponse>;
}
