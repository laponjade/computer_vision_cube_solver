"use client";

import { useEffect, useState } from "react";
import Lottie from "lottie-react";

async function loadAnimation(): Promise<object> {
  const res = await fetch(
    `/home_page/${encodeURIComponent("Geometric Cube 3D Animation.json")}`,
  );
  if (!res.ok) throw new Error("Could not load home page animation");
  return (await res.json()) as object;
}

export function HomeHeroAnimation() {
  const [data, setData] = useState<object | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void loadAnimation()
      .then((json) => {
        if (!cancelled) setData(json);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Could not load animation");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className="flex h-[232px] w-full items-center justify-center rounded-xl bg-zinc-100 text-sm text-zinc-600 dark:bg-zinc-900/40 dark:text-zinc-300">
        {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex h-[232px] w-full items-center justify-center rounded-xl bg-zinc-100 text-sm text-zinc-600 dark:bg-zinc-900/40 dark:text-zinc-300">
        Loading animation…
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-md">
      <Lottie animationData={data} loop autoplay />
    </div>
  );
}

