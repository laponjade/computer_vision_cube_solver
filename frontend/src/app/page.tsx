import Link from "next/link";
import Image from "next/image";

import { HomeHeroAnimation } from "@/components/HomeHeroAnimation";

export default function Home() {
  return (
    <div className="min-h-[calc(100vh-0)] bg-zinc-50 text-zinc-900 dark:bg-black dark:text-zinc-50">
      <div className="mx-auto max-w-6xl px-4 py-10 sm:py-14">
        <header className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex items-center">
              <Image
                src="/logo.png"
                alt="CUBE.AI logo"
                width={132}
                height={40}
                priority
                className="h-14 w-auto"
              />
            </div>
            <div>
              <p className="text-sm font-semibold leading-tight">Cube Solver</p>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Computer vision Rubik&apos;s capture + solve
              </p>
            </div>
          </div>
          <nav className="flex items-center gap-3">
            <Link
              href="/capture"
              className="rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm font-semibold text-zinc-900 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-50 dark:hover:bg-zinc-900"
            >
              Open capture
            </Link>
          </nav>
        </header>

        <main className="mt-12 grid gap-10 lg:grid-cols-2 lg:items-center">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-zinc-200 bg-white px-3 py-1 text-xs font-medium text-zinc-700 shadow-sm dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200">
              <span className="h-2 w-2 rounded-full bg-emerald-500" />
              Live camera capture
              <span className="text-zinc-400">·</span>6 faces
              <span className="text-zinc-400">·</span>
              Solve moves
            </div>

            <h1 className="mt-5 text-balance text-4xl font-semibold tracking-tight sm:text-5xl">
              Scan a Rubik&apos;s cube with your camera and get the solve steps.
            </h1>
            <p className="mt-4 max-w-xl text-pretty text-base leading-7 text-zinc-600 dark:text-zinc-300">
              Capture each face (U → R → F → D → L → B). We validate the colors,
              build the cube state, then return an algorithm you can follow
              step-by-step.
            </p>

            <div className="mt-7 flex flex-wrap items-center gap-3">
              <Link
                href="/capture"
                className="inline-flex items-center justify-center rounded-xl bg-zinc-900 px-5 py-3 text-sm font-semibold text-white shadow-sm hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
              >
                Demo
              </Link>
              <Link
                href="/capture/single"
                className="inline-flex items-center justify-center rounded-xl border border-zinc-200 bg-white px-5 py-3 text-sm font-semibold text-zinc-900 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-50 dark:hover:bg-zinc-900"
              >
                Single-face debug
              </Link>
            </div>

            <dl className="mt-10 grid gap-4 sm:grid-cols-2">
              <div className="rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
                <dt className="text-sm font-semibold">Guided capture</dt>
                <dd className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
                  Step-by-step face order with a live preview net.
                </dd>
              </div>
              <div className="rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
                <dt className="text-sm font-semibold">Solve output</dt>
                <dd className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
                  Returns a move sequence you can follow.
                </dd>
              </div>
            </dl>
          </div>

          <div className="relative">
            <div className="absolute -inset-6 -z-10 rounded-[32px] bg-linear-to-br from-yellow-300/35 via-red-500/15 to-blue-500/25 blur-2xl dark:from-yellow-300/20 dark:via-red-500/10 dark:to-blue-500/15" />

            <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
              <div className="flex items-center justify-between">
                <p className="text-sm font-semibold">Preview</p>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  Demo animation
                </p>
              </div>

              <div className="mt-6 flex items-center justify-center">
                <HomeHeroAnimation />
              </div>

              <div className="mt-6 grid gap-3 rounded-xl bg-zinc-50 p-4 dark:bg-zinc-900/40">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold">Demo flow</span>
                  <span className="text-zinc-500 dark:text-zinc-400">
                    ~30–60s
                  </span>
                </div>
                <div className="grid gap-2 text-sm text-zinc-700 dark:text-zinc-200">
                  <div className="flex items-center justify-between">
                    <span>1) Start camera</span>
                    <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
                      /capture
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>2) Capture 6 faces</span>
                    <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
                      U→R→F→D→L→B
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>3) Solve</span>
                    <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
                      moves
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>

        <footer className="mt-14 flex flex-wrap items-center justify-between gap-3 border-t border-zinc-200 pt-6 text-xs text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
          <p>Built for fast iteration on vision + UX.</p>
          <p className="font-mono">/capture</p>
        </footer>
      </div>
    </div>
  );
}
