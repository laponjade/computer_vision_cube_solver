import type { NextConfig } from "next";
import path from "path";

const backendOrigin = process.env.CUBE_BACKEND_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  turbopack: {
    // Ensure Turbopack resolves deps from /frontend, not repo root.
    root: path.resolve(__dirname),
  },
  async rewrites() {
    return [
      {
        source: "/cube-api/:path*",
        destination: `${backendOrigin.replace(/\/$/, "")}/:path*`,
      },
    ];
  },
};

export default nextConfig;
