import type { NextConfig } from "next";

const internalApiUrl = process.env.RAG_API_INTERNAL_URL;

const nextConfig: NextConfig = {
  output: "standalone",
  async headers() {
    const noStore = [{ key: "Cache-Control", value: "no-store, max-age=0, must-revalidate" }];
    return [
      { source: "/", headers: noStore },
      { source: "/login", headers: noStore },
      { source: "/chat/:path*", headers: noStore },
    ];
  },
  async rewrites() {
    if (!internalApiUrl) return [];
    return [
      {
        source: "/api/:path*",
        destination: `${internalApiUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
