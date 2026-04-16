const apiProxyTarget = process.env.AUTODERM_API_PROXY_TARGET ?? "http://127.0.0.1:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${apiProxyTarget}/api/:path*`,
      },
      {
        source: "/healthz",
        destination: `${apiProxyTarget}/healthz`,
      },
    ];
  },
};

export default nextConfig;
