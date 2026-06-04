/** @type {import('next').NextConfig} */
const nextConfig = {
  // 开发模式下的 API 代理，避免跨域问题
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://backend:8000/api/:path*",
      },
    ];
  },
  // 允许使用外部图片（如 GitHub 头像等）
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
    ],
  },
};

module.exports = nextConfig;
