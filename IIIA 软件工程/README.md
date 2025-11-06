# 喻园易站（YuYuanYiZhan）

面向校园场景的综合服务平台，支持闲置交易与跑腿/互助任务，内置消息私聊与邮箱验证码注册。架构为前后端分离：后端 Node.js + Express + MySQL，前端 React + Vite + Tailwind，API 采用 JWT 鉴权，消息使用 WebSocket。
[阿里云部署网站](http://47.122.126.189/)
[源码见这个仓库](https://github.com/Ilosyi/YuYuanYiZhan/tree/main)
## 功能特性

- 账号与安全
   - 邮箱验证码注册（学号 → hust.edu.cn 邮箱），登录，JWT 鉴权
   - 基础资料、关注（follow）、收藏（favorite）
- 帖子与任务
   - 帖子发布/编辑/删除，图片上传（Multer），列表筛选与搜索
   - 跑腿/互助：接单、进度查看，发布者可确认完成；被接单后支持禁用编辑
- 订单/我的
   - 我的发布、我的订单（买家/卖家/跑腿）、我的消息
- 消息与通知
   - 私信聊天（WebSocket /ws，在线推送新消息与会话摘要）

## 目录结构

```
README.md
backend/
   package.json
   server.js              # API、WebSocket、表结构初始化、图片上传、邮箱验证码
   uploads/               # 图片上传目录（持久化）
docs/
   原型设计.md / 进度计划.md 等
frontend/
   package.json
   vite.config.js / tailwind.config.js
   src/
      api/
         index.js           # Axios 实例，自动附加 JWT
      pages/
         HomePage.jsx / LoginPage.jsx / RegisterPage.jsx
         MyListingsPage.jsx / MyOrdersPage.jsx / MyMessagesPage.jsx
      components/          # ListingCard / OrderCard / PostModal 等
      context/
         AuthContext.jsx    # 登录状态与令牌管理
```

## 技术栈

- 后端：Node.js, Express, MySQL (mysql2/promise), Multer, JSON Web Token, Bcrypt, ws
- 前端：React 18/19（见 package.json）、Vite、Axios、Tailwind CSS
- 通信：REST + WebSocket（路径 /ws）

## 快速开始（部署简述）

前置要求：Node.js 18+、MySQL 8.x、可用的 SMTP（可选，用于邮箱验证码）。

1) 后端

```bash
cd backend
npm ci
# 在 backend 目录创建 .env（见下方环境变量）
node server.js   # 生产建议用 PM2 守护
```

2) 前端

```bash
cd frontend
npm ci
npm run dev      # 开发
# 或
npm run build    # 生产构建，将 dist/ 交由 Nginx 等静态服务托管
```

3) 反向代理（生产建议）

- Nginx/Apache 提供前端静态资源，并反向代理：
   - `/api` → 后端 Node 端口
   - `/uploads` → 后端静态目录
   - `/ws` → WebSocket（需开启升级与连接保持）

## 环境变量（backend/.env）

```env
# 基础服务
PORT=3000
DB_HOST=127.0.0.1
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=yuyuan_yizhan
JWT_SECRET=replace_with_a_strong_secret

# 跨域
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# 静态托管前端（可选）
SERVE_FRONTEND=false
FRONTEND_DIST_PATH=../frontend/dist

# SMTP（可选，用于邮箱验证码注册）
SMTP_ENABLED=true
SMTP_HOST=smtp.qq.com
SMTP_PORT=465
SMTP_SECURE=true
SMTP_USER=your_smtp_user
SMTP_PASS=your_smtp_app_password
SMTP_FROM=Your Name <your_smtp_user@example.com>
```

前端可在 `frontend/.env` 或 `.env.production` 指定 API：

```env
VITE_API_BASE_URL=http://localhost:3000
VITE_DEV_API_PORT=3000
```

## 注意事项与常见问题（简）

- 云服务器发信失败：需在安全组放行 SMTP 端口（465/587），使用真实 SMTP 域名与授权码；查看后端日志 `request-email-code error:` 排查。
- 图片上传：`backend/uploads/` 需持久化（挂载磁盘/共享存储）。
- WebSocket 反代：确保代理对 `/ws` 开启 `Upgrade`/`Connection` 头的透传与超时设置。

## 未来方向（摘）

- 高级搜索与排序、个性化推荐
- 通知中心（订单/消息变更）
- 移动端适配与性能优化（前端按需加载、后端缓存）
- 自动化测试与覆盖率提升


```




