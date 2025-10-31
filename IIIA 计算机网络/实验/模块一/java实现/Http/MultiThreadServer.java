package Http;

import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Web服务器 - 多线程版本
 * 功能：使用线程池处理并发HTTP请求，支持Range请求
 */
public class MultiThreadServer {
    // 服务器监听地址
    private String serverAddress;
    // 服务器监听端口
    private int serverPort;
    // 网页文件主目录
    private String rootDirectory;
    // 服务器Socket
    private ServerSocket serverSocket;
    // 线程池
    private ExecutorService threadPool;
    // 日期格式化工具
    private SimpleDateFormat dateFormat;
    // 请求计数器（线程安全）
    private AtomicInteger requestCounter = new AtomicInteger(0);

    /**
     * 构造函数：从配置文件加载服务器配置
     * @param configFile 配置文件路径
     */
    public MultiThreadServer(String configFile) {
        loadConfig(configFile);
        // 创建固定大小的线程池
        threadPool = Executors.newFixedThreadPool(10);
        dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    }

    /**
     * 加载配置文件
     * @param configFile 配置文件路径
     */
    private void loadConfig(String configFile) {
        Properties properties = new Properties();
        try (FileInputStream fis = new FileInputStream(configFile)) {
            properties.load(fis);
            serverAddress = properties.getProperty("server.address", "localhost");
            serverPort = Integer.parseInt(properties.getProperty("server.port", "8080"));
            rootDirectory = properties.getProperty("server.root", "./www");

            System.out.println("配置加载成功:");
            System.out.println("  服务器地址: " + serverAddress);
            System.out.println("  服务器端口: " + serverPort);
            System.out.println("  主目录: " + rootDirectory);
        } catch (IOException e) {
            System.err.println("配置文件加载失败，使用默认配置");
            serverAddress = "localhost";
            serverPort = 8080;
            rootDirectory = "./www";
        }
    }

    /**
     * 启动服务器
     */
    public void start() {
        try {
            // 创建服务器Socket
            InetAddress address = InetAddress.getByName(serverAddress);
            serverSocket = new ServerSocket(serverPort, 50, address);

            System.out.println("\n========================================");
            System.out.println("多线程Web服务器启动成功！");
            System.out.println("监听地址: " + serverAddress + ":" + serverPort);
            System.out.println("主目录: " + new File(rootDirectory).getAbsolutePath());
            System.out.println("线程池大小: 10");
            System.out.println("等待客户端连接...");
            System.out.println("========================================\n");

            // 循环接受客户端连接
            while (true) {
                Socket clientSocket = serverSocket.accept();
                // 使用线程池处理请求
                threadPool.execute(new RequestHandler(clientSocket));
            }
        } catch (IOException e) {
            System.err.println("服务器启动失败: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 请求处理器（Runnable任务）
     */
    private class RequestHandler implements Runnable {
        private Socket clientSocket;
        private int requestId;

        public RequestHandler(Socket clientSocket) {
            this.clientSocket = clientSocket;
            this.requestId = requestCounter.incrementAndGet();
        }

        @Override
        public void run() {
            String clientIP = clientSocket.getInetAddress().getHostAddress();
            int clientPort = clientSocket.getPort();
            String timestamp = dateFormat.format(new Date());
            String threadName = Thread.currentThread().getName();

            System.out.println("\n----------------------------------------");
            System.out.println("[请求#" + requestId + "] [" + timestamp + "] [" + threadName + "]");
            System.out.println("客户端IP: " + clientIP);
            System.out.println("客户端端口: " + clientPort);

            try (
                    BufferedReader in = new BufferedReader(
                            new InputStreamReader(clientSocket.getInputStream())
                    );
                    DataOutputStream out = new DataOutputStream(
                            clientSocket.getOutputStream()
                    )
            ) {
                // 读取HTTP请求行
                String requestLine = in.readLine();

                if (requestLine == null || requestLine.isEmpty()) {
                    System.out.println("[请求#" + requestId + "] 状态: 空请求");
                    return;
                }

                System.out.println("[请求#" + requestId + "] 请求行: " + requestLine);

                // 读取请求头，并保存Range信息
                String headerLine;
                String rangeHeader = null;
                while ((headerLine = in.readLine()) != null && !headerLine.isEmpty()) {
                    System.out.println("[请求#" + requestId + "] 请求头: " + headerLine);

                    // 检测Range请求头（不区分大小写）
                    String lowerHeader = headerLine.toLowerCase();
                    if (lowerHeader.startsWith("range:")) {
                        // 找到冒号的位置，提取Range值
                        int colonIndex = headerLine.indexOf(':');
                        if (colonIndex >= 0 && colonIndex < headerLine.length() - 1) {
                            rangeHeader = headerLine.substring(colonIndex + 1).trim();
                            System.out.println("[请求#" + requestId + "] >>> 检测到Range请求: [" + rangeHeader + "]");
                        }
                    }
                }

                // 解析请求行
                String[] requestParts = requestLine.split(" ");
                if (requestParts.length < 3) {
                    sendErrorResponse(out, 400, "Bad Request", requestId);
                    return;
                }

                String method = requestParts[0];
                String requestPath = requestParts[1];

                // 只支持GET方法
                if (!method.equals("GET")) {
                    sendErrorResponse(out, 405, "Method Not Allowed", requestId);
                    return;
                }

                // 处理根路径
                if (requestPath.equals("/")) {
                    requestPath = "/index.html";
                }

                // 构造完整文件路径
                String filePath = rootDirectory + requestPath;
                File file = new File(filePath);

                // 安全检查
                if (!file.getCanonicalPath().startsWith(
                        new File(rootDirectory).getCanonicalPath())) {
                    sendErrorResponse(out, 403, "Forbidden", requestId);
                    return;
                }

                // 检查文件是否存在
                if (!file.exists()) {
                    sendErrorResponse(out, 404, "Not Found", requestId);
                    return;
                }

                // 检查是否为文件
                if (!file.isFile()) {
                    sendErrorResponse(out, 403, "Forbidden", requestId);
                    return;
                }

                // 根据是否有Range请求头决定发送方式
                if (rangeHeader != null && !rangeHeader.isEmpty()) {
                    System.out.println("[请求#" + requestId + "] >>> 使用Range响应处理");
                    sendRangeResponse(out, file, rangeHeader, requestId);
                } else {
                    System.out.println("[请求#" + requestId + "] >>> 使用普通响应处理");
                    sendSuccessResponse(out, file, requestId);
                }

            } catch (IOException e) {
                System.err.println("[请求#" + requestId + "] 处理请求时发生错误: " +
                        e.getMessage());
            } finally {
                try {
                    clientSocket.close();
                    System.out.println("[请求#" + requestId + "] 连接关闭");
                    System.out.println("----------------------------------------");
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        /**
         * 发送Range响应（206 Partial Content）
         */
        private void sendRangeResponse(DataOutputStream out, File file,
                                       String rangeHeader, int requestId)
                throws IOException {
            long fileSize = file.length();
            long startByte = 0;
            long endByte = fileSize - 1;

            // 解析Range头：格式为 "bytes=start-end" 或 "bytes=start-"
            if (rangeHeader.startsWith("bytes=")) {
                String range = rangeHeader.substring(6); // 去掉"bytes="
                String[] parts = range.split("-");

                try {
                    if (parts.length >= 1 && !parts[0].isEmpty()) {
                        startByte = Long.parseLong(parts[0]);
                    }
                    if (parts.length >= 2 && !parts[1].isEmpty()) {
                        endByte = Long.parseLong(parts[1]);
                    }

                    // 验证范围的有效性
                    if (startByte < 0 || startByte >= fileSize || endByte < startByte) {
                        sendErrorResponse(out, 416, "Range Not Satisfiable", requestId);
                        return;
                    }
                    if (endByte >= fileSize) {
                        endByte = fileSize - 1;
                    }
                } catch (NumberFormatException e) {
                    sendErrorResponse(out, 400, "Bad Request", requestId);
                    return;
                }
            }

            long contentLength = endByte - startByte + 1;
            String contentType = getContentType(file.getName());

            // 构造HTTP 206响应头
            String responseHeader = "HTTP/1.1 206 Partial Content\r\n" +
                    "Content-Type: " + contentType + "\r\n" +
                    "Content-Length: " + contentLength + "\r\n" +
                    "Content-Range: bytes " + startByte + "-" + endByte + "/" + fileSize + "\r\n" +
                    "Accept-Ranges: bytes\r\n" +
                    "Connection: close\r\n" +
                    "\r\n";

            // 发送响应头
            out.writeBytes(responseHeader);

            // 读取并发送文件的指定范围
            try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
                raf.seek(startByte); // 定位到开始字节

                byte[] buffer = new byte[8192]; // 8KB缓冲区
                long remaining = contentLength;

                while (remaining > 0) {
                    int toRead = (int) Math.min(buffer.length, remaining);
                    int bytesRead = raf.read(buffer, 0, toRead);

                    if (bytesRead == -1) {
                        break;
                    }

                    out.write(buffer, 0, bytesRead);
                    remaining -= bytesRead;
                }
            }

            out.flush();

            // 输出处理结果
            System.out.println("[请求#" + requestId + "] 响应状态: 206 Partial Content");
            System.out.println("[请求#" + requestId + "] 文件路径: " + file.getAbsolutePath());
            System.out.println("[请求#" + requestId + "] 文件总大小: " + fileSize + " 字节");
            System.out.println("[请求#" + requestId + "] Range: bytes " + startByte + "-" + endByte + "/" + fileSize);
            System.out.println("[请求#" + requestId + "] 发送字节数: " + contentLength + " 字节");
            System.out.println("[请求#" + requestId + "] Content-Type: " + contentType);
            System.out.println("[请求#" + requestId + "] 处理成功！");
        }

        /**
         * 发送成功响应（200 OK）- 流式传输版本
         */
        private void sendSuccessResponse(DataOutputStream out, File file, int requestId)
                throws IOException {
            long fileSize = file.length();
            String contentType = getContentType(file.getName());

            // 构造HTTP响应头（添加Accept-Ranges支持）
            String responseHeader = "HTTP/1.1 200 OK\r\n" +
                    "Content-Type: " + contentType + "\r\n" +
                    "Content-Length: " + fileSize + "\r\n" +
                    "Accept-Ranges: bytes\r\n" +  // 告知客户端支持Range请求
                    "Connection: close\r\n" +
                    "\r\n";

            // 发送响应头
            out.writeBytes(responseHeader);

            // 使用流式传输
            try (FileInputStream fis = new FileInputStream(file)) {
                byte[] buffer = new byte[8192]; // 8KB缓冲区
                int bytesRead;
                long totalSent = 0;

                while ((bytesRead = fis.read(buffer)) != -1) {
                    out.write(buffer, 0, bytesRead);
                    totalSent += bytesRead;

                    // 每发送1MB输出一次进度（仅对大文件）
                    if (fileSize > 1024 * 1024 && totalSent % (1024 * 1024) == 0) {
                        System.out.println("[请求#" + requestId + "] 已发送: " +
                                (totalSent / 1024 / 1024) + "MB / " +
                                (fileSize / 1024 / 1024) + "MB");
                    }
                }

                out.flush();

                System.out.println("[请求#" + requestId + "] 响应状态: 200 OK");
                System.out.println("[请求#" + requestId + "] 文件路径: " + file.getAbsolutePath());
                System.out.println("[请求#" + requestId + "] 文件大小: " + fileSize + " 字节");
                System.out.println("[请求#" + requestId + "] 实际发送: " + totalSent + " 字节");
                System.out.println("[请求#" + requestId + "] Content-Type: " + contentType);
                System.out.println("[请求#" + requestId + "] 处理成功！");

            } catch (IOException e) {
                System.err.println("[请求#" + requestId + "] 发送文件内容时出错: " + e.getMessage());
                throw e;
            }
        }

        /**
         * 发送错误响应
         */
        private void sendErrorResponse(DataOutputStream out, int statusCode,
                                       String statusMessage, int requestId) {
            try {
                String errorPage = "<html><head><title>" + statusCode + " " +
                        statusMessage + "</title></head>" +
                        "<body><h1>" + statusCode + " " + statusMessage + "</h1>" +
                        "<p>请求的资源无法找到或访问。</p>" +
                        "<hr><p>Multi-Thread Web Server</p></body></html>";

                byte[] content = errorPage.getBytes("UTF-8");

                String responseHeader = "HTTP/1.1 " + statusCode + " " + statusMessage +
                        "\r\n" +
                        "Content-Type: text/html; charset=UTF-8\r\n" +
                        "Content-Length: " + content.length + "\r\n" +
                        "Connection: close\r\n" +
                        "\r\n";

                out.writeBytes(responseHeader);
                out.write(content);
                out.flush();

                System.out.println("[请求#" + requestId + "] 响应状态: " + statusCode +
                        " " + statusMessage);

            } catch (IOException e) {
                System.err.println("[请求#" + requestId + "] 发送错误响应失败: " +
                        e.getMessage());
            }
        }
    }

    /**
     * 根据文件扩展名获取MIME类型
     */
    private String getContentType(String fileName) {
        String extension = "";
        int index = fileName.lastIndexOf('.');
        if (index > 0) {
            extension = fileName.substring(index + 1).toLowerCase();
        }

        switch (extension) {
            case "html":
            case "htm":
                return "text/html; charset=UTF-8";
            case "css":
                return "text/css";
            case "js":
                return "application/javascript";
            case "json":
                return "application/json";
            case "jpg":
            case "jpeg":
                return "image/jpeg";
            case "png":
                return "image/png";
            case "gif":
                return "image/gif";
            case "ico":
                return "image/x-icon";
            case "svg":
                return "image/svg+xml";
            case "txt":
                return "text/plain; charset=UTF-8";
            case "pdf":
                return "application/pdf";
            case "mp3":
                return "audio/mpeg";
            case "mp4":
                return "video/mp4";
            case "webm":
                return "video/webm";
            case "ogg":
                return "video/ogg";
            case "avi":
                return "video/x-msvideo";
            case "mov":
                return "video/quicktime";
            default:
                return "application/octet-stream";
        }
    }

    /**
     * 停止服务器
     */
    public void stop() {
        try {
            threadPool.shutdown();
            if (!threadPool.awaitTermination(5, TimeUnit.SECONDS)) {
                threadPool.shutdownNow();
            }
            if (serverSocket != null && !serverSocket.isClosed()) {
                serverSocket.close();
                System.out.println("服务器已停止");
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * 主函数
     */
    public static void main(String[] args) {
        String configFile = "server.properties";

        if (args.length > 0) {
            configFile = args[0];
        }

        MultiThreadServer server = new MultiThreadServer(configFile);

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\n正在关闭服务器...");
            server.stop();
        }));

        server.start();
    }
}
