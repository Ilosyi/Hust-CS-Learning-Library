package Http;

import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Web服务器 - 单线程版本
 * 功能：处理HTTP请求，返回网页文件和多媒体资源，支持Range请求
 */
public class Server {
    // 服务器监听地址
    private String serverAddress;
    // 服务器监听端口
    private int serverPort;
    // 网页文件主目录
    private String rootDirectory;
    // 服务器Socket
    private ServerSocket serverSocket;
    // 日期格式化工具
    private SimpleDateFormat dateFormat;

    /**
     * 构造函数：从配置文件加载服务器配置
     * @param configFile 配置文件路径
     */
    public Server(String configFile) {
        loadConfig(configFile);
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
            // 创建服务器Socket，绑定到指定地址和端口
            InetAddress address = InetAddress.getByName(serverAddress);
            serverSocket = new ServerSocket(serverPort, 50, address);

            System.out.println("\n========================================");
            System.out.println("Web服务器启动成功！");
            System.out.println("监听地址: " + serverAddress + ":" + serverPort);
            System.out.println("主目录: " + new File(rootDirectory).getAbsolutePath());
            System.out.println("等待客户端连接...");
            System.out.println("========================================\n");

            // 循环接受客户端连接
            while (true) {
                Socket clientSocket = serverSocket.accept();
                handleRequest(clientSocket);
            }
        } catch (IOException e) {
            System.err.println("服务器启动失败: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 处理客户端请求（支持Range请求）
     * @param clientSocket 客户端Socket连接
     */
    private void handleRequest(Socket clientSocket) {
        String clientIP = clientSocket.getInetAddress().getHostAddress();
        int clientPort = clientSocket.getPort();
        String timestamp = dateFormat.format(new Date());

        System.out.println("\n----------------------------------------");
        System.out.println("[" + timestamp + "] 新连接");
        System.out.println("客户端IP: " + clientIP);
        System.out.println("客户端端口: " + clientPort);

        BufferedReader in = null;
        DataOutputStream out = null;

        try {
            // 设置Socket选项
            clientSocket.setSoTimeout(30000);        // 30秒超时
            clientSocket.setTcpNoDelay(true);        // 禁用Nagle算法，提高实时性
            clientSocket.setSendBufferSize(65536);   // 64KB发送缓冲区

            in = new BufferedReader(
                    new InputStreamReader(clientSocket.getInputStream())
            );
            out = new DataOutputStream(
                    clientSocket.getOutputStream()
            );

            // 读取HTTP请求的第一行（请求行）
            String requestLine = in.readLine();

            if (requestLine == null || requestLine.isEmpty()) {
                System.out.println("状态: 空请求");
                return;
            }

            System.out.println("请求行: " + requestLine);

            // 读取请求头，并保存Range信息
            String headerLine;
            String rangeHeader = null;
            while ((headerLine = in.readLine()) != null && !headerLine.isEmpty()) {
                System.out.println("请求头: " + headerLine);

                // 检测Range请求头
                String lowerHeader = headerLine.toLowerCase();
                if (lowerHeader.startsWith("range:")) {
                    int colonIndex = headerLine.indexOf(':');
                    if (colonIndex >= 0 && colonIndex < headerLine.length() - 1) {
                        rangeHeader = headerLine.substring(colonIndex + 1).trim();
                        System.out.println(">>> 检测到Range请求: [" + rangeHeader + "]");
                    }
                }
            }

            // 解析请求行
            String[] requestParts = requestLine.split(" ");
            if (requestParts.length < 3) {
                sendErrorResponse(out, 400, "Bad Request", clientIP, clientPort);
                return;
            }

            String method = requestParts[0];
            String requestPath = requestParts[1];

            // 只支持GET方法
            if (!method.equals("GET")) {
                sendErrorResponse(out, 405, "Method Not Allowed", clientIP, clientPort);
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
                sendErrorResponse(out, 403, "Forbidden", clientIP, clientPort);
                return;
            }

            // 检查文件是否存在
            if (!file.exists()) {
                sendErrorResponse(out, 404, "Not Found", clientIP, clientPort);
                return;
            }

            // 检查是否为文件
            if (!file.isFile()) {
                sendErrorResponse(out, 403, "Forbidden", clientIP, clientPort);
                return;
            }

            // 根据是否有Range请求头决定发送方式
            if (rangeHeader != null && !rangeHeader.isEmpty()) {
                System.out.println(">>> 使用Range响应处理");
                sendRangeResponse(out, file, rangeHeader, clientIP, clientPort);
            } else {
                System.out.println(">>> 使用普通响应处理");
                sendSuccessResponse(out, file, clientIP, clientPort);
            }

        } catch (SocketTimeoutException e) {
            System.err.println("连接超时: " + e.getMessage());
        } catch (SocketException e) {
            // 客户端主动断开连接（这是常见情况，不算错误）
            if (e.getMessage().contains("中止") || e.getMessage().contains("Connection reset")) {
                System.out.println(">>> 客户端断开连接（可能是用户关闭了页面）");
            } else {
                System.err.println("Socket错误: " + e.getMessage());
            }
        } catch (IOException e) {
            System.err.println("处理请求时发生IO错误: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("处理请求时发生未知错误: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // 确保资源被正确关闭
            try {
                if (out != null) out.close();
                if (in != null) in.close();
                if (clientSocket != null && !clientSocket.isClosed()) {
                    clientSocket.close();
                }
                System.out.println("连接关闭");
                System.out.println("----------------------------------------");
            } catch (IOException e) {
                // 忽略关闭时的异常
            }
        }
    }


    /**
     * 发送Range响应（206 Partial Content）
     * @param out 输出流
     * @param file 要发送的文件
     * @param rangeHeader Range请求头内容（如："bytes=0-"）
     * @param clientIP 客户端IP
     * @param clientPort 客户端端口
     */
    private void sendRangeResponse(DataOutputStream out, File file,
                                   String rangeHeader, String clientIP, int clientPort)
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
                    sendErrorResponse(out, 416, "Range Not Satisfiable", clientIP, clientPort);
                    return;
                }
                if (endByte >= fileSize) {
                    endByte = fileSize - 1;
                }
            } catch (NumberFormatException e) {
                sendErrorResponse(out, 400, "Bad Request", clientIP, clientPort);
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
        System.out.println("响应状态: 206 Partial Content");
        System.out.println("文件路径: " + file.getAbsolutePath());
        System.out.println("文件总大小: " + fileSize + " 字节");
        System.out.println("Range: bytes " + startByte + "-" + endByte + "/" + fileSize);
        System.out.println("发送字节数: " + contentLength + " 字节");
        System.out.println("Content-Type: " + contentType);
        System.out.println("处理成功！");
    }

    /**
     * 发送成功响应（200 OK）- 改进版，支持大文件流式传输
     * @param out 输出流
     * @param file 要发送的文件
     * @param clientIP 客户端IP
     * @param clientPort 客户端端口
     */
    private void sendSuccessResponse(DataOutputStream out, File file,
                                     String clientIP, int clientPort) throws IOException {
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

        // 使用流式传输，避免大文件占用太多内存
        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] buffer = new byte[8192]; // 8KB缓冲区
            int bytesRead;
            long totalSent = 0;

            while ((bytesRead = fis.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
                totalSent += bytesRead;

                // 每发送1MB输出一次进度（仅对大文件）
                if (fileSize > 1024 * 1024 && totalSent % (1024 * 1024) == 0) {
                    System.out.println("已发送: " + (totalSent / 1024 / 1024) + "MB / " +
                            (fileSize / 1024 / 1024) + "MB");
                }
            }

            out.flush();

            // 输出处理结果
            System.out.println("响应状态: 200 OK");
            System.out.println("文件路径: " + file.getAbsolutePath());
            System.out.println("文件大小: " + fileSize + " 字节");
            System.out.println("实际发送: " + totalSent + " 字节");
            System.out.println("Content-Type: " + contentType);
            System.out.println("处理成功！");

        } catch (IOException e) {
            System.err.println("发送文件内容时出错: " + e.getMessage());
            throw e;
        }
    }

    /**
     * 发送错误响应
     * @param out 输出流
     * @param statusCode HTTP状态码
     * @param statusMessage 状态消息
     * @param clientIP 客户端IP
     * @param clientPort 客户端端口
     */
    private void sendErrorResponse(DataOutputStream out, int statusCode,
                                   String statusMessage, String clientIP, int clientPort) {
        try {
            // 构造错误页面HTML
            String errorPage = "<html><head><title>" + statusCode + " " + statusMessage +
                    "</title></head>" +
                    "<body><h1>" + statusCode + " " + statusMessage + "</h1>" +
                    "<p>请求的资源无法找到或访问。</p>" +
                    "<hr><p>Simple Web Server</p></body></html>";

            byte[] content = errorPage.getBytes("UTF-8");

            // 构造HTTP响应头
            String responseHeader = "HTTP/1.1 " + statusCode + " " + statusMessage + "\r\n" +
                    "Content-Type: text/html; charset=UTF-8\r\n" +
                    "Content-Length: " + content.length + "\r\n" +
                    "Connection: close\r\n" +
                    "\r\n";

            // 发送响应
            out.writeBytes(responseHeader);
            out.write(content);
            out.flush();

            // 输出处理结果
            System.out.println("响应状态: " + statusCode + " " + statusMessage);
            System.out.println("错误信息已发送");

        } catch (IOException e) {
            System.err.println("发送错误响应失败: " + e.getMessage());
        }
    }

    /**
     * 根据文件扩展名获取MIME类型
     * @param fileName 文件名
     * @return MIME类型字符串
     */
    private String getContentType(String fileName) {
        String extension = "";
        int index = fileName.lastIndexOf('.');
        if (index > 0) {
            extension = fileName.substring(index + 1).toLowerCase();
        }

        // 常见MIME类型映射
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
            if (serverSocket != null && !serverSocket.isClosed()) {
                serverSocket.close();
                System.out.println("服务器已停止");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 主函数
     */
    public static void main(String[] args) {
        // 配置文件路径
        String configFile = "server.properties";

        // 如果命令行指定了配置文件，使用命令行参数
        if (args.length > 0) {
            configFile = args[0];
        }

        // 创建并启动服务器
        Server server = new Server(configFile);

        // 添加关闭钩子，确保优雅关闭
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\n正在关闭服务器...");
            server.stop();
        }));

        // 启动服务器
        server.start();
    }
}
