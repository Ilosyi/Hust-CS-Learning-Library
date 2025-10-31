package Http;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Properties;

public class HttpClient {
    // 保存从客户端到Web服务器的HTTP请求报文
    String request = null;
    // Web服务器的域名
    String serverDomainName = null;
    // Web服务器的端口
    int srvProt = -1;
    // Socket对象
    Socket socket = null;
    // 向Web服务器发送HTTP请求报文的输出流
    PrintWriter outToServer = null;
    // 读取Web服务器HTTP响应报文内容的输入流
    DataInputStream inFromServer = null;
    // 将Web服务器HTTP响应报文内容写入另外一个文本文件的输出流
    DataOutputStream outToFile = null;

    /**
     * 返回Web服务器域名
     * @return the serverDomainName
     */
    public String getServerDomainName() {
        return serverDomainName;
    }

    /**
     * 设置Web服务器域名
     * @param serverDomainName the serverDomainName to set
     */
    public void setServerDomainName(String serverDomainName) {
        this.serverDomainName = serverDomainName;
    }

    /**
     * 返回Web服务器端口号
     * @return the srvProt
     */
    public int getSrvProt() {
        return srvProt;
    }

    /**
     * 设置Web服务器端口号
     * @param srvProt the srvProt to set
     */
    public void setSrvProt(int srvProt) {
        this.srvProt = srvProt;
    }

    /**
     * 判断是否连接到服务器
     * @return the connected
     */
    public boolean isConnected() {
        if (this.socket == null) {
            return false;
        } else {
            return this.socket.isConnected();
        }
    }

    /**
     * 设置HTTP请求报文的内容
     * @param request the request to set
     */
    public void setRequest(String request) {
        this.request = request;
    }

    /**
     * 构造函数
     * @param serverDomainName:服务器域名
     * @param srvProt: 服务器端口
     */
    public HttpClient(String serverDomainName, int srvProt) {
        this.serverDomainName = serverDomainName;
        this.srvProt = srvProt;
    }

    /**
     * 连接到服务器
     */
    public void Connect() {
        if (!isConnected()) {
            try {
                // 创建Socket对象并连接到指定的域名和端口
                this.socket = new Socket(serverDomainName, srvProt);
                outToServer = new PrintWriter(this.socket.getOutputStream(), true);
                inFromServer = new DataInputStream(this.socket.getInputStream());
                System.out.println("已成功连接到服务器：" + serverDomainName + ":" + srvProt);
            } catch (UnknownHostException e) {
                System.err.println("无法识别服务器地址：" + serverDomainName);
                e.printStackTrace();
            } catch (IOException e) {
                System.err.println("连接服务器失败：" + serverDomainName + ":" + srvProt);
                e.printStackTrace();
            }
        }
    }

    /**
     * 断开与服务器的连接
     */
    public void Disconnect() {
        if (isConnected()) {
            try {
                this.inFromServer.close();
                this.outToServer.close();
                this.socket.close();
                System.out.println("已断开与服务器的连接");
            } catch (IOException e) {
                System.err.println("断开连接时发生错误");
                e.printStackTrace();
            }
        }
    }

    /**
     * 向服务器发送HTTP请求报文
     */
    public void sendHttptRequest() {
        if (request != null && !request.isEmpty()) {
            outToServer.println(this.request);
            System.out.println("已发送HTTP请求：");
            System.out.println("------------------------------");
            System.out.println(request);
            System.out.println("------------------------------");
        } else {
            System.err.println("HTTP请求报文为空，无法发送");
        }
    }

    /**
     * 得到服务器的HTTP响应报文并保存在文件里
     * @param fileName:文件名
     */
    public void getHttpResponse(String fileName) {
        try {
            // 从服务器的应答中读入一个字节
            Byte b = inFromServer.readByte();
            outToFile = new DataOutputStream(new FileOutputStream(fileName));
            System.out.println("正在接收服务器响应，将保存到：" + fileName);
            while (true) {
                // 将该字节写入到文件里
                outToFile.writeByte(b);
                b = inFromServer.readByte();
            }
            // 当到了服务器应答的末尾，会抛出EOFException异常
        } catch (EOFException e) {
            System.out.println("响应接收完成！");
        } catch (FileNotFoundException e) {
            System.err.println("无法创建输出文件：" + fileName);
            e.printStackTrace();
        } catch (IOException e) {
            System.err.println("接收响应或写入文件时发生错误");
            e.printStackTrace();
        } finally {
            try {
                // 关闭文件输出流
                if (outToFile != null) {
                    outToFile.close();
                }
            } catch (IOException e) {
                System.err.println("关闭文件输出流时发生错误");
                e.printStackTrace();
            }
        }
    }

    /**
     * 从配置文件读取服务器地址和端口
     * @param configFilePath 配置文件路径
     * @return 包含地址和端口的数组，index0=地址，index1=端口
     */
    private static String[] loadServerConfig(String configFilePath) {
        Properties properties = new Properties();
        // 默认配置（与服务器默认配置一致）
        String defaultAddress = "localhost";
        String defaultPort = "8080";

        try (FileInputStream fis = new FileInputStream(configFilePath)) {
            properties.load(fis);
            // 读取配置，若配置不存在则用默认值
            String address = properties.getProperty("server.address", defaultAddress);
            String port = properties.getProperty("server.port", defaultPort);
            System.out.println("配置文件加载成功：");
            System.out.println("  服务器地址：" + address);
            System.out.println("  服务器端口：" + port);
            return new String[]{address, port};
        } catch (FileNotFoundException e) {
            System.err.println("配置文件未找到：" + configFilePath + "，使用默认配置");
        } catch (IOException e) {
            System.err.println("配置文件读取失败，使用默认配置");
        }
        // 返回默认配置
        System.out.println("使用默认配置：");
        System.out.println("  服务器地址：" + defaultAddress);
        System.out.println("  服务器端口：" + defaultPort);
        return new String[]{defaultAddress, defaultPort};
    }

    public static void main(String[] args) {
        // 1. 读取配置文件（默认路径为"server.properties"，与服务器配置文件一致）
        String configFile = "server.properties";
        // 支持命令行参数指定配置文件路径（如：java Http.HttpClient custom_config.properties）
        if (args.length > 0) {
            configFile = args[0];
        }
        String[] serverConfig = loadServerConfig(configFile);
        String serverAddress = serverConfig[0];
        int serverPort = Integer.parseInt(serverConfig[1]);

        // 2. 初始化HttpClient（使用配置文件中的地址和端口）
        HttpClient httpClient = new HttpClient(serverAddress, serverPort);
        // 3. 建立与服务器的连接
        httpClient.Connect();

        // 4. 构造HTTP请求报文（请求根路径，自动映射到index.html）
        String request = "GET / HTTP/1.1\r\n" +
                "Host: " + serverAddress + ":" + serverPort + "\r\n" +  // Host头与配置一致
                "Connection: close\r\n" +
                "User-agent: MyCustomClient/1.0\r\n" +
                "Accept: text/html,image/jpeg,image/png,*/*\r\n" +
                "\r\n";  // 空行结束请求头，必须保留

        // 5. 设置并发送请求
        httpClient.setRequest(request);
        httpClient.sendHttptRequest();

        // 6. 接收服务器响应，保存到response.txt
        httpClient.getHttpResponse("response.txt");

        // 7. 断开连接
        httpClient.Disconnect();

        System.out.println("请求流程结束！响应已保存到 response.txt");
    }
}