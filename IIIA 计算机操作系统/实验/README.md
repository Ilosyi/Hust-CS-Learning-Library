## [代码看这个仓库](https://gitee.com/ilosyi/hustos-pke)
每个子实验对应一个分支

具体要改动哪些代码可结合git记录和实验文档

## 环境准备教程（docker）
### **安装Docker桌面环境**

* 第一步，安装Docker

  * **Ubuntu**

  对于 **x86_64** 架构，安装具体的过程可以参考[这篇](https://blog.csdn.net/magic_ll/article/details/139985543?spm=1001.2014.3001.5506)文章。对于 **arm64** 架构，安装具体的过程可以参考[这篇](https://blog.csdn.net/sglin123/article/details/139754107?ops_request_misc=&request_id=&biz_id=102&utm_term=arm64%E5%AE%89%E8%A3%85docker&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-139754107.142^v100^pc_search_result_base8&spm=1018.2226.3001.4187)文章。

  * **Windows**

  Windows 版本的 Docker 安装可以参考[这篇](https://blog.csdn.net/Liuj666/article/details/126099982?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522EC5F862A-6D9C-439D-96AB-6CB77A783F13%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=EC5F862A-6D9C-439D-96AB-6CB77A783F13&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-126099982-null-null.142^v100^pc_search_result_base8&utm_term=windows%E5%AE%89%E8%A3%85docker&spm=1018.2226.3001.4187)文章。

  * **macOS**

  macOS 版本的 Docker 安装可以参考[这篇](https://blog.csdn.net/weixin_41860471/article/details/135048312?ops_request_misc=%257B%2522request%255Fid%2522%253A%25220015913A-5C64-4FD0-A192-3686BF8FE2C4%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=0015913A-5C64-4FD0-A192-3686BF8FE2C4&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-4-135048312-null-null.142^v100^pc_search_result_base8&utm_term=macos%E5%AE%89%E8%A3%85docker&spm=1018.2226.3001.4187)文章。

  

* 第二步，拉取镜像

  * **x86_64/amd64 版本镜像**

  *Dockerhub 镜像源*

  `$ docker pull docker.io/tjr9098/amd64_pke_mirrors:1.0`

  *阿里云镜像源*

  `$ docker pull crpi-vycj2ba2y82yi8d0.cn-hangzhou.personal.cr.aliyuncs.com/pke_mirrors/amd64_pke_mirrors:1.0`

  * **arm64 版本镜像**

  *Dockerhub 镜像源*

  `$ docker pull docker.io/tjr9098/arm64_pke_mirrors:1.0`

  *阿里云镜像源*

  `$ docker pull crpi-vycj2ba2y82yi8d0.cn-hangzhou.personal.cr.aliyuncs.com/pke_mirrors/arm64_pke_mirrors:1.0`

  

* 第三步，运行镜像

`$ docker run -it --name pke_mirror crpi-vycj2ba2y82yi8d0.cn-hangzhou.personal.cr.aliyuncs.com/pke_mirrors/amd64_pke_mirrors:1.0`

​	● **`IMAGE`** 是镜像名称，名称可通过`$ docker images`查看。

​	● **`-it`**: 交互式运行容器，分配一个伪终端。

​	● **`--name`**: 为容器命名，便于后续使用。

**再次进入容器**
通过Docker desktop软件点击运行
<img width="1254" height="395" alt="image" src="https://github.com/user-attachments/assets/13d4de06-ca18-4a20-a8ab-ad3d7458273e" />

