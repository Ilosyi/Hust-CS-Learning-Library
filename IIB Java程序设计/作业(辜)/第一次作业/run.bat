@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

rem ----------------------------------
rem 1. 运行 homework.ch1 中的 Welcome 类
rem ----------------------------------
echo [INFO] 正在运行 class 目录中的 homework.ch1.Welcome...
java -cp "class" homework.ch1.Welcome

if !errorlevel! neq 0 (
    echo [ERROR] 运行失败，请检查以下内容：
    echo   1. class\homework\ch1 目录下是否存在 Welcome.class
    echo   2. Java 环境变量是否配置正确（cmd 中执行 java -version 验证）
    pause
    exit /b
)

rem ----------------------------------
rem 2. 运行 run.jar 中的 Welcome 类
rem ----------------------------------
echo [INFO] 正在运行 jar/run.jar 中的 homework.ch1.Welcome...
java -cp "jar/run.jar" homework.ch1.Welcome

if !errorlevel! neq 0 (
    echo [ERROR] 运行失败，请检查以下内容：
    echo   1. jar/run.jar 文件是否存在
    echo   2. run.jar 中是否包含 homework.ch1.Welcome 类（可用 jar -tvf jar/run.jar 查看内容）
    pause
    exit /b
)

rem ----------------------------------
echo [SUCCESS] 所有任务执行完成！
pause
