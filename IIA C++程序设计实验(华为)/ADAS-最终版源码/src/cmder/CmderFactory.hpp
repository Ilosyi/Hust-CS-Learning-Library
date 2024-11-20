#pragma once

#include <functional>
#include <list>
#include <unordered_map>
#include <map>
#include "Command.hpp"
#include"ActionGroup.hpp"

namespace adas
{ // std::function 是个模板，类型实参是函数参数及返回值，
    // std::function<ActionGroup(PoseHandler& poseHandler)> 表示这样一个函数：
    // 参数为 PoseHandler& poseHandler，返回类型为 ActionGroup。
    // 作为 std::function 的类型实参，但不光是一个函数可以作为 std::function 的类型实参，
    // 重载了 ActionGroup operator()(PoseHandler& poseHandler) 的函数对象
    // 以及输入参数为 PoseHandler& poseHandler，返回类型为 ActionGroup 的 lambda 表达式
    // 都可以作为类型实参。

    // ActionGroup.cpp 里面的 ForwardAction 等不都是重载了
    // ActionGroup operator()(PoseHandler& poseHandler) 的函数对象吗？

    // std::function 模板定义了一个可调用对象：函数，函数对象，lambda 表达式，函数指针都是可调用对象。
    // 它将不同类型的可调用对象统一起来
    using Cmder = std::function<ActionGroup(PoseHandler &poseHandler, const CmderOrchestrator
    &orchestrator)>;
    using CmderList = std::list<Cmder>;
    class CmderFactory final
    {
    public:
        CmderFactory(void) = default;  // 默认构造函数
        ~CmderFactory(void) = default; // 默认析构函数

        CmderFactory(const CmderFactory &) = delete; // 删除拷贝构造函数
        CmderFactory &operator=(const CmderFactory &) = delete;

    public:
        CmderList GetCmders(const std::string &commands) const noexcept; // 获取命令

    private:
        const std::unordered_map<char, Cmder> cmderMap{
            // 命令表
            {'M', MoveCommand()},
            {'L', TurnLeftCommand()},
            {'R', TurnRightCommand()},
            {'F', FastCommand()},
            {'B', ReverseCommand()},
            {'Z', TurnRoundCommand()},
        };
    private:
    //将字符串中的"TR"替换为'Z'
    std::string parseCommandString(std::string_view commands) const noexcept;
    //string_view表示一个字符串的视图，不拥有字符串的所有权，只是一个字符串的引用
    void ReplaceAll(std::string &inout, std::string what, std::string with)const noexcept;
    };
}