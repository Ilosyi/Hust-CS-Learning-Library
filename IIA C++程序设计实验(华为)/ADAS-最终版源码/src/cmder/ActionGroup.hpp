#pragma once

#include <list>
#include "../src/core/PoseHandler.hpp"

namespace adas
{
    enum class ActionType : uint16_t
    {
        FORWARD_1_STEP_ACTION = 0,
        BACKWARD_1_STEP_ACTION,
        TURN_LEFT_ACTION,
        REVERSE_TURN_LEFT_ACTION,
        TURN_RIGHT_ACTION,
        REVERSE_TURN_RIGHT_ACTION,
        BE_FAST_ACTION,
        BE_REVERSE_ACTION,
    };

    class ActionGroup final
    {
    public:
        ActionGroup(void) = default;// 默认构造函数
        explicit ActionGroup(const std::list<ActionType> &actions) noexcept; // 有参构造函数,explicit代表显示构造函数
        ~ActionGroup() = default;// 默认析构函数

        //重载+=运算符
        ActionGroup &operator+=(const ActionGroup &rhs) noexcept;


        void PushAction(const ActionType &ActionType) noexcept;// 添加动作
        void DoOperate(PoseHandler &poseHandler) const noexcept;// 执行动作

    private:
        std::list<ActionType> actions;// 动作列表
    };
}