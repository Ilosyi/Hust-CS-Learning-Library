#pragma once

#include "ActionGroup.hpp"

namespace adas
{

    class CmderOrchestrator//移动适配器抽象父类
    {
    public:
        virtual ~CmderOrchestrator() = default;

    public://四种抽象方法
        virtual ActionGroup Move(const PoseHandler &poseHandler) const noexcept = 0;
        virtual ActionGroup TurnLeft(const PoseHandler &poseHandler) const noexcept = 0;
        virtual ActionGroup TurnRight(const PoseHandler &poseHandler) const noexcept = 0;
        virtual ActionGroup TurnRound(const PoseHandler &poseHandler) const noexcept = 0;
    };

} // namespace adas