#include <gtest/gtest.h>
#include "PoseEq.hpp"
#include "../include/Executor.hpp"

namespace adas
{
    TEST(ExecutorFastTest, should_return_x_plus_2_given_status_is_fast_command_is_M_and_facing_is_E)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));
        // when
        executor->Execute("FM"); // FM: F状态下Move
        // then
        const Pose target{2, 0, 'E'};
        ASSERT_EQ(target, executor->Query());
    }
    TEST(ExecutorFastTest, should_return_N_and_x_plus_1_given_status_is_fast_command_is_L_and_facing_is_E)
    {
        // 命令是FL，起始状态{0,0,’E’}
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'})); // 起始状态{0, 0, 'E'}
        // when
        executor->Execute("FL"); // FL: 向前移动然后左转
        // then
        const Pose target{1, 0, 'N'}; // 结果应为{1, 0, 'N'}
        ASSERT_EQ(target, executor->Query());
    }
    TEST(ExecutorFastTest, should_return_S_and_x_plus_1_given_status_is_fast_given_command_is_R_and_facing_is_E)
    {
        // 命令是FR，起始状态{0,0,’E’}
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'})); // 起始状态{0, 0, 'E'}
        // when
        executor->Execute("FR"); // FR: 向前移动然后右转
        // then
        const Pose target{1, 0, 'S'}; // 结果应为{1, 0, 'S'}
        ASSERT_EQ(target, executor->Query());
    }
    TEST(ExecutorFastTest, should_return_y_plus_1_given_command_is_FFM_and_facing_is_N)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor()); // 默认起始状态是{0,0,'N'}
        // when
        executor->Execute("FFM"); // FFM等价于M
        // then
        const Pose target{0, 1, 'N'};
        ASSERT_EQ(target, executor->Query());
    }
} // namespace adas
