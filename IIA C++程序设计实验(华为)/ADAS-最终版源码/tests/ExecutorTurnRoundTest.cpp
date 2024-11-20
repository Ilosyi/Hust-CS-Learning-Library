#include <gtest/gtest.h>
#include "Executor.hpp"
#include "PoseEq.hpp"

namespace adas
{

    // 测试输入：TR
    TEST(ExecutorTurnRoundTest, should_normal_tr_build_left_forward_left)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));

        // when
        executor->Execute("TR");

        // then
        const Pose target{0, 1, 'W'};
        ASSERT_EQ(target, executor->Query());
    }

    // 测试输入：FTR
    TEST(ExecutorTurnRoundTest, should_fast_tr_build_left_forward_left)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));

        // when
        executor->Execute("FTR");

        // then
        const Pose target{1, 1, 'W'};
        ASSERT_EQ(target, executor->Query());
    }

    // 测试输入：BTR
    TEST(ExecutorTurnRoundTest, in_the_B_state_the_reverse_command_will_be_ignored)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));

        // when
        executor->Execute("BTR");

        // then
        const Pose target{0, 0, 'E'};
        ASSERT_EQ(target, executor->Query());
    }
}