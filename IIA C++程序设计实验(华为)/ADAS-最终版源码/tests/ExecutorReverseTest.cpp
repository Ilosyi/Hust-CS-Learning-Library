#include <gtest/gtest.h>
#include "Executor.hpp"
#include "PoseEq.hpp"

namespace adas
{

    // 测试输入: BM
    TEST(ExecutorReverseTest, should_return_x_minus_1_given_status_is_back_command_is_M_and_facing_is_E)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));
        // when
        executor->Execute("BM");
        // then
        const Pose target{-1, 0, 'E'};
        ASSERT_EQ(target, executor->Query());
    }
    // 测试输入: BL
    TEST(ExecutorReverseTest, should_return_S_given_status_is_reverse_command_is_L_and_facing_is_E)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));
        // when
        executor->Execute("BL"); // 切换到倒车模式并执行左转命令
        // then
        const Pose target{0, 0, 'S'};
        ASSERT_EQ(target, executor->Query());
    }

    // 测试输入: BR
    TEST(ExecutorReverseTest, should_return_N_given_status_is_reverse_command_is_R_and_facing_is_E)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));
        // when
        executor->Execute("BR"); // 切换到倒车模式并执行右转命令
        // then
        const Pose target{0, 0, 'N'};
        ASSERT_EQ(target, executor->Query());
    }

    // 测试输入: BBM
    TEST(ExecutorReverseTest, should_return_y_plus_1_given_command_is_BBM_and_facing_is_N)
    {
        // given
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'N'}));
        // when
        executor->Execute("BBM"); // 切换到倒车模式，再切换回正常模式，然后执行前进命令
        // then
        const Pose target{0, 1, 'N'};
        ASSERT_EQ(target, executor->Query());
    }

} // namespace adas