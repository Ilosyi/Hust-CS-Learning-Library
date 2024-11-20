#include <gtest/gtest.h>
#include <memory>
#include "PoseEq.hpp"
#include "../include/Executor.hpp"

namespace adas
{

        // 测试返回初始化的 Pose
    TEST(ExecutorTest, should_return_init_pose_when_without_command)
    {
        // given 给定测试条件
        // 测试条件是就是调用Executor的静态方法NewExecutor返回一个指向 Executor 对象的智能指针 executor，这样我们就不需要delete了
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'})); // 给了初始姿态

        // when

        // then
        const Pose expectedPose = {0, 0, 'E'}; // 构造一个姿态对象，其内容为(0, 0, 'E')
        // 既然构造对象时的初始姿态是(0, 0, 'E')，那么调用Query()方法返回的姿态也应该是(0, 0, 'E')
        // 所以这里用了断言，executor->Query()返回的姿态应该等于expectedPose，否则测试失败，说明Query()方法有问题
        ASSERT_EQ(expectedPose, executor->Query()); // 内部调用了重载的pose的==操作符
    }

    // 测试返回默认的 Pose
    TEST(ExecutorTest, should_return_default_pose_when_without_init_and_command)
    {
        // given 给定测试条件

        std::unique_ptr<Executor> executor(Executor::NewExecutor());

        // when
        // 不给初始姿态，也不给指令
        // then
        const Pose expectedPose = {0, 0, 'N'}; // 构造一个姿态对象，其内容为(0, 0, 'N')
        // 由于没有给定初始姿态，所以调用Query()方法返回的姿态应该是默认的(0, 0, 'N')
        ASSERT_EQ(expectedPose, executor->Query());
    }
    // 移动指令测试
    // 当朝向是东时，向前移动，x坐标加1
    TEST(ExecutorTest, should_return_x_plus_1_given_command_is_M_and_facing_is_E)
    {
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'})); //
        executor->Execute("M");
        ASSERT_EQ(executor->Query(), Pose({1, 0, 'E'}));
    }
    // 当朝向是西时，向前移动，x坐标减1
    TEST(ExecutorTest, should_return_x_minus_1_given_command_is_M_and_facing_is_W)
    {
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'W'}));
        executor->Execute("M");
        ASSERT_EQ(executor->Query(), Pose({-1, 0, 'W'}));
    }
    // 当朝向是北时，向前移动，y坐标加1
    TEST(ExecutorTest, should_return_y_plus_1_given_command_is_M_and_facing_is_N)
    {
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'N'}));
        executor->Execute("M");
        ASSERT_EQ(executor->Query(), Pose({0, 1, 'N'}));
    }
    // 当朝向是南时，向前移动，y坐标减1
    TEST(ExecutorTest, should_return_y_minus_1_given_command_is_M_and_facing_is_S)
    {
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'S'}));
        executor->Execute("M");
        ASSERT_EQ(executor->Query(), Pose({0, -1, 'S'}));
    }

    // 左转指令测试
    // 当朝向是东时，左转，朝向变为北
    TEST(ExecutorTest, should_return_facing_N_given_command_is_L_and_facing_is_E)
    {
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));
        executor->Execute("L");
        ASSERT_EQ(executor->Query(), Pose({0, 0, 'N'}));
    }
    // 当朝向是西时，左转，朝向变为南
    TEST(ExecutorTest, should_return_facing_S_given_command_is_L_and_facing_is_W)
    {
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'W'}));
        executor->Execute("L");
        ASSERT_EQ(executor->Query(), Pose({0, 0, 'S'}));
    }
    // 当朝向是北时，左转，朝向变为西
    TEST(ExecutorTest, should_return_facing_W_given_command_is_L_and_facing_is_N)
    {
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'N'}));
        executor->Execute("L");
        ASSERT_EQ(executor->Query(), Pose({0, 0, 'W'}));
    }
    // 当朝向是南时，左转，朝向变为东
    TEST(ExecutorTest, should_return_facing_E_given_command_is_L_and_facing_is_S)
    {
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'S'}));
        executor->Execute("L");
        ASSERT_EQ(executor->Query(), Pose({0, 0, 'E'}));
    }

    // 右转指令测试
    // 当朝向是东时，右转，朝向变为南
    TEST(ExecutorTest, should_return_facing_S_given_command_is_R_and_facing_is_E)
    {
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'E'}));
        executor->Execute("R");
        ASSERT_EQ(executor->Query(), Pose({0, 0, 'S'}));
    }
    // 当朝向是南时，右转，朝向变为西
    TEST(ExecutorTest, should_return_facing_W_given_command_is_R_and_facing_is_S)
    {
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'S'}));
        executor->Execute("R");
        ASSERT_EQ(executor->Query(), Pose({0, 0, 'W'}));
    }
    // 当朝向是西时，右转，朝向变为北
    TEST(ExecutorTest, should_return_facing_N_given_command_is_R_and_facing_is_W)
    {
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'W'}));
        executor->Execute("R");
        ASSERT_EQ(executor->Query(), Pose({0, 0, 'N'}));
    }
    // 当朝向是北时，右转，朝向变为东
    TEST(ExecutorTest, should_return_facing_E_given_command_is_R_and_facing_is_N)
    {
        std::unique_ptr<Executor> executor(Executor::NewExecutor({0, 0, 'N'}));
        executor->Execute("R");
        ASSERT_EQ(executor->Query(), Pose({0, 0, 'E'}));
    }
} // namespace adas
