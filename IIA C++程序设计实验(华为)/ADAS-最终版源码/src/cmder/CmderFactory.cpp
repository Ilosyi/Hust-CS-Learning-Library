#include "CmderFactory.hpp"

namespace adas
{
    CmderList CmderFactory::GetCmders(const std::string &commands) const noexcept
    {
        // 需要在 GetCmders 里面首先构造 std::list 对象，然后遍历 cmderMap 里面每个键值对，得到其中命令对象，加到 std::list 对象：
        CmderList cmderList;
        for (const auto cmd : parseCommandString(commands))
        {
            const auto it = cmderMap.find(cmd);
            if (it != cmderMap.end())
            {
                cmderList.push_back(it->second); // std::list 类型使用 push_back 函数添加元素
            }
        }
        return cmderList;
    }
    std::string CmderFactory::parseCommandString(std::string_view commands) const noexcept
    {
        std::string result(commands);
        ReplaceAll(result, "TR", "Z");
        return result;
    }
    void CmderFactory::ReplaceAll(std::string &inout, std::string what, std::string with) const noexcept
    {
        for (
            std::string::size_type pos{};//std::string::size_type 是 std::string 的 size 类型
            inout.npos != (pos = inout.find(what.data(), pos, what.length()));
            pos += with.length())
        {
            inout.replace(pos, what.length(), with.data(), with.length());
        }
    }
}