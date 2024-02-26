#include "utils.hpp"

using namespace core;

bool test_broadcast_1()
{
    std::vector<int> s1 = {1,2,4};
    std::vector<int> s2 = {3,4};
    try{
        std::vector<int> broadcasted1 =  broadcastShapes(s1, s2);
    }
    catch(const std::logic_error& e){
        return false;
    }
    return true;
}

bool test_broadcast_2()
{
    std::vector<int> s1 = {1,2,4};
    std::vector<int> s2 = {1,1,4};
    std::vector<int> broadcasted1 =  broadcastShapes(s1, s2);
    for (size_t i = 0; i < s1.size(); i++)
        if(s1[i] != broadcasted1[i])
            return true;
    return false;
}

bool test_broadcast_3()
{
    std::vector<int> s1 = {1,2,4};
    std::vector<int> s2 = {1,2,1};
    std::vector<int> broadcasted1 =  broadcastShapes(s1, s2);
    for (size_t i = 0; i < s1.size(); i++)
        if(s1[i] != broadcasted1[i])
            return true;
    return false;
}

bool test_broadcast_4()
{
    std::vector<int> s1 = {1,2};
    std::vector<int> s2 = {1,2,4};
    try{
        std::vector<int> broadcasted1 =  broadcastShapes(s1, s2);
    }
    catch(const std::logic_error& e){
        return false;
    }
    return true;
}

bool test_broadcast_5()
{
    std::vector<int> s1 = {2};
    std::vector<int> s2 = {1,2,4};
    try{
        std::vector<int> broadcasted1 =  broadcastShapes(s1, s2);
    }
    catch(const std::logic_error& e){
        return false;
    }
    return true;
}

int main()
{
    if (test_broadcast_1())
        return 1;
    if (test_broadcast_2())
        return 1;
    if (test_broadcast_3())
        return 1;
    if (test_broadcast_4())
        return 1;
    if (test_broadcast_5())
        return 1;
    return 0;
}
