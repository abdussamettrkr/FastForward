#include <chrono>
#include "ops.hpp"
#include "tensor.hpp"
#include <string>

double run(int h, int w, int in_c, int out_c, int k_size, int b_size);
void print_results(std::vector<std::string> header, std::vector<std::vector<int>> cases, std::vector<double> results);
std::string padResult(int input, int pad_size = 6);
std::string padResult(std::string input, int pad_size = 6);

int main()
{
    std::vector<std::vector<int>> cases;
    std::vector<std::string> header = {"H", "W", "In_c", "Out_c", "K_size", "B_size"};
    std::vector<double> elapsed_times;
    cases.push_back({32, 16, 3, 8, 3, 8});
    cases.push_back({32, 16, 8, 32, 3, 8});
    cases.push_back({224, 224, 8, 32, 3, 8});
    cases.push_back({32, 16, 8, 32, 5, 8});
    cases.push_back({224, 224, 8, 32, 5, 8});
    cases.push_back({32, 32, 8, 32, 3, 8});
    cases.push_back({32, 32, 32, 32, 3, 8});
    cases.push_back({32, 32, 128, 32, 3, 8});
    cases.push_back({32, 32, 32, 128, 3, 8});
    cases.push_back({32, 32, 64, 64, 3, 8});
    cases.push_back({32, 32, 128, 128, 3, 8});

    for (auto sc : cases)
    {
        auto ms = run(sc[0], sc[1], sc[2], sc[3], sc[4], sc[5]);
        elapsed_times.push_back(ms);
    }

    print_results(header, cases, elapsed_times);
}

double run(int h, int w, int in_c, int out_c, int k_size, int b_size)
{
    auto inp = core::Tensor::ones({b_size, h, w, in_c});
    auto kernel = core::Tensor::ones({out_c, k_size, k_size, in_c});

    for (size_t i = 0; i < 3; i++)
    {
        core::Tensor out = ops::conv2d(inp, kernel);
        for (size_t i = 0; i < out.size(); i++)
        {
            if (out.data()[i] != k_size * k_size * in_c)
            {
                throw std::logic_error("");
            }
        }
    }


    auto start_time = std::chrono::steady_clock::now();

    for (size_t i = 0; i < 20; i++)
    {
        ops::conv2d(inp, kernel);
    }
    auto end_time = std::chrono::steady_clock::now();

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    return static_cast<double>(elapsed_time.count()) / 20.0f;
}

void print_results(std::vector<std::string> header, std::vector<std::vector<int>> cases, std::vector<double> results)
{
    for (auto item : header)
    {
        std::cout << padResult(item) << ", ";
    }
    std::cout << padResult("ms") << std::endl;

    for (size_t i = 0; i < cases.size(); i++)
    {
        for (size_t j = 0; j < cases[i].size(); j++)
        {
            std::cout << padResult(cases[i][j]) << ", ";
        }
        std::cout << results[i] << std::endl;
    }
}

std::string padResult(int input, int pad_size)
{
    auto str = std::to_string(input);
    while (str.length() < pad_size)
    {
        str = " " + str;
    }
    return str;
}

std::string padResult(std::string str, int pad_size)
{
    while (str.length() < pad_size)
    {
        str = " " + str;
    }
    return str;
}
