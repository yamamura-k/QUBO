#include <iostream>
#include <vector>
#include <ranges>

void print(std::vector<int> best_s)
{
    for (int x : best_s)
        std::cout << x << " ";
    std::cout << std::endl;
}
double cost(std::vector<int> s, std::vector<std::vector<double>> Q)
{
    double cost = 0.0;
    int N = s.size();
    for (int i : std::views::iota(0, N))
    {
        for (int j : std::views::iota(0, N) | std::views::filter([i](int n)
                                                                 { return i < n; }))
            cost += Q[i][j] * s[i] * s[j];
        cost += Q[i][i] * s[i];
    }
    return cost;
}