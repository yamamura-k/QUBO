#include "model.h"
std::random_device rd;
std::mt19937 mt(rd());

void Model::read_Q()
{
    double v;
    for (int i : std::views::iota(0, N))
    {
        for (int j : std::views::iota(i, N))
        {
            std::cin >> v;
            Q[i][j] = Q[j][i] = v;
        }
    }
}

double Model::energy() const
{
    double ene = 0.0;
    for (int i : std::views::iota(0, N))
    {
        for (int j : std::views::iota(0, N) | std::views::filter([i](int n)
                                                                 { return i < n; }))
            ene += Q[i][j] * s[i] * s[j];
        ene += Q[i][i] * s[i];
    }
    return ene;
}

void Model::print()
{
    std::cerr << energy() << " : ";
    for (int i : std::views::iota(0, N))
        std::cerr << s[i] << " ";
    std::cerr << std::endl;
}

void Ising::setup(int i)
{
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    for (int i : std::views::iota(0, N))
        s[i] = 1;
    if (i == 0)
        return;
    for (int i : std::views::iota(0, N))
        for (int j : std::views::iota(0, N))
            Q[i][j] = dist(mt);
}

double Ising::flip_energy(int i) const
{
    double ene = 0.0;
    for (int j : std::views::iota(0, N) | std::views::filter([i](int n)
                                                             { return i != n; }))
        ene += Q[i][j] * s[i] * s[j];
    ene += Q[i][i] * s[i];
    return -2.0 * ene;
}

double QUBO::flip_energy(int i) const
{
    double ene = 0.0;
    for (int j : std::views::iota(0, N) | std::views::filter([i](int n)
                                                             { return i != n; }))
        ene += Q[i][j] /* *1 */ * s[j];
    ene += Q[i][i] /* *1 */;
    return ene * (s[i] == 0 ? 1.0 : -1.0);
}
void QUBO::setup(int i)
{
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    for (int i : std::views::iota(0, N))
        s[i] = 1;
    if (i == 0)
        return;
    for (int i : std::views::iota(0, N))
        for (int j : std::views::iota(0, N))
            Q[i][j] = i != j ? dist(mt) : 0.0;
}