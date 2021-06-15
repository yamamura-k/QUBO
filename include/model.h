#include <iostream>
#include <vector>
#include <random>
#include <ranges>
#include <chrono>
#include <cmath>
class Model
{
protected:
    int N;
    std::vector<int> s;
    std::vector<std::vector<double>> Q;

public:
    Model(int N) : N(N),
                   s(std::vector<int>(N)),
                   Q(std::vector<std::vector<double>>(N, std::vector<double>(N))) {}
    virtual ~Model() = default;
    int size() const { return N; }
    void read_Q();
    const std::vector<int> &get_s() const
    {
        return s;
    }
    const std::vector<std::vector<double>> &get_Q() const
    {
        return Q;
    }
    double energy() const;
    void print();
    virtual void setup(int i) = 0;
    virtual void flip(int i) = 0;
    virtual double flip_energy(int i) const = 0;
};

class Ising : public Model
{
public:
    Ising(int N) : Model(N) {}
    virtual ~Ising() = default;
    void setup(int i) override;
    void flip(int i) override { s[i] *= -1; }
    double flip_energy(int i) const override;
};

class QUBO : public Model
{
public:
    QUBO(int N) : Model(N) {}
    virtual ~QUBO() = default;
    void setup(int i) override;
    void flip(int i) override { s[i] ^= 1; }
    double flip_energy(int i) const override;
};