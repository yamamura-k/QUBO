#include <iostream>
#include <vector>
#include <random>
#include <ranges>
#include <chrono>
#include <cmath>
#include "qubo_greedy.h"

std::random_device rd;
std::mt19937 mt(rd());
std::chrono::system_clock::time_point start_time;

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

    void read_Q()
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

    const std::vector<int> &get_s() const
    {
        return s;
    }
    const std::vector<std::vector<double>> &get_Q() const
    {
        return Q;
    }
    double energy() const
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

    void print()
    {
        std::cerr << energy() << " : ";
        for (int i : std::views::iota(0, N))
            std::cerr << s[i] << " ";
        std::cerr << std::endl;
    }

    virtual void setup(int i) = 0;
    virtual void flip(int i) = 0;
    virtual double flip_energy(int i) const = 0;
};

class Ising : public Model
{
public:
    Ising(int N) : Model(N) {}
    virtual ~Ising() = default;

    void setup(int i) override
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

    void flip(int i) override { s[i] *= -1; }

    double flip_energy(int i) const override
    {
        double ene = 0.0;
        for (int j : std::views::iota(0, N) | std::views::filter([i](int n)
                                                                 { return i != n; }))
            ene += Q[i][j] * s[i] * s[j];
        ene += Q[i][i] * s[i];
        return -2.0 * ene;
    }
};

class QUBO : public Model
{
public:
    QUBO(int N) : Model(N) {}
    virtual ~QUBO() = default;

    void setup(int i) override
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

    void flip(int i) override { s[i] ^= 1; }

    double flip_energy(int i) const override
    {
        double ene = 0.0;
        for (int j : std::views::iota(0, N) | std::views::filter([i](int n)
                                                                 { return i != n; }))
            ene += Q[i][j] /* *1 */ * s[j];
        ene += Q[i][i] /* *1 */;
        return ene * (s[i] == 0 ? 1.0 : -1.0);
    }
};

void anneal(Model *model, int max_iters, double t, double alpha, double tm, std::vector<int> &best_s)
{
    double current_energy = model->energy();
    double min_energy = current_energy;
    best_s = model->get_s();
    std::uniform_int_distribution<> randint(0, model->size() - 1);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    for ([[maybe_unused]] int i : std::views::iota(0) | std::views::take_while([tm, max_iters](int i)
                                                                               {
                                                                                   double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count();
                                                                                   return tm - elapsed_time > tm * 0.05 && (max_iters < 0 || i < max_iters);
                                                                               }))
    {
        t *= alpha;
        int idx = randint(mt);
        double d = model->flip_energy(idx);
        if (d < 0.0 || std::exp(-d / t) > dist(mt))
        {
            current_energy += d;
            model->flip(idx);
            if (current_energy < min_energy)
            {
                min_energy = current_energy;
                best_s = model->get_s();
            }
        }
    }
}
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
                                                                 { return i != n; }))
            cost += Q[i][j] * s[i] * s[j];
        cost += Q[i][i] * s[i];
    }
    return cost;
}

void greedy(Model *model, double tm, std::vector<int> &best_s)
{
    best_s = model->get_s();
    int j, k, idx, n = best_s.size();
    int N = n + 1;
    std::vector<std::vector<double>> Q = model->get_Q();
    std::vector<int> V(n, 0);
    std::vector<double> a(n);
    std::priority_queue<std::pair<double, int>> que;
    for ([[maybe_unused]] int i : std::views::iota(0) | std::views::take_while([tm, N](int i)
                                                                               {
                                                                                   double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count();
                                                                                   return tm - elapsed_time > tm * 0.05 && (i < N);
                                                                               }))
    {
        /*****************
         * greedy 145/460
         * 値を割り当てていない変数があるうちは以下の操作を繰り返す
         * 方針Xに従って、未割り当ての変数(index)を選択
         * 方針Yに従って選択した変数に値を割り当てる
         *****************/
        if (i == 0)
        {
            // best linear approximation
            for (j = 0; j < n; ++j)
            {
                a[j] = 0;
                for (k = 0; k < n; ++k)
                {
                    a[j] += Q[j][k];
                }
                a[j] /= 2;
                que.push(std::make_pair(std::abs(a[j]), j));
            }
        }
        else
        {
            auto tmp = que.top();
            idx = tmp.second;
            V[idx] = 1;
            if (a[idx] <= 0)
            {
                if (best_s[idx] == 0)
                    model->flip(idx);
                best_s[idx] = 1;
            }
            else
            {
                if (best_s[idx] == 1)
                    model->flip(idx);
                best_s[idx] = 0;
            }
            que.pop();
        }
    }
    assert(que.empty());
}

/*
TODO :
- QUBOアルゴリズムの資料を参考に実装する
  - とりあえずOne pathのgreedyを実装してみる
- 遺伝的アルゴリズムを実装する
- solution poolみたいな機能を実装する
*/
int main(int argc, char *argv[])
{
    start_time = std::chrono::system_clock::now();
    std::string target_model = argc > 1 ? argv[1] : "qubo";
    std::string algo = argc > 2 ? argv[2] : "anneal";
    int type = (argc > 3 && std::string(argv[3]) == "test") ? 1 : 0;
    double T;
    std::cin >> T;
    int N;
    std::cin >> N;
    Model *model = target_model.compare("qubo") == 0 ? static_cast<Model *>(new QUBO(N)) : new Ising(N);
    model->setup(type);
    std::vector<std::vector<double>> Q = model->get_Q();
    model->read_Q();
    int max_iters = -1;                       // max_iterations; -1:unlimited
    double alpha = 1.0 - 0.04 / T;            // damping_factor
    double temp = std::fabs(model->energy()); // init_temperature
    double C;
    std::cin >> C;
    std::vector<int> best_s(N);

    if (algo.compare("anneal") == 0)
    {
        anneal(model, max_iters, temp, alpha, T, best_s);
        //std::cout << "annealing : " << cost(best_s, Q) << std::endl;
        print(best_s);
    }
    else if (algo.compare("greedy") == 0)
    {
        greedy(model, T, best_s);
        //std::cout << "greedy : " << cost(best_s, Q) << std::endl;
        print(best_s);
    }

    else if (algo.compare("best_l2") == 0)
    {
        Greedy<Model> gd;
        gd.best_l2(model, T, best_s);
        //std::cout << "greedy(l2) : " << cost(best_s, Q) << std::endl;
        print(best_s);
    }

    else if (algo.compare("gd_anneal") == 0)
    {
        greedy(model, T, best_s);
        anneal(model, max_iters, temp, alpha, T, best_s);
        //std::cout << "greedy -> annealing : " << cost(best_s, Q) << std::endl;
        print(best_s);
    }

    delete model;
    return 0;
}
