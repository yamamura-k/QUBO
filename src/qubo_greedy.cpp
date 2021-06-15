#include "qubo_greedy.h"
#include "model.h"

template <class T>
void Greedy<T>::delta(std::vector<double> s, std::vector<std::vector<double>> Q)
{
    int n = s.size();
    Delta.resize(n);
    for (int i = 0; i < n; i++)
    {
        Delta[i] = 0;
        for (int j = 0; j < n; j++)
        {
            if (i != j)
                Delta[i] += s[j] * Q[i][j];
        }
    }
};

template <class T>
std::vector<double> Greedy<T>::init(std::vector<std::vector<double>> Q, std::vector<double> h, int type)
{
    int n = Q[0].size();
    std::vector<double> p(n, 0);
    if (type == 0) // center
    {
        std::vector<double> p(n, 0.5);
    }
    else if (type == 1) // pos / neg
    {
        double tmp = 0;
        for (int i = 0; i < n; ++i)
        {
            tmp += std::max(0.0, h[i]);
            for (int j = i + 1; j < n; ++j)
            {
                tmp += std::max(0.0, Q[i][j]);
            }
        }
        double denom = 0;
        for (int i = 0; i < n; ++i)
        {
            denom += std::abs(h[i]);
            for (int j = i + 1; j < n; ++j)
            {
                denom += std::abs(Q[i][j]);
            }
        }
        double rho = tmp / denom;
        std::vector<double> p(n, 1 - rho);
    }
    else if (type == 2) // best
    {
        double tmp = 0;
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                tmp += Q[i][j];
            }
        }
        double lambda;
        if (tmp < 0)
            lambda = 1;
        else
            lambda = 0;
        std::vector<double> p(n, lambda);
    }
    return p;
}
template <class T>
void Greedy<T>::best_linear(T *model, double tm, std::vector<int> &best_s)
{
    best_s = model->get_s();
    int j, k, idx, n = best_s.size();
    int N = n + 1;
    std::vector<std::vector<double>> Q = model->get_Q();
    std::vector<int> V(n, 0);
    std::vector<double> a(n);
    std::priority_queue<std::pair<double, int>> que;
    for (int i = 0; i < N; ++i)
    {
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
};

template <class T>
void Greedy<T>::best_l2(T *model, double tm, std::vector<int> &best_s)
{
    best_s = model->get_s();
    int j, k, idx, n = best_s.size();
    double max_value;
    std::vector<std::vector<double>> Q = model->get_Q();
    std::vector<double> a(n), tmp_s(n, 0.5);
    for (int i = 0; i < n; ++i)
    {
        // best linear l2-approximation
        delta(tmp_s, Q);
        max_value = -1;
        idx = -1;
        for (j = 0; j < n; ++j)
        {
            if (tmp_s[j] != 0.5)
            {
                a[j] = Delta[j];
                if (max_value < std::abs(a[j]))
                {
                    max_value = std::abs(a[j]);
                    idx = j;
                }
            }
        }
        if (idx < 0)
            return;
        if (a[idx] <= 0)
        {
            if (best_s[idx] == 0)
                model->flip(idx);
            tmp_s[idx] = 1;
            best_s[idx] = 1;
        }
        else
        {
            if (best_s[idx] == 1)
                model->flip(idx);
            best_s[idx] = 0;
        }
    }
};
template void Greedy<Model>::best_l2(Model *, double, std::vector<int> &);
