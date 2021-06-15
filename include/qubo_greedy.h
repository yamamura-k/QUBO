#include <vector>
#include <chrono>
#include <queue>
#include <cassert>
template <class T>
class Greedy
{
protected:
    std::vector<double> Delta;
    void delta(std::vector<double> s, std::vector<std::vector<double>> Q);

public:
    Greedy() {}
    std::vector<double> init(std::vector<std::vector<double>> Q, std::vector<double> h, int type);
    void best_linear(T *model, double tm, std::vector<int> &best_s);
    void best_l2(T *model, double tm, std::vector<int> &best_s);
};