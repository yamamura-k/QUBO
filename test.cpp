#include "model.h"
#include "qubo_greedy.h"
#include "utils.h"

int main(int argc, char *argv[])
{
    std::chrono::system_clock::time_point start_time, tmp_time;
    std::string target_model = argc > 1 ? argv[1] : "qubo";
    double T;
    std::cin >> T;
    int N;
    std::cin >> N;
    Model *model = target_model.compare("qubo") == 0 ? static_cast<Model *>(new QUBO(N)) : new Ising(N);
    model->setup(1);
    std::vector<std::vector<double>> Q = model->get_Q();
    start_time = std::chrono::system_clock::now();
    std::vector<int> best_s;

    start_time = std::chrono::system_clock::now();
    model->setup(0);
    Greedy<Model> gd;
    gd.best_l2(model, T, best_s);
    tmp_time = std::chrono::system_clock::now();
    std::cout << "greedy(l2) : " << cost(best_s, Q) << " [ " << std::chrono::duration_cast<std::chrono::milliseconds>(tmp_time - start_time).count() << " ms ]" << std::endl;
    //print(best_s);

    delete model;
    return 0;
}