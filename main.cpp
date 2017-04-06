#include "evaluation.h"
#include "train.h"

#include <iostream>
#include <random>
#include <vector>


using namespace NGCForest;

void GenerateData(std::vector<TFeatures> &x, std::vector<size_t> &y, size_t count, std::mt19937 &rng) {
    std::uniform_real_distribution<double> noise(-0.1, 0.1), other(0.0, 1.0);
    std::bernoulli_distribution answer(0.3);
    x.resize(count);
    y.resize(count);
    for (size_t i = 0; i < count; ++i) {
        x[i].resize(50);
        if (answer(rng)) {
            y[i] = 1;
            for (size_t j = 0; j < 50; ++j) {
                if (false && j % 25 == 0) {
                    x[i][j] = 0.6 + noise(rng);
                } else if (j == 49) {
                    x[i][j] = x[i][21] + x[i][22];
                } else {
                    x[i][j] = other(rng);
                }
            }
        } else {
            y[i] = 0;
            for (size_t j = 0; j < 50; ++j) {
                if (false && j % 25 == 0) {
                    x[i][j] = 0.5 + noise(rng);
                } else if (j == 49) {
                    x[i][j] = x[i][21] - x[i][22] + 0.1;
                } else {
                    x[i][j] = other(rng);
                }
            }
        }
    }
}

int main() {
    std::mt19937 rng;
    std::vector<TFeatures> train_x, test_x;
    std::vector<size_t> train_y, test_y;
    GenerateData(train_x, train_y, 10000, rng);
    GenerateData(test_x, test_y, 1000, rng);
    //TCalculatorPtr forest = TrainRandomForest(train_x, train_y, 2, 10, 100);
    //TCalculatorPtr forest = TrainFullRandomForest(train_x, train_y, 2, 10, 100);
    TCalculatorPtr forest = TrainCascadeForest(train_x, train_y, 2, 10, 100, 2);
    std::vector<std::pair<int, double>> answers(test_x.size());
    for (size_t i = 0; i < test_x.size(); ++i) {
        TFeatures res = forest->Calculate(test_x[i]);
        answers[i] = std::make_pair(test_y[i], res[1]);
        //std::cerr << test_y[i] << "\t" << res[1] << std::endl;
    }
    std::cout << "AUC: " << AUC(std::move(answers)) << std::endl;
    return 0;
}

