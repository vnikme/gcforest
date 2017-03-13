#include "train.h"

#include <boost/random.hpp>
#include <iostream>


using namespace NGCForest;

void GenerateData(std::vector<TFeatures> &x, std::vector<size_t> &y, size_t count, boost::random::mt19937 &rng) {
    std::uniform_real_distribution<double> noise(-0.1, 0.1), other(0.0, 1.0);
    std::bernoulli_distribution answer(0.3);
    x.resize(count);
    y.resize(count);
    for (size_t i = 0; i < count; ++i) {
        x[i].resize(50);
        if (answer(rng)) {
            y[i] = 1;
            for (size_t j = 0; j < 50; ++j) {
                if (j % 10 == 0) {
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
                if (j % 10 == 0) {
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
    boost::random::mt19937 rng;
    std::vector<TFeatures> train_x, test_x;
    std::vector<size_t> train_y, test_y;
    GenerateData(train_x, train_y, 10000, rng);
    GenerateData(test_x, test_y, 1000, rng);
    TCalculatorPtr forest = Train(train_x, train_y, 2, 100);
    for (size_t i = 0; i < test_x.size(); ++i) {
        TFeatures res = forest->Calculate(test_x[i]);
        std::cout << test_y[i] << "\t" << res[1] << std::endl;
    }
    return 0;
}

