#include "evaluation.h"
#include "train.h"
#include "forest.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <sstream>
#include <string>
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

static double RandId(const std::string &id) {
    size_t val = 0;
    for (char ch : id) {
        val *= 119;
        val += static_cast<unsigned int>(ch);
    }
    return static_cast<double>(val) / std::numeric_limits<size_t>::max();
}

static void ReadPoolTransposed(TMiniBatch &x, std::vector<size_t> &y, const std::string &path, double prob, size_t expectedCount) {
    std::ifstream file(path);
    while (!file.eof()) {
        std::string line;
        std::getline(file, line);
        std::istringstream str(line);
        std::string evid, cvid, item;
        std::getline(str, evid, '\t');
        if (evid == "EventId")
            continue;
        std::getline(str, cvid, '\t');
        if (RandId(evid + cvid) > prob)
            continue;
        y.push_back(0);
        str >> y.back();
        std::getline(str, item, '\t');
        size_t feature = 0;
        while (!str.eof()) {
            if (feature >= x.size()) {
                x.emplace_back();
                x.back().reserve(expectedCount);
            }
            x[feature].push_back(0.0);
            str >> x[feature].back();
            ++feature;
        }
        if (feature == 0)
            y.pop_back();
        //if (x.front().size() >= 10000)
        //    break;
    }
}

static void ReadPool(TMiniBatch &x, std::vector<size_t> &y, const std::string &path, double prob) {
    std::ifstream file(path);
    while (!file.eof()) {
        std::string line;
        std::getline(file, line);
        std::istringstream str(line);
        std::string evid, cvid, item;
        std::getline(str, evid, '\t');
        if (evid == "EventId")
            continue;
        std::getline(str, cvid, '\t');
        if (RandId(evid + cvid) > prob)
            continue;
        y.push_back(0);
        str >> y.back();
        std::getline(str, item, '\t');
        x.emplace_back();
        TFeatures &features = x.back();
        while (!str.eof()) {
            features.push_back(0.0);
            str >> features.back();
        }
        if (features.empty()) {
            x.pop_back();
            y.pop_back();
        }
        //if (x.size() >= 1000)
        //    break;
    }
}

void Work() {
    std::vector<TFeatures> train_x, test_x;
    std::vector<size_t> train_y, test_y;
    ReadPoolTransposed(train_x, train_y, "../train.tsv", 0.7, 4480000);
    std::cout << train_x.back().size() << " " << train_x.size() << std::endl;
    //GenerateData(train_x, train_y, 100000, rng);
    //TCalculatorPtr forest = TrainRandomForest(train_x, train_y, 2, 10, 100);
    //TCalculatorPtr forest = TrainFullRandomForest(train_x, train_y, 2, 10, 100);
    constexpr size_t levelCount = 6;
    TCalculatorPtr forest = TrainCascadeForest(train_x, train_y, 2, 15, 1000, levelCount);
    train_x.clear();
    train_y.clear();
    ReadPool(test_x, test_y, "../test.tsv", 0.01);
    //GenerateData(test_x, test_y, 10000, rng);
    size_t instanceCount = test_x.size();
    for (size_t k = 1; k <= levelCount; ++k) {
        TCalculatorPtr frst = dynamic_cast<TCascadeForestCalculator*>(forest.get())->GetSlice(k);
        std::vector<std::pair<int, double>> answers(test_x.size());
        std::vector<std::thread> threads(4);
        for (size_t t = 0; t < 4; ++t) {
            std::thread thrd([t, instanceCount, &answers, &test_x, &test_y, &frst]() {
                for (size_t i = instanceCount / 4 * t; i < std::min(instanceCount, instanceCount / 4 * (t + 1)); ++i) {
                    TFeatures res = frst->Calculate(test_x[i]);
                    answers[i] = std::make_pair(test_y[i], res[1]);
                }
            });
            threads[t] = std::move(thrd);
        }
        for (size_t t = 0; t < 4; ++t)
            threads[t].join();
        std::cout << "AUC " << k << ": " << AUC(std::move(answers)) << std::endl;
    }
}

int main() {
    try {
        Work();
    }
    catch (const std::exception &ex) {
        std::cout << "Exception caught: " << ex.what() << std::endl;
    }
    return 0;
}
