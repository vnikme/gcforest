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

void GenerateData(std::vector<TFeatures> &x, std::vector<size_t> &y, std::vector<size_t> &g, size_t count, std::mt19937 &rng) {
    std::uniform_real_distribution<double> noise(-0.1, 0.1), other(0.0, 1.0);
    std::bernoulli_distribution answer(0.3);
    x.resize(count);
    y.resize(count);
    g.resize(count);
    for (size_t i = 0; i < count; ++i) {
        g[i] = i;
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

static void ReadPoolTransposed(TMiniBatch &x, std::vector<size_t> &y, std::vector<size_t> &g, const std::string &path, double prob, size_t expectedCount, std::vector<std::string> &featureNames) {
    std::ifstream file(path);
    std::string prevId;
    size_t group = -1;
    while (!file.eof()) {
        std::string line;
        std::getline(file, line);
        std::istringstream str(line);
        std::string clicked, evid, item;
        std::getline(str, clicked, '\t');
        std::getline(str, evid, '\t');
        if (clicked == "ClickColumn") {
            while (!str.eof()) {
                std::getline(str, item, '\t');
                featureNames.push_back(item);
            }
            continue;
        }
        if (RandId(evid) > prob)
            continue;
        if (evid != prevId) {
            prevId = evid;
            ++group;
        }
        y.push_back(clicked == "1" ? 1 : 0);
        g.push_back(group);
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
        if (feature == 0) {
            y.pop_back();
            g.pop_back();
        }
        else {
            x[87].back() = 0.0;
        }
        //if (x.front().size() >= 100000)
        //    break;
    }
}

static void ReadPool(TMiniBatch &x, std::vector<size_t> &y, std::vector<size_t> &g, const std::string &path, double prob) {
    std::ifstream file(path);
    std::string prevId;
    size_t group = -1;
    while (!file.eof()) {
        std::string line;
        std::getline(file, line);
        std::istringstream str(line);
        std::string clicked, evid, item;
        std::getline(str, clicked, '\t');
        std::getline(str, evid, '\t');
        if (clicked == "ClickColumn")
            continue;
        if (RandId(evid) > prob)
            continue;
        if (evid != prevId) {
            prevId = evid;
            ++group;
        }
        y.push_back(clicked == "1" ? 1 : 0);
        g.push_back(group);
        x.emplace_back();
        TFeatures &features = x.back();
        while (!str.eof()) {
            features.push_back(0.0);
            str >> features.back();
        }
        if (features.empty()) {
            x.pop_back();
            y.pop_back();
            g.pop_back();
        }
        //if (x.size() >= 10000)
        //    break;
    }
}

void Work() {
    std::mt19937 rng;
    std::vector<TFeatures> x;
    std::vector<size_t> y, g;
    std::vector<std::string> featureNames;
    ReadPoolTransposed(x, y, g, "train.tsv", 0.2, 2200000, featureNames);
    //GenerateData(x, y, g, 100000, rng);
    //x = Transpose(x);
    std::cout << y.size() << " " << x.size() << std::endl;
    //TCalculatorPtr forest = TrainRandomForest(train_x, train_y, 2, 10, 100);
    //TCalculatorPtr forest = TrainFullRandomForest(train_x, train_y, 2, 10, 100);
    constexpr size_t levelCount = 10;
    TCalculatorPtr forest = TrainCascadeForest(x, y, g, 2, 20, 128, 0.9, 5, levelCount);
    x.clear();
    y.clear();
    g.clear();
    ReadPool(x, y, g, "test.tsv", 0.2);
    //GenerateData(x, y, g, 100000, rng);
    size_t instanceCount = y.size();
    time_t startTime = time(nullptr);
    TCascadeForestCalculator *calc = dynamic_cast<TCascadeForestCalculator*>(forest.get());
    std::vector<std::vector<std::pair<int, double>>> answers(levelCount, std::vector<std::pair<int, double>>(instanceCount));
    std::vector<std::thread> threads(4);
    for (size_t t = 0; t < 4; ++t) {
        std::thread thrd([t, levelCount, instanceCount, &answers, &x, &y, calc]() {
            for (size_t i = instanceCount / 4 * t; i < std::min(instanceCount, instanceCount / 4 * (t + 1)); ++i) {
                std::vector<TFeatures> res = calc->CalculateForAllLevels(x[i]);
                for (size_t k = 0; k < levelCount; ++k)
                    answers[k][i] = std::make_pair(y[i], res[k][1]);
            }
        });
        threads[t] = std::move(thrd);
    }
    for (size_t t = 0; t < 4; ++t)
        threads[t].join();
    std::cout << "Score calculation time: " << time(nullptr) - startTime << std::endl;
    {
        std::ofstream fout("scores.txt");
        for (size_t i = 0; i < instanceCount; ++i) {
            fout << g[i] << '\t' << y[i];
            for (size_t j = 0; j < levelCount; ++j)
                fout << '\t' << answers[j][i].second;
            fout << std::endl;
        }
    }
    for (size_t k = 0; k < levelCount; ++k) {
        std::cout << "AUC " << k << ": " << AUC(std::move(answers[k])) << std::endl;
    }
    {
        std::ofstream fout("model.txt");
        fout << featureNames.size();
        for (const std::string &name : featureNames)
            fout << ' ' << name;
        fout << std::endl;
        forest->Save(fout);
    }
}

int main() {
    try {
        Work();
    }
    catch (const std::exception &ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
    }
    return 0;
}
