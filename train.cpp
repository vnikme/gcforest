#include "evaluation.h"
#include "forest.h"
#include "forest_impl.h"
#include "train.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <set>
#include <list>
#include <random>
#include <thread>


namespace NGCForest {

    namespace {

        std::vector<double> CountClasses(const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes, size_t begin, size_t end) {
            std::vector<double> f(classCount);
            for (size_t i = begin; i < end; ++i) {
                f[y[indexes[i]]] += 1;
            }
            return f;
        }

        double GiniImpurity(const std::vector<double> &p) {
            double sum = 1e-38;
            for (double val : p)
                sum += val;
            double res = 0.0;
            for (double val : p) {
                res += (val / sum * (1 - val / sum));
            }
            return res;
        }

        std::vector<double> ClassDistribution(const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes, size_t begin, size_t end) {
            std::vector<double> f = CountClasses(y, classCount, indexes, begin, end);
            double sum = 1e-38;
            for (double val : f)
                sum += val;
            for (double &val : f)
                val /= sum;
            return f;
        }

        size_t WinnerClass(const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes, size_t begin, size_t end, std::mt19937 &rng) {
            std::vector<double> f(ClassDistribution(y, classCount, indexes, begin, end));
            std::discrete_distribution<> dist(f.begin(), f.end());
            return dist(rng);
        }

        bool IsOnlyOneClass(const std::vector<size_t> &y, const std::vector<size_t> indexes, size_t begin, size_t end) {
            if (begin == end)
                return true;
            size_t firstClass = y[indexes[begin]];
            for (size_t i = begin + 1; i < end; ++i) {
                if (y[indexes[i]] != firstClass)
                    return false;
            }
            return true;
        }

        TConstFeaturesPtr OneHot(size_t hot, size_t count) {
            std::shared_ptr<TFeatures> ans(new TFeatures(count));
            (*ans)[hot] = 1.0;
            return ans;
        }

        struct TBucket {
            TTreeNodePtr Node;
            std::vector<size_t> &Indexes;
            size_t Begin, End, Depth;
            size_t BestFeature;
            double BestThreshold, BestGain;

            TBucket(TTreeNodePtr node, std::vector<size_t> &indexes, size_t begin, size_t end, size_t depth, size_t bestFeature, double bestThreshold, double bestGain)
                : Node(node)
                , Indexes(indexes)
                , Begin(begin)
                , End(end)
                , Depth(depth)
                , BestFeature(bestFeature)
                , BestThreshold(bestThreshold)
                , BestGain(bestGain)
            {
            }

            bool operator < (const TBucket &rgt) const {
                if (fabs(BestGain - rgt.BestGain) > 1e-10)
                    return BestGain > rgt.BestGain;
                if (Begin != rgt.Begin)
                    return Begin < rgt.Begin;
                return End < rgt.End;
            }
        };

        // returns best Gini Inpurity
        bool BestSplitForFeature(const TFeatures &x, const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes, size_t begin, size_t end, double &bestThreshold, double &bestGiniImpurity, double &bestGain) {
            size_t n = end - begin;
            if (n < 2)
                return false;
            //std::cout << "\t" << n << std::endl;
            std::vector<size_t> idx(n);
            for (size_t i = 0; i < n; ++i)
                idx[i] = i + begin;
            std::sort(idx.begin(), idx.end(), [&x, &indexes] (size_t a, size_t b) { return x[indexes[a]] < x[indexes[b]]; });
            bool result = false, median = false;
            bestThreshold = x[indexes[idx[0]]] - 1.0;
            std::vector<double> left(classCount), right = CountClasses(y, classCount, indexes, begin, end);
            double fullGiniImpurity = n * GiniImpurity(right) / 2.0;
            bestGiniImpurity = fullGiniImpurity;
            double medianThreshold = bestThreshold, medianGini = bestGiniImpurity;
            for (size_t i = 0; i + 1 < n; ++i) {
                size_t prev = indexes[idx[i]], next = indexes[idx[i + 1]];
                right[y[prev]] -= 1;
                left[y[prev]] += 1;
                if (fabs(x[prev] - x[next]) < 1e-10)
                    continue;
                size_t leftCount = i + 1, rightCount = n - i - 1;
                double threshold = (x[prev] + x[next]) / 2.0;
                double gini = (leftCount * GiniImpurity(left) + rightCount * GiniImpurity(right)) / 2.0;
                if (gini < bestGiniImpurity) {
                    result = true;
                    bestGiniImpurity = gini;
                    bestThreshold = threshold;
                }
                if (!median && i > n / 2) {
                    median = true;
                    medianThreshold = threshold;
                    medianGini = gini;
                }
            }
            if (median && !result) {
                result = true;
                bestThreshold = medianThreshold;
                bestGiniImpurity = medianGini;
            }
            if (result)
                bestGain = fullGiniImpurity - bestGiniImpurity;
            return result;
        }

        void SplitIndexes(const TFeatures &x, double threshold, std::vector<size_t> &indexes, size_t begin, size_t end, size_t &rightBegin) {
            size_t a = begin, b = end;
            while (a < b) {
                size_t &left = indexes[a], &right = indexes[b - 1];
                if (x[left] < threshold)
                    ++a;
                else if (x[right] >= threshold)
                    --b;
                else {
                    std::swap(left, right);
                }
            }
            rightBegin = b;
            std::sort(indexes.begin() + begin, indexes.begin() + rightBegin);
            std::sort(indexes.begin() + rightBegin, indexes.begin() + end);
        }

        bool BestSplitForNode(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, std::vector<size_t> &indexes, size_t begin, size_t end,
                       size_t &featureIndex, double &bestThreshold, double &bestGain, std::mt19937 &rng) {
            if (indexes.size() < 30)
                return false;
            bool result = false;
            double bestGiniImpurity = 0.0;
            bestGain = 0.0;
            //std::bernoulli_distribution bern(0.5);
            size_t featureCount = x.size();
            std::bernoulli_distribution bern(1.0 / sqrt(featureCount + 0.0));
            for (size_t i = 0; i < featureCount; ++i) {
                if (!bern(rng))
                    continue;
                double threshold = 0.0, giniImpurity = 0.0, gain = 0.0;
                bool splitFound = BestSplitForFeature(x[i], y, classCount, indexes, begin, end, threshold, giniImpurity, gain);
                if (!splitFound)
                    continue;
                if (!result || giniImpurity < bestGiniImpurity) {
                    result = true;
                    featureIndex = i;
                    bestThreshold = threshold;
                    bestGiniImpurity = giniImpurity;
                    bestGain = gain;
                }
            }
            if (!result)
                return false;
            //std::cout << "Best:\t" << featureIndex << "\t" << bestThreshold << "\t" << bestGiniImpurity << "\t" << begin << "\t" << rightBegin << "\t" << end << std::endl;
            return true;
        }

        bool SplitNodeRandom(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, std::vector<size_t> &indexes, size_t begin, size_t end,
                             size_t &featureIndex, double &threshold, size_t &rightBegin, std::mt19937 &rng) {
            {
                std::uniform_int_distribution<> dist(0, x.size() - 1);
                featureIndex = dist(rng);
            }
            size_t n = end - begin;
            std::vector<double> values(n);
            for (size_t i = 0; i < n; ++i)
                values[i] = x[featureIndex][indexes[i + begin]];
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
            if (values.size() < 2)
                return false;
            {
                std::uniform_int_distribution<> dist(0, values.size() - 2);
                size_t k = dist(rng);
                threshold = (values[k] + values[k + 1]) / 2.0;
            }
            SplitIndexes(x[featureIndex], threshold, indexes, begin, end, rightBegin);
            //std::cout << featureIndex << "\t" << threshold << std::endl;
            return true;
        }

        TTreeImplPtr TrainRandomTree(const std::vector<TFeatures> &x, const std::vector<size_t> &y, const std::vector<size_t> &g, size_t classCount, size_t maxDepth, size_t maxLeaves, double poolPart, std::mt19937 &rng) {
            time_t startTime = time(nullptr);
            size_t sampleCount = y.size();
            std::bernoulli_distribution bern(poolPart);
            std::vector<size_t> indexes;
            indexes.reserve(static_cast<size_t>(sampleCount * (poolPart + 0.01)));
            for (size_t i = 0; i < sampleCount; ) {
                bool take = bern(rng);
                if (take)
                    indexes.push_back(i);
                ++i;
                while (i < sampleCount && g[i] == g[i - 1]) {
                    if (take)
                        indexes.push_back(i);
                    ++i;
                }
            }
            sampleCount = indexes.size();
            TTreeNodePtr root(new TTreeNode);
            std::set<TBucket> queue;
            size_t featureIndex = 0;
            double threshold = 0.0, gain = 0.0;
            if (BestSplitForNode(x, y, classCount, indexes, 0, indexes.size(), featureIndex, threshold, gain, rng)) {
                queue.insert(TBucket(root, indexes, 0, indexes.size(), 0, featureIndex, threshold, gain));
            } else {
                TConstFeaturesPtr distr = std::make_shared<TFeatures>(std::move(ClassDistribution(y, classCount, indexes, 0, indexes.size())));
                //TConstFeaturesPtr distr = OneHot(WinnerClass(x, y, classCount, indexes, 0, indexes.size(), rng), classCount);
                root->SetAnswers(distr);
            }
            size_t leaves = 0;
            while (!queue.empty() && leaves + queue.size() < maxLeaves) {
                TBucket item(std::move(*queue.begin()));
                queue.erase(queue.begin());
                size_t rightBegin = item.End;
                SplitIndexes(x[item.BestFeature], item.BestThreshold, indexes, item.Begin, item.End, rightBegin);
                TTreeNodePtr left(new TTreeNode), right(new TTreeNode);
                item.Node->SplitNode(item.BestFeature, item.BestThreshold, left, right);
                if (rightBegin - item.Begin > 30 && item.Depth + 1 < maxDepth && BestSplitForNode(x, y, classCount, indexes, item.Begin, rightBegin, featureIndex, threshold, gain, rng)) {
                    queue.insert(TBucket(left, indexes, item.Begin, rightBegin, item.Depth + 1, featureIndex, threshold, gain));
                } else {
                    TConstFeaturesPtr distr = std::make_shared<TFeatures>(std::move(ClassDistribution(y, classCount, indexes, item.Begin, rightBegin)));
                    //TConstFeaturesPtr distr = OneHot(WinnerClass(x, y, classCount, indexes, item.Begin, rightBegin, rng), classCount);
                    left->SetAnswers(distr);
                    ++leaves;
                }
                if (item.End - rightBegin > 30 && item.Depth + 1 < maxDepth && BestSplitForNode(x, y, classCount, indexes, rightBegin, item.End, featureIndex, threshold, gain, rng)) {
                    queue.insert(TBucket(right, indexes, rightBegin, item.End, item.Depth + 1, featureIndex, threshold, gain));
                }
                else {
                    TConstFeaturesPtr distr = std::make_shared<TFeatures>(std::move(ClassDistribution(y, classCount, indexes, rightBegin, item.End)));
                    //TConstFeaturesPtr distr = OneHot(WinnerClass(x, y, classCount, indexes, rightBegin, item.End, rng), classCount);
                    right->SetAnswers(distr);
                    ++leaves;
                }
            }
            while (!queue.empty()) {
                TBucket item(std::move(*queue.begin()));
                queue.erase(queue.begin());
                TConstFeaturesPtr distr = std::make_shared<TFeatures>(std::move(ClassDistribution(y, classCount, indexes, item.Begin, item.End)));
                //TConstFeaturesPtr distr = OneHot(WinnerClass(x, y, classCount, indexes, item.Begin, item.End, rng), classCount);
                item.Node->SetAnswers(distr);
            }
            return TTreeImplPtr(new TTreeImpl(root));
        }

        TTreeImplPtr TrainFullRandomTree(const std::vector<TFeatures> &x, const std::vector<size_t> &y, const std::vector<size_t> &g, size_t classCount, size_t maxDepth, size_t maxLeaves, double poolPart, std::mt19937 &rng) {
            time_t startTime = time(nullptr);
            size_t sampleCount = y.size();
            std::bernoulli_distribution bern(poolPart);
            std::vector<size_t> indexes;
            indexes.reserve(static_cast<size_t>(sampleCount * (poolPart + 0.01)));
            for (size_t i = 0; i < sampleCount; ) {
                bool take = bern(rng);
                if (take)
                    indexes.push_back(i);
                ++i;
                while (i < sampleCount && g[i] == g[i - 1]) {
                    if (take)
                        indexes.push_back(i);
                    ++i;
                }
            }
            sampleCount = indexes.size();
            TTreeNodePtr root(new TTreeNode);
            std::list<TBucket> queue;
            if (!IsOnlyOneClass(y, indexes, 0, indexes.size()))
                queue.push_back(TBucket(root, indexes, 0, indexes.size(), 0, 0, 0.0, 0.0));
            size_t cnt = 1;
            while (!queue.empty()) {
                TBucket item(std::move(queue.front()));
                queue.pop_front();
                if (cnt >= maxLeaves || item.End - item.Begin <= 30 || item.Depth + 1 >= maxDepth) {
                    TConstFeaturesPtr distr = std::make_shared<TFeatures>(std::move(ClassDistribution(y, classCount, indexes, item.Begin, item.End)));
                    //TConstFeaturesPtr distr = OneHot(WinnerClass(x, y, classCount, leftIndexes, item.Begin, item.End, rng), classCount);
                    item.Node->SetAnswers(distr);
                    continue;
                }
                size_t featureIndex = 0;
                double threshold = 0.0;
                size_t rightBegin = item.End;
                if (SplitNodeRandom(x, y, classCount, item.Indexes, item.Begin, item.End, featureIndex, threshold, rightBegin, rng)) {
                    TTreeNodePtr left(new TTreeNode), right(new TTreeNode);
                    item.Node->SplitNode(featureIndex, threshold, left, right);
                    queue.push_back(TBucket(left, indexes, item.Begin, rightBegin, item.Depth + 1, 0, 0.0, 0.0));
                    queue.push_back(TBucket(right, indexes, rightBegin, item.End, item.Depth + 1, 0, 0.0, 0.0));
                    cnt += 2;
                }
                else {
                    TConstFeaturesPtr distr = std::make_shared<TFeatures>(std::move(ClassDistribution(y, classCount, indexes, item.Begin, item.End)));
                    //TConstFeaturesPtr distr = OneHot(WinnerClass(x, y, classCount, leftIndexes, item.Begin, item.End, rng), classCount);
                    item.Node->SetAnswers(distr);
                }
            }
            //std::cout << "\tDone, time: " << time(nullptr) - startTime << std::endl;
            //std::cout << std::endl;
            return TTreeImplPtr(new TTreeImpl(root));
        }

    } // namespace

    template<typename TTreeTrainer>
    TCalculatorPtr DoTrain(const TMiniBatch &x, const std::vector<size_t> &y, const std::vector<size_t> &g, size_t classCount, size_t maxDepth, size_t maxLeaves, double poolPart, size_t treeCount, TTreeTrainer treeTrainer) {
        std::mt19937 rng; //(time(nullptr));
        TForest forest(treeCount);
        for (size_t i = 0; i < treeCount; ++i)
            forest[i] = treeTrainer(x, y, g, classCount, maxDepth, maxLeaves, poolPart, rng);
        TCombinerPtr combiner(new TAverageCombiner);
        //TCombinerPtr combiner(new TMajorityVoteCombiner);
        return TCalculatorPtr(new TForestCalculator(std::move(forest), combiner));
    }

    TCalculatorPtr TrainRandomForest(const TMiniBatch &x, const std::vector<size_t> &y, const std::vector<size_t> &g, size_t classCount, size_t maxDepth, size_t maxLeaves, double poolPart, size_t treeCount) {
        return DoTrain(x, y, g, classCount, maxDepth, maxLeaves, poolPart, treeCount, &TrainRandomTree);
    }

    TCalculatorPtr TrainFullRandomForest(const TMiniBatch &x, const std::vector<size_t> &y, const std::vector<size_t> &g, size_t classCount, size_t maxDepth, size_t maxLeaves, double poolPart, size_t treeCount) {
        return DoTrain(x, y, g, classCount, maxDepth, maxLeaves, poolPart, treeCount, &TrainFullRandomTree);
    }

    TCalculatorPtr TrainCascadeForest(TMiniBatch &x, const std::vector<size_t> &y, const std::vector<size_t> &g, size_t classCount, size_t maxDepth, size_t maxLeaves, double poolPart, size_t treeCount, size_t levelCount) {
        time_t startTime = time(nullptr);
        std::mt19937 rng; //(time(nullptr));
        TCascadeForest cascade(levelCount, TForests(4, TForest(treeCount)));
        TCombinerPtr combiner(new TAverageCombiner);
        size_t featureCount = x.size(), instanceCount = x.front().size();
        x.resize(featureCount + 4 * classCount, TFeatures(instanceCount));
        for (size_t i = 0; i < levelCount; ++i) {
            std::cout << "Train level: " << i << ", time: " << time(nullptr) - startTime << std::endl;
            std::vector<std::thread> threads(4);
            for (size_t t = 0; t < 4; ++t) {
                std::uniform_int_distribution<size_t> dist;
                size_t rndSeed = dist(rng);
                std::thread thrd([treeCount, classCount, maxDepth, maxLeaves, poolPart, i, t, levelCount, rndSeed, &cascade, &x, &y, &g]() {
                    try {
                        std::mt19937 r(rndSeed);
                        for (size_t k = 0; k < treeCount; ++k) {
                            if (t < 3 /*|| i + 1 == levelCount*/)
                                cascade[i][t][k] = TrainRandomTree(x, y, g, classCount, maxDepth, maxLeaves, poolPart, r);
                            else
                                cascade[i][t][k] = TrainFullRandomTree(x, y, g, classCount, maxDepth, maxLeaves, poolPart, r);
                        }
                    }
                    catch (const std::exception &ex) {
                        std::cerr << "Exception caught: " << ex.what() << std::endl;
                    }
                });
                threads[t] = std::move(thrd);
            }
            for (size_t t = 0; t < 4; ++t)
                threads[t].join();
            std::cout << "Level trained, calculating features for next level, time: " << time(nullptr) - startTime  << std::endl;
            std::vector<TCalculatorPtr> calcs(4);
            for (size_t j = 0; j < 4; ++j) {
                calcs[j] = TCalculatorPtr(new TForestCalculator(TForest(cascade[i][j]), combiner));
            }
            std::vector<std::pair<int, double>> answers(instanceCount);
            for (size_t t = 0; t < 4; ++t) {
                std::thread thrd([t, instanceCount, featureCount, classCount, &combiner, &answers, &x, &y, &calcs] () {
                    try {
                        for (size_t j = instanceCount / 4 * t; j < std::min(instanceCount, instanceCount / 4 * (t + 1)); ++j) {
                            TFeatures features(featureCount + 4 * classCount);
                            for (size_t k = 0; k < featureCount + 4 * classCount; ++k)
                                features[k] = x[k][j];
                            std::vector<TConstFeaturesPtr> scores(4);
                            for (size_t k = 0; k < 4; ++k) {
                                scores[k] = std::make_shared<TFeatures>(calcs[k]->Calculate(features));
                                for (size_t u = 0; u < classCount; ++u)
                                    x[featureCount + k * classCount + u][j] = (*scores[k])[u];
                            }
                            scores.resize(3);
                            TFeatures res;
                            combiner->Combine(scores, res);
                            answers[j] = std::make_pair(y[j], res[1]);
                        }
                    }
                    catch (const std::exception &ex) {
                        std::cerr << "Exception caught: " << ex.what() << std::endl;
                    }
                });
                threads[t] = std::move(thrd);
            }
            for (size_t t = 0; t < 4; ++t)
                threads[t].join();
            std::cout << "Train AUC: " << AUC(std::move(answers)) << ", time: " << time(nullptr) - startTime << std::endl << std::endl;
        }
        return std::make_shared<TCascadeForestCalculator>(std::move(cascade), combiner);
    }

} // namespace NGCForest

