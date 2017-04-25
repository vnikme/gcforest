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

        std::vector<double> ThresholdsInRange(const TFeatures &x, const std::vector<size_t> &indexes, size_t begin, size_t end) {
            size_t n = end - begin;
            std::vector<double> values(n);
            for (size_t i = 0; i < n; ++i)
                values[i] = x[indexes[i + begin]];
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
            for (size_t i = 0; i + 1 < values.size(); ++i)
                values[i] = (values[i] + values[i + 1]) / 2.0;
            values.pop_back();
            return values;
        }

        bool SplitNodeRandom(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, std::vector<size_t> &indexes, size_t begin, size_t end,
                             size_t &featureIndex, double &threshold, size_t &rightBegin, std::mt19937 &rng) {
            {
                std::uniform_int_distribution<> dist(0, x.size() - 1);
                featureIndex = dist(rng);
            }
            std::vector<double> thresholds = ThresholdsInRange(x[featureIndex], indexes, begin, end);
            if (thresholds.empty())
                return false;
            {
                std::uniform_int_distribution<> dist(0, thresholds.size() - 1);
                size_t k = dist(rng);
                threshold = thresholds[k];
            }
            SplitIndexes(x[featureIndex], threshold, indexes, begin, end, rightBegin);
            //std::cout << featureIndex << "\t" << threshold << std::endl;
            return true;
        }

        std::vector<size_t> GetSampleFromPool(const std::vector<size_t> &g, double poolPart, std::mt19937 &rng) {
            size_t sampleCount = g.size();
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
            return indexes;
        }

        TTreeImplPtr TrainRandomTree(const std::vector<TFeatures> &x, const std::vector<size_t> &y, const std::vector<size_t> &g, size_t classCount, size_t maxDepth, size_t maxLeaves, double poolPart, std::mt19937 &rng) {
            time_t startTime = time(nullptr);
            std::vector<size_t> indexes = GetSampleFromPool(g, poolPart, rng);
            size_t sampleCount = indexes.size();
            TTreeNodePtr root(new TTreeNode);
            std::set<TBucket> queue;
            size_t featureIndex = 0;
            double threshold = 0.0, gain = 0.0;
            if (BestSplitForNode(x, y, classCount, indexes, 0, indexes.size(), featureIndex, threshold, gain, rng)) {
                queue.insert(TBucket(root, indexes, 0, indexes.size(), 0, featureIndex, threshold, gain));
            } else {
                TFeatures distr = std::move(ClassDistribution(y, classCount, indexes, 0, indexes.size()));
                //TConstFeaturesPtr distr = OneHot(WinnerClass(x, y, classCount, indexes, 0, indexes.size(), rng), classCount);
                root->SetAnswers(std::move(distr));
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
                    TFeatures distr = std::move(ClassDistribution(y, classCount, indexes, item.Begin, rightBegin));
                    //TConstFeaturesPtr distr = OneHot(WinnerClass(x, y, classCount, indexes, item.Begin, rightBegin, rng), classCount);
                    left->SetAnswers(std::move(distr));
                    ++leaves;
                }
                if (item.End - rightBegin > 30 && item.Depth + 1 < maxDepth && BestSplitForNode(x, y, classCount, indexes, rightBegin, item.End, featureIndex, threshold, gain, rng)) {
                    queue.insert(TBucket(right, indexes, rightBegin, item.End, item.Depth + 1, featureIndex, threshold, gain));
                }
                else {
                    TFeatures distr = std::move(ClassDistribution(y, classCount, indexes, rightBegin, item.End));
                    //TConstFeaturesPtr distr = OneHot(WinnerClass(x, y, classCount, indexes, rightBegin, item.End, rng), classCount);
                    right->SetAnswers(std::move(distr));
                    ++leaves;
                }
            }
            while (!queue.empty()) {
                TBucket item(std::move(*queue.begin()));
                queue.erase(queue.begin());
                TFeatures distr = std::move(ClassDistribution(y, classCount, indexes, item.Begin, item.End));
                //TConstFeaturesPtr distr = OneHot(WinnerClass(x, y, classCount, indexes, item.Begin, item.End, rng), classCount);
                item.Node->SetAnswers(std::move(distr));
            }
            return TTreeImplPtr(new TDynamicTreeImpl(root));
        }

        TTreeImplPtr TrainFullRandomTree(const std::vector<TFeatures> &x, const std::vector<size_t> &y, const std::vector<size_t> &g, size_t classCount, size_t maxDepth, size_t maxLeaves, double poolPart, std::mt19937 &rng) {
            time_t startTime = time(nullptr);
            std::vector<size_t> indexes = GetSampleFromPool(g, poolPart, rng);
            size_t sampleCount = indexes.size();
            TTreeNodePtr root(new TTreeNode);
            std::list<TBucket> queue;
            if (!IsOnlyOneClass(y, indexes, 0, indexes.size()))
                queue.push_back(TBucket(root, indexes, 0, indexes.size(), 0, 0, 0.0, 0.0));
            size_t cnt = 1;
            while (!queue.empty()) {
                TBucket item(std::move(queue.front()));
                queue.pop_front();
                if (cnt >= maxLeaves || item.End - item.Begin <= 30 || item.Depth + 1 >= maxDepth) {
                    TFeatures distr = std::move(ClassDistribution(y, classCount, indexes, item.Begin, item.End));
                    //TConstFeaturesPtr distr = OneHot(WinnerClass(x, y, classCount, leftIndexes, item.Begin, item.End, rng), classCount);
                    item.Node->SetAnswers(std::move(distr));
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
                    TFeatures distr = std::move(ClassDistribution(y, classCount, indexes, item.Begin, item.End));
                    //TConstFeaturesPtr distr = OneHot(WinnerClass(x, y, classCount, leftIndexes, item.Begin, item.End, rng), classCount);
                    item.Node->SetAnswers(std::move(distr));
                }
            }
            //std::cout << "\tDone, time: " << time(nullptr) - startTime << std::endl;
            //std::cout << std::endl;
            return TTreeImplPtr(new TDynamicTreeImpl(root));
        }

        bool BestSplitForFeature(const TFeatures &x, const std::vector<size_t> &y, size_t classCount,
                                 std::vector<size_t> &indexes,
                                 const std::vector<size_t> &bins,
                                 double &bestThreshold, double &bestGini) {
            size_t sampleCount = indexes.size();
            std::vector<double> values = ThresholdsInRange(x, indexes, 0, indexes.size());
            std::vector<size_t> iters(bins.size() - 1);
            std::vector<std::vector<double>> lefts(bins.size() - 1, std::vector<double>(classCount)), rights(bins.size() - 1, std::vector<double>(classCount));
            for (size_t i = 0; i + 1 < bins.size(); ++i) {
                iters[i] = bins[i];
                std::sort(indexes.begin() + bins[i], indexes.begin() + bins[i + 1], [&x] (size_t a, size_t b) { return x[a] < x[b]; });
                for (size_t j = bins[i]; j < bins[i + 1]; ++j)
                    rights[i][y[indexes[j]]] += 1;
            }
            bool result = false;
            for (double val : values) {
                double gini = 0.0;
                for (size_t i = 0; i + 1 < bins.size(); ++i) {
                    for (; iters[i] < bins[i + 1] && x[indexes[iters[i]]] < val; ++iters[i]) {
                        lefts[i][y[indexes[iters[i]]]] += 1;
                        rights[i][y[indexes[iters[i]]]] -= 1;
                    }
                    gini += (GiniImpurity(lefts[i]) * (iters[i] - bins[i]) + GiniImpurity(rights[i]) * (bins[i + 1] - iters[i]));
                }
                if (gini < bestGini) {
                    bestGini = gini;
                    bestThreshold = val;
                    result = true;
                }
            }
            return result;
        }

        TTreeImplPtr TrainObliviousTree(const std::vector<TFeatures> &x, const std::vector<size_t> &y, const std::vector<size_t> &g, size_t classCount, size_t maxDepth, bool random, double poolPart, std::mt19937 &rng) {
            time_t startTime = time(nullptr);
            std::vector<size_t> indexes = GetSampleFromPool(g, poolPart, rng);
            size_t sampleCount = indexes.size();
            std::vector<size_t> bins;
            bins.push_back(0);
            bins.push_back(indexes.size());
            std::vector<size_t> features;
            std::vector<double> thresholds;
            std::vector<TFeatures> answers;
            answers.push_back(ClassDistribution(y, classCount, indexes, 0, indexes.size()));
            double fullGini = GiniImpurity(CountClasses(y, classCount, indexes, 0, indexes.size())) * indexes.size();
            size_t featureCount = x.size();
            std::bernoulli_distribution bern(1.0 / sqrt(featureCount + 0.0));
            for (size_t level = 0; level < maxDepth; ++level) {
                double bestGini = fullGini + 1.0, bestThreshold = 0.0;
                size_t bestFeature = featureCount;
                if (!random) {
                    for (size_t feature = 0; feature < featureCount; ++feature) {
                        if (!bern(rng))
                            continue;
                        double threshold = 0.0, gini = fullGini + 1.0;
                        if (!BestSplitForFeature(x[feature], y, classCount, indexes, bins, threshold, gini))
                            continue;
                        if (bestFeature == featureCount || gini < bestGini) {
                            bestFeature = feature;
                            bestThreshold = threshold;
                            bestGini = gini;
                        }
                    }
                    if (bestGini >= fullGini)
                        break;
                } else {
                    std::uniform_int_distribution<> dist1(0, featureCount - 1);
                    bestFeature = dist1(rng);
                    std::vector<double> thresholds = ThresholdsInRange(x[bestFeature], indexes, 0, indexes.size());
                    if (thresholds.empty())
                        break;
                    std::uniform_int_distribution<> dist2(0, thresholds.size() - 1);
                    size_t k = dist2(rng);
                    bestThreshold = thresholds[k];
                }
                features.push_back(bestFeature);
                thresholds.push_back(bestThreshold);
                fullGini = bestGini;
                std::vector<size_t> nextBins;
                nextBins.reserve((bins.size() - 1) * 2 + 1);
                std::vector<TFeatures> nextAnswers;
                nextAnswers.reserve(answers.size() * 2);
                for (size_t i = 0; i + 1 < bins.size(); ++i) {
                    nextBins.push_back(bins[i]);
                    nextBins.emplace_back();
                    SplitIndexes(x[bestFeature], bestThreshold, indexes, bins[i], bins[i + 1], nextBins.back());
                    if (nextBins.back() - bins[i] > 30)
                        nextAnswers.push_back(ClassDistribution(y, classCount, indexes, bins[i], nextBins.back()));
                    else
                        nextAnswers.push_back(answers[i]);
                    if (bins[i + 1] - nextBins.back() > 30)
                        nextAnswers.push_back(ClassDistribution(y, classCount, indexes, nextBins.back(), bins[i + 1]));
                    else
                        nextAnswers.push_back(answers[i]);
                }
                nextBins.push_back(bins.back());
                bins.swap(nextBins);
                answers.swap(nextAnswers);
            }
            return TTreeImplPtr(new TObliviousTreeImpl(features, thresholds, answers));
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
                            if (t < 4 /*|| i + 1 == levelCount*/)
                                //cascade[i][t][k] = TrainRandomTree(x, y, g, classCount, maxDepth, maxLeaves, poolPart, r);
                                cascade[i][t][k] = TrainObliviousTree(x, y, g, classCount, 5, false, poolPart, r);
                            else
                                //cascade[i][t][k] = TrainFullRandomTree(x, y, g, classCount, maxDepth, maxLeaves, poolPart, r);
                                cascade[i][t][k] = TrainObliviousTree(x, y, g, classCount, 5, true, poolPart, r);
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
                            std::vector<TFeatures> scores(4);
                            std::vector<const TFeatures *> pscores(4);
                            for (size_t k = 0; k < 4; ++k) {
                                scores[k] = calcs[k]->Calculate(features);
                                pscores[k] = &scores[k];
                                for (size_t u = 0; u < classCount; ++u)
                                    x[featureCount + k * classCount + u][j] = scores[k][u];
                            }
                            //pscores.resize(3);
                            TFeatures res;
                            combiner->Combine(pscores, res);
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

