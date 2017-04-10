#include "forest.h"
#include "forest_impl.h"
#include "train.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <list>
#include <random>


namespace NGCForest {

    namespace {

        std::vector<double> CountClasses(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes, size_t begin, size_t end) {
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

        std::vector<double> ClassDistribution(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes, size_t begin, size_t end) {
            std::vector<double> f = CountClasses(x, y, classCount, indexes, begin, end);
            double sum = 1e-38;
            for (double val : f)
                sum += val;
            for (double &val : f)
                val /= sum;
            return f;
        }

        size_t WinnerClass(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes, size_t begin, size_t end, std::mt19937 &rng) {
            std::vector<double> f(ClassDistribution(x, y, classCount, indexes, begin, end));
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

            TBucket(TTreeNodePtr node, std::vector<size_t> &indexes, size_t begin, size_t end, size_t depth)
                : Node(node)
                , Indexes(indexes)
				, Begin(begin)
				, End(end)
                , Depth(depth)
            {
            }
        };

        // returns best Gini Inpurity
        bool BestSplitForFeature(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes, size_t begin, size_t end, size_t featureIndex, double &bestThreshold, double &bestGiniImpurity) {
			size_t n = end - begin;
            if (n < 2)
                return false;
            //std::cout << "\t" << n << std::endl;
            std::vector<size_t> idx(n);
            for (size_t i = 0; i < n; ++i)
                idx[i] = i + begin;
            std::sort(idx.begin(), idx.end(), [&x, &indexes, featureIndex] (size_t a, size_t b) { return x[indexes[a]][featureIndex] < x[indexes[b]][featureIndex]; });
            bool result = false, median = false;
            bestThreshold = x[indexes[idx[0]]][featureIndex] - 1.0;
            std::vector<double> left(classCount), right = CountClasses(x, y, classCount, indexes, begin, end);
            bestGiniImpurity = GiniImpurity(right) / 2.0;
            double medianThreshold = bestThreshold, medianGini = bestGiniImpurity;
            for (size_t i = 0; i + 1 < n; ++i) {
                size_t prev = indexes[idx[i]], next = indexes[idx[i + 1]];
                right[y[prev]] -= 1;
                left[y[prev]] += 1;
                if (fabs(x[prev][featureIndex] - x[next][featureIndex]) < 1e-10)
                    continue;
                double threshold = (x[prev][featureIndex] + x[next][featureIndex]) / 2.0;
                double gini = (GiniImpurity(left) + GiniImpurity(right)) / 2.0;
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
            return result;
        }

        void SplitIndexes(const std::vector<TFeatures> &x, size_t featureIndex, double threshold, std::vector<size_t> &indexes, size_t begin, size_t end, size_t &rightBegin) {
            size_t a = begin, b = end;
            while (a < b) {
                size_t &left = indexes[a], &right = indexes[b - 1];
                if (x[left][featureIndex] < threshold)
                    ++a;
                else if (x[right][featureIndex] >= threshold)
                    --b;
                else {
                    std::swap(left, right);
                }
            }
            rightBegin = b;
            //std::sort(indexes.begin() + begin, indexes.begin() + rightBegin);
            //std::sort(indexes.begin() + rightBegin, indexes.begin() + end);
        }

        bool SplitNode(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, std::vector<size_t> &indexes, size_t begin, size_t end,
                       size_t &featureIndex, double &bestThreshold, size_t &rightBegin, std::mt19937 &rng) {
            bool result = false;
            double bestGiniImpurity = 0.0;
            //std::bernoulli_distribution bern(0.5);
            std::bernoulli_distribution bern(1.0 / sqrt(x[0].size() + 0.0));
            for (size_t i = 0; i < x[0].size(); ++i) {
                if (!bern(rng))
                    continue;
                double threshold = 0.0, giniImpurity = 0.0;
                bool splitFound = BestSplitForFeature(x, y, classCount, indexes, begin, end, i, threshold, giniImpurity);
                if (!splitFound)
                    continue;
                if (!result || giniImpurity < bestGiniImpurity) {
                    result = true;
                    featureIndex = i;
                    bestThreshold = threshold;
                    bestGiniImpurity = giniImpurity;
                }
            }
            if (!result)
                return false;
            SplitIndexes(x, featureIndex, bestThreshold, indexes, begin, end, rightBegin);
            //std::cout << "Best:\t" << featureIndex << "\t" << bestThreshold << "\t" << bestGiniImpurity << std::endl;
            return true;
        }

        bool SplitNodeRandom(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, std::vector<size_t> &indexes, size_t begin, size_t end,
                             size_t &featureIndex, double &threshold, size_t &rightBegin, std::mt19937 &rng) {
            {
                std::uniform_int_distribution<> dist(0, x[0].size() - 1);
                featureIndex = dist(rng);
            }
            size_t n = end - begin;
            std::vector<double> values(n);
            for (size_t i = 0; i < n; ++i)
                values[i] = x[indexes[i + begin]][featureIndex];
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
            if (values.size() < 2)
                return false;
            {
                std::uniform_int_distribution<> dist(0, values.size() - 2);
                size_t k = dist(rng);
                threshold = (values[k] + values[k + 1]) / 2.0;
            }
            SplitIndexes(x, featureIndex, threshold, indexes, begin, end, rightBegin);
            //std::cout << featureIndex << "\t" << threshold << std::endl;
            return true;
        }

        TTreeImplPtr TrainRandomTree(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, size_t maxDepth, std::mt19937 &rng) {
            time_t startTime = time(nullptr);
            size_t sampleCount = x.size();
            if (sampleCount > 10000)
                sampleCount = 10000 + (sampleCount - 10000) / 10;
            //std::cout << "\tTrain random tree, sampleCount: " << sampleCount << ", time: " << time(nullptr) - startTime << std::endl;
            std::uniform_int_distribution<> dist(0, sampleCount - 1);
            std::vector<size_t> indexes(sampleCount);
            for (size_t i = 0; i < sampleCount; ++i) {
                indexes[i] = dist(rng);
            }
            TConstFeaturesPtr distr = std::make_shared<TFeatures>(std::move(ClassDistribution(x, y, classCount, indexes, 0, indexes.size())));
            TTreeNodePtr root(new TTreeNode(distr));
            std::list<TBucket> queue;
            if (!IsOnlyOneClass(y, indexes, 0, indexes.size()))
                queue.push_back(TBucket(root, indexes, 0, indexes.size(), 0));
            int cnt = 0;
            while (!queue.empty()) {
                ++cnt;
                TBucket item(std::move(queue.front()));
                queue.pop_front();
                size_t featureIndex = 0;
                double threshold = 0.0;
				size_t rightBegin = item.End;
                if (SplitNode(x, y, classCount, item.Indexes, item.Begin, item.End, featureIndex, threshold, rightBegin, rng)) {
                    TConstFeaturesPtr leftDistr = std::make_shared<TFeatures>(std::move(ClassDistribution(x, y, classCount, indexes, item.Begin, rightBegin)));
                    TConstFeaturesPtr rightDistr = std::make_shared<TFeatures>(std::move(ClassDistribution(x, y, classCount, indexes, rightBegin, item.End)));
                    //TConstFeaturesPtr leftDistr = OneHot(WinnerClass(x, y, classCount, leftIndexes, item.Begin, rightBegin, rng), classCount);
                    //TConstFeaturesPtr rightDistr = OneHot(WinnerClass(x, y, classCount, rightIndexes, rightBegin, item.End, rng), classCount);
                    TTreeNodePtr left(new TTreeNode(leftDistr)), right(new TTreeNode(rightDistr));
                    item.Node->SplitNode(featureIndex, threshold, left, right);
                    if (item.Depth + 1 < maxDepth && !IsOnlyOneClass(y, indexes, item.Begin, rightBegin))
                        queue.push_back(TBucket(left, indexes, item.Begin, rightBegin, item.Depth + 1));
                    if (item.Depth + 1 < maxDepth && !IsOnlyOneClass(y, indexes, rightBegin, item.End))
                        queue.push_back(TBucket(right, indexes, rightBegin, item.End, item.Depth + 1));
                }
            }
            //std::cout << cnt << std::endl;
            //std::cout << "\tDone, time: " << time(nullptr) - startTime << std::endl;
            //std::cout << std::endl;
            return TTreeImplPtr(new TTreeImpl(root));
        }

        TTreeImplPtr TrainFullRandomTree(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, size_t maxDepth, std::mt19937 &rng) {
            time_t startTime = time(nullptr);
            size_t sampleCount = x.size();
            if (sampleCount > 10000)
                sampleCount = 10000 + (sampleCount - 10000) / 10;
            //std::cout << "\tTrain full random tree, sampleCount: " << sampleCount << ", time: " << time(nullptr) - startTime << std::endl;
            std::uniform_int_distribution<> dist(0, sampleCount - 1);
            std::vector<size_t> indexes(sampleCount);
            for (size_t i = 0; i < sampleCount; ++i) {
                indexes[i] = dist(rng);
            }
            TConstFeaturesPtr wholeDistr = std::make_shared<TFeatures>(std::move(ClassDistribution(x, y, classCount, indexes, 0, indexes.size())));
            TTreeNodePtr root(new TTreeNode(wholeDistr));
            std::list<TBucket> queue;
            if (!IsOnlyOneClass(y, indexes, 0, indexes.size()))
                queue.push_back(TBucket(root, indexes, 0, indexes.size(), 0));
            while (!queue.empty()) {
                TBucket item(std::move(queue.front()));
                queue.pop_front();
                size_t featureIndex = 0;
                double threshold = 0.0;
                size_t rightBegin = item.End;
                if (SplitNodeRandom(x, y, classCount, item.Indexes, item.Begin, item.End, featureIndex, threshold, rightBegin, rng)) {
                    TConstFeaturesPtr leftDistr = wholeDistr, rightDistr = wholeDistr;
                    if (item.Depth + 1 >= maxDepth || IsOnlyOneClass(y, indexes, item.Begin, rightBegin)) {
                        leftDistr = std::make_shared<TFeatures>(std::move(ClassDistribution(x, y, classCount, indexes, item.Begin, rightBegin)));
                        //std::cout << "Gini left: " << GiniImpurity(*leftDistr) << std::endl;
                    }
                    if (item.Depth + 1 >= maxDepth || IsOnlyOneClass(y, indexes, rightBegin, item.End)) {
                        rightDistr = std::make_shared<TFeatures>(std::move(ClassDistribution(x, y, classCount, indexes, rightBegin, item.End)));
                        //std::cout << "Gini right: " << GiniImpurity(*rightDistr) << std::endl;
                    }
                    TTreeNodePtr left(new TTreeNode(leftDistr)), right(new TTreeNode(rightDistr));
                    item.Node->SplitNode(featureIndex, threshold, left, right);
                    if (item.Depth < maxDepth && !IsOnlyOneClass(y, indexes, item.Begin, rightBegin))
                        queue.push_back(TBucket(left, indexes, item.Begin, rightBegin, item.Depth + 1));
                    if (item.Depth < maxDepth && !IsOnlyOneClass(y, indexes, rightBegin, item.End))
                        queue.push_back(TBucket(right, indexes, rightBegin, item.End, item.Depth + 1));
                }
            }
            //std::cout << "\tDone, time: " << time(nullptr) - startTime << std::endl;
            //std::cout << std::endl;
            return TTreeImplPtr(new TTreeImpl(root));
        }

    } // namespace

    template<typename TTreeTrainer>
    TCalculatorPtr DoTrain(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, size_t maxDepth, size_t treeCount, TTreeTrainer treeTrainer) {
        std::mt19937 rng; //(time(nullptr));
        TForest forest(treeCount);
        for (size_t i = 0; i < treeCount; ++i)
            forest[i] = treeTrainer(x, y, classCount, maxDepth, rng);
        TCombinerPtr combiner(new TAverageCombiner);
        //TCombinerPtr combiner(new TMajorityVoteCombiner);
        return TCalculatorPtr(new TForestCalculator(std::move(forest), combiner));
    }

    TCalculatorPtr TrainRandomForest(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, size_t maxDepth, size_t treeCount) {
        return DoTrain(x, y, classCount, maxDepth, treeCount, &TrainRandomTree);
    }

    TCalculatorPtr TrainFullRandomForest(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, size_t maxDepth, size_t treeCount) {
        return DoTrain(x, y, classCount, maxDepth, treeCount, &TrainFullRandomTree);
    }

    TCalculatorPtr TrainCascadeForest(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, size_t maxDepth, size_t treeCount, size_t levelCount) {
        time_t startTime = time(nullptr);
        std::mt19937 rng; //(time(nullptr));
        TCascadeForest cascade(levelCount, TForests(4, TForest(treeCount)));
        TCombinerPtr combiner(new TAverageCombiner);
        std::vector<std::vector<TFeatures>> prevLevel(x.size(), std::vector<TFeatures>(4));
        for (size_t i = 0; i < levelCount; ++i) {
            std::cout << "Train level: " << i << ", time: " << time(nullptr) - startTime << std::endl;
            std::vector<TFeatures> features(x);
            for (size_t j = 0; j < x.size(); ++j) {
                for (const TFeatures &prev : prevLevel[j]) {
                    features[j].insert(features[j].end(), prev.begin(), prev.end());
                }
            }
            for (size_t j = 0; j < 2; ++j) {
                for (size_t k = 0; k < treeCount; ++k) {
                    cascade[i][j][k] = TrainRandomTree(features, y, classCount, maxDepth, rng);
                    cascade[i][2 + j][k] = TrainFullRandomTree(features, y, classCount, maxDepth, rng);
                }
            }
            std::cout << "Level trained, calculating features for next level, time: " << time(nullptr) - startTime  << std::endl;
            std::vector<TCalculatorPtr> calcs(4);
            for (size_t j = 0; j < 4; ++j) {
                calcs[j] = TCalculatorPtr(new TForestCalculator(TForest(cascade[i][j]), combiner));
            }
            for (size_t j = 0; j < x.size(); ++j) {
                for (size_t k = 0; k < 4; ++k) {
                    prevLevel[j][k] = calcs[k]->Calculate(features[j]);
                }
            }
            std::cout << "Done, time: " << time(nullptr) - startTime  << std::endl << std::endl;
        }
        return TCalculatorPtr(new TCascadeForestCalculator(std::move(cascade), combiner));
    }

} // namespace NGCForest

