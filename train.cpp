#include "forest.h"
#include "forest_impl.h"
#include "train.h"

#include <boost/random.hpp>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <list>


namespace NGCForest {

    namespace {

        std::vector<double> CountClasses(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes) {
            std::vector<double> f(classCount);
            for (size_t i : indexes) {
                f[y[i]] += 1;
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

        std::vector<double> ClassDistribution(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes) {
            std::vector<double> f = CountClasses(x, y, classCount, indexes);
            double sum = 1e-38;
            for (double val : f)
                sum += val;
            for (double &val : f)
                val /= sum;
            return f;
        }

        size_t WinnerClass(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes, boost::random::mt19937 &rng) {
            std::vector<double> f(ClassDistribution(x, y, classCount, indexes));
            boost::random::discrete_distribution<> dist(f);
            return dist(rng);
        }

        bool IsOnlyOneClass(const std::vector<size_t> &y, const std::vector<size_t> indexes) {
            if (indexes.empty())
                return true;
            size_t firstClass = y[indexes[0]];
            for (size_t i = 1; i < indexes.size(); ++i) {
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
            std::vector<size_t> Indexes;
            size_t Depth;

            TBucket(TTreeNodePtr node, std::vector<size_t> &&indexes, size_t depth)
                : Node(node)
                , Indexes(std::move(indexes))
                , Depth(depth)
            {
            }
        };

        // returns best Gini Inpurity
        bool BestSplitForFeature(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes, size_t featureIndex, double &bestThreshold, double &bestGiniImpurity) {
            if (indexes.size() < 2)
                return false;
            size_t n = indexes.size();
            std::vector<size_t> idx(n);
            for (size_t i = 0; i < n; ++i)
                idx[i] = i;
            std::sort(idx.begin(), idx.end(), [&x, &indexes, featureIndex] (size_t a, size_t b) { return x[indexes[a]][featureIndex] < x[indexes[b]][featureIndex]; });
            bool result = false, median = false;
            bestThreshold = x[indexes[idx[0]]][featureIndex] - 1.0;
            std::vector<double> left(classCount), right = CountClasses(x, y, classCount, indexes);
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

        bool SplitNode(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes,
                       size_t &featureIndex, double &bestThreshold,
                       std::vector<size_t> &leftIndexes, std::vector<size_t> &rightIndexes,
                       boost::random::mt19937 &rng) {
            bool result = false;
            double bestGiniImpurity = 0.0;
            //std::bernoulli_distribution bern(0.5);
            std::bernoulli_distribution bern(1.0 / sqrt(x[0].size() + 0.0));
            for (size_t i = 0; i < x[0].size(); ++i) {
                if (!bern(rng))
                    continue;
                double threshold = 0.0, giniImpurity = 0.0;
                bool splitFound = BestSplitForFeature(x, y, classCount, indexes, i, threshold, giniImpurity);
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
            std::cout << "Best:\t" << featureIndex << "\t" << bestThreshold << "\t" << bestGiniImpurity << std::endl;
            for (size_t i : indexes) {
                if (x[i][featureIndex] < bestThreshold)
                    leftIndexes.push_back(i);
                else
                    rightIndexes.push_back(i);
            }
            return true;
        }

        bool SplitNodeRandom(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, const std::vector<size_t> &indexes,
                             size_t &featureIndex, double &threshold,
                             std::vector<size_t> &leftIndexes, std::vector<size_t> &rightIndexes,
                             boost::random::mt19937 &rng) {
            {
                boost::random::uniform_int_distribution<> dist(0, x[0].size() - 1);
                featureIndex = dist(rng);
            }
            std::vector<double> values(indexes.size());
            for (size_t i = 0; i < indexes.size(); ++i)
                values[i] = x[indexes[i]][featureIndex];
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
            {
                boost::random::uniform_int_distribution<> dist(0, values.size() - 2);
                size_t k = dist(rng);
                threshold = (values[k] + values[k + 1]) / 2.0;
            }
            std::cout << featureIndex << "\t" << threshold << std::endl;
            for (size_t i : indexes) {
                if (x[i][featureIndex] < threshold)
                    leftIndexes.push_back(i);
                else
                    rightIndexes.push_back(i);
            }
            return true;
        }

        TTreeImplPtr TrainRandomTree(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, size_t maxDepth, boost::random::mt19937 &rng) {
            size_t sampleCount = x.size();
            boost::random::uniform_int_distribution<> dist(0, sampleCount - 1);
            std::vector<size_t> indexes(sampleCount);
            for (size_t i = 0; i < sampleCount; ++i) {
                indexes[i] = dist(rng);
            }
            TConstFeaturesPtr distr = std::make_shared<TFeatures>(std::move(ClassDistribution(x, y, classCount, indexes)));
            TTreeNodePtr root(new TTreeNode(distr));
            std::list<TBucket> queue;
            if (!IsOnlyOneClass(y, indexes))
                queue.push_back(TBucket(root, std::move(indexes), 0));
            while (!queue.empty()) {
                TBucket item(std::move(queue.front()));
                queue.pop_front();
                size_t featureIndex = 0;
                double threshold = 0.0;
                std::vector<size_t> leftIndexes, rightIndexes;
                if (SplitNode(x, y, classCount, item.Indexes, featureIndex, threshold, leftIndexes, rightIndexes, rng)) {
                    TConstFeaturesPtr leftDistr = std::make_shared<TFeatures>(std::move(ClassDistribution(x, y, classCount, leftIndexes)));
                    TConstFeaturesPtr rightDistr = std::make_shared<TFeatures>(std::move(ClassDistribution(x, y, classCount, rightIndexes)));
                    //TConstFeaturesPtr leftDistr = OneHot(WinnerClass(x, y, classCount, leftIndexes, rng), classCount);
                    //TConstFeaturesPtr rightDistr = OneHot(WinnerClass(x, y, classCount, rightIndexes, rng), classCount);
                    TTreeNodePtr left(new TTreeNode(leftDistr)), right(new TTreeNode(rightDistr));
                    item.Node->SplitNode(featureIndex, threshold, left, right);
                    if (item.Depth + 1 < maxDepth && !IsOnlyOneClass(y, leftIndexes))
                        queue.push_back(TBucket(left, std::move(leftIndexes), item.Depth + 1));
                    if (item.Depth + 1 < maxDepth && !IsOnlyOneClass(y, rightIndexes))
                        queue.push_back(TBucket(right, std::move(rightIndexes), item.Depth + 1));
                }
            }
            std::cout << std::endl;
            return TTreeImplPtr(new TTreeImpl(root));
        }

        TTreeImplPtr TrainFullRandomTree(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, size_t maxDepth, boost::random::mt19937 &rng) {
            size_t sampleCount = x.size();
            boost::random::uniform_int_distribution<> dist(0, sampleCount - 1);
            std::vector<size_t> indexes(sampleCount);
            for (size_t i = 0; i < sampleCount; ++i) {
                indexes[i] = dist(rng);
            }
            TConstFeaturesPtr wholeDistr = std::make_shared<TFeatures>(std::move(ClassDistribution(x, y, classCount, indexes)));
            TTreeNodePtr root(new TTreeNode(wholeDistr));
            std::list<TBucket> queue;
            if (!IsOnlyOneClass(y, indexes))
                queue.push_back(TBucket(root, std::move(indexes), 0));
            while (!queue.empty()) {
                TBucket item(std::move(queue.front()));
                queue.pop_front();
                size_t featureIndex = 0;
                double threshold = 0.0;
                std::vector<size_t> leftIndexes, rightIndexes;
                if (SplitNodeRandom(x, y, classCount, item.Indexes, featureIndex, threshold, leftIndexes, rightIndexes, rng)) {
                    TConstFeaturesPtr leftDistr = wholeDistr, rightDistr = wholeDistr;
                    if (item.Depth + 1 >= maxDepth || IsOnlyOneClass(y, leftIndexes)) {
                        leftDistr = std::make_shared<TFeatures>(std::move(ClassDistribution(x, y, classCount, leftIndexes)));
                        std::cout << "Gini left: " << GiniImpurity(*leftDistr) << std::endl;
                    }
                    if (item.Depth + 1 >= maxDepth || IsOnlyOneClass(y, rightIndexes)) {
                        rightDistr = std::make_shared<TFeatures>(std::move(ClassDistribution(x, y, classCount, rightIndexes)));
                        std::cout << "Gini right: " << GiniImpurity(*rightDistr) << std::endl;
                    }
                    TTreeNodePtr left(new TTreeNode(leftDistr)), right(new TTreeNode(rightDistr));
                    item.Node->SplitNode(featureIndex, threshold, left, right);
                    if (item.Depth < maxDepth && !IsOnlyOneClass(y, leftIndexes))
                        queue.push_back(TBucket(left, std::move(leftIndexes), item.Depth + 1));
                    if (item.Depth < maxDepth && !IsOnlyOneClass(y, rightIndexes))
                        queue.push_back(TBucket(right, std::move(rightIndexes), item.Depth + 1));
                }
            }
            std::cout << std::endl;
            return TTreeImplPtr(new TTreeImpl(root));
        }

    } // namespace

    template<typename TTreeTrainer>
    TCalculatorPtr DoTrain(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, size_t maxDepth, size_t treeCount, TTreeTrainer treeTrainer) {
        boost::random::mt19937 rng; //(time(nullptr));
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

} // namespace NGCForest

