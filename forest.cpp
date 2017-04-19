#include "forest.h"
#include "forest_impl.h"


namespace NGCForest {

    // TForestCalculator
    TForestCalculator::TForestCalculator(TForest &&forest, TCombinerPtr combiner)
        : Forest(std::move(forest))
        , Combiner(combiner)
    {
    }

    namespace {

        void CalculateOneForest(const TFeatures &features, const TForest &forest, TCombinerPtr combiner, TFeatures &result) {
            std::vector<const TFeatures*> res(forest.size());
            for (size_t i = 0; i < forest.size(); ++i)
                res[i] = &forest[i]->Calculate(features);
            combiner->Combine(res, result);
        }

    } // namespace

    void TForestCalculator::DoCalculate(const TFeatures &features, TFeatures &result) const {
        CalculateOneForest(features, Forest, Combiner, result);
    }


    // TCascadeForestCalculator
    TCascadeForestCalculator::TCascadeForestCalculator(TCascadeForest &&forest, TCombinerPtr combiner)
        : CascadeForest(std::move(forest))
        , Combiner(combiner)
    {
    }

    TCalculatorPtr TCascadeForestCalculator::GetSlice(size_t k) const {
        TCascadeForest forest;
        forest.reserve(k);
        for (size_t i = 0; i < k && i < CascadeForest.size(); ++i)
            forest.push_back(CascadeForest[i]);
        return std::make_shared<TCascadeForestCalculator>(std::move(forest), Combiner);
    }

    namespace {

        void CalculateOneLevel(const TFeatures &features, const TForests &level, TCombinerPtr combiner, std::vector<TFeatures> &result) {
            result.resize(level.size());
            for (size_t i = 0; i < level.size(); ++i) {
                CalculateOneForest(features, level[i], combiner, result[i]);
            }
        }

    } // namespace

    std::vector<TFeatures> TCascadeForestCalculator::CalculateForAllLevels(const TFeatures &plainFeatures) const {
        std::vector<TFeatures> result, prevLevel;
        for (size_t i = 0; i < CascadeForest.size(); ++i) {
            TFeatures features(plainFeatures);
            for (const TFeatures &prev : prevLevel) {
                features.insert(features.end(), prev.begin(), prev.end());
            }
            CalculateOneLevel(features, CascadeForest[i], Combiner, prevLevel);
            std::vector<const TFeatures*> res(3 + 0 * prevLevel.size());
            for (size_t j = 0; j < 3 + 0 * prevLevel.size(); ++j)
                res[j] = &prevLevel[j];
            result.emplace_back();
            Combiner->Combine(res, result.back());
        }
        return result;
    }

    void TCascadeForestCalculator::DoCalculate(const TFeatures &plainFeatures, TFeatures &result) const {
        std::vector<TFeatures> prevLevel;
        for (size_t i = 0; i < CascadeForest.size(); ++i) {
            TFeatures features(plainFeatures);
            for (const TFeatures &prev : prevLevel) {
                features.insert(features.end(), prev.begin(), prev.end());
            }
            CalculateOneLevel(features, CascadeForest[i], Combiner, prevLevel);
        }
        std::vector<const TFeatures*> res(prevLevel.size());
        for (size_t i = 0; i < prevLevel.size(); ++i)
            res[i] = &prevLevel[i];
        Combiner->Combine(res, result);
    }

} // namespace NDecisionTree

