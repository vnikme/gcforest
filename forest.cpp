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
            std::vector<TConstFeaturesPtr> res(forest.size());
            for (size_t i = 0; i < forest.size(); ++i)
                res[i] = forest[i]->DoCalculate(features);
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

    namespace {

        void CalculateOneLevel(const TFeatures &features, const TForests &level, TCombinerPtr combiner, std::vector<TFeatures> &result) {
            result.resize(level.size());
            for (size_t i = 0; i < level.size(); ++i) {
                CalculateOneForest(features, level[i], combiner, result[i]);
            }
        }

    } // namespace

    void TCascadeForestCalculator::DoCalculate(const TFeatures &plainFeatures, TFeatures &result) const {
        std::vector<TFeatures> prevLevel;
        for (size_t i = 0; i < CascadeForest.size(); ++i) {
            TFeatures features(plainFeatures);
            for (const TFeatures &prev : prevLevel) {
                features.insert(features.end(), prev.begin(), prev.end());
            }
            CalculateOneLevel(features, CascadeForest[i], Combiner, prevLevel);
        }
        std::vector<TConstFeaturesPtr> res(prevLevel.size());
        for (size_t i = 0; i < prevLevel.size(); ++i)
            res[i] = std::make_shared<TFeatures>(std::move(prevLevel[i]));
        Combiner->Combine(res, result);
    }

} // namespace NDecisionTree

