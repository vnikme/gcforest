#include "forest.h"
#include "forest_impl.h"


namespace NGCForest {

    // TForestCalculator
    TForestCalculator::TForestCalculator(TForest &&forest, TCombinerPtr combiner)
        : Forest(std::move(forest))
        , Combiner(combiner)
    {
    }

    void TForestCalculator::DoCalculate(const TFeatures &features, TFeatures &result) const {
        std::vector<TConstFeaturesPtr> res(Forest.size());
        for (size_t i = 0; i < Forest.size(); ++i)
            res[i] = Forest[i]->DoCalculate(features);
        Combiner->Combine(res, result);
    }

} // namespace NDecisionTree

