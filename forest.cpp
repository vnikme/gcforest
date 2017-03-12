#include "forest.h"
#include "forest_impl.h"

namespace NGCForest {

    // TForestCalculator
    void TForestCalculator::DoCalculate(const TFeatures &features, TFeatures &result) const {
        std::vector<TConstFeaturesPtr> res(Forest.size());
        for (size_t i = 0; i < features.size(); ++i)
            res[i] = Forest[i]->DoCalculate(features);
        Combiner->Combine(res, result);
    }

} // namespace NDecisionTree

