#include "common.h"

namespace NGCForest {

    // TCalculator
    TFeatures TCalculator::Calculate(const TFeatures &features) const {
        TFeatures result;
        DoCalculate(features, result);
        return result;
    }

    TMiniBatch TCalculator::Calculate(const TMiniBatch &minibatch) const {
        TMiniBatch result(minibatch.size());
        for (size_t i = 0; i < minibatch.size(); ++i) {
            DoCalculate(minibatch[i], result[i]);
        }
        return result;
    }

} // namespace NGCForest

