#include "common.h"

namespace NGCForest {

    TMiniBatch Transpose(const TMiniBatch &features) {
        size_t rows = features.size(), cols = features.front().size();
        TMiniBatch result(cols, TFeatures(rows));
        for (size_t i = 0; i < cols; ++i)
            for (size_t j = 0; j < rows; ++j)
                result[i][j] = features[j][i];
        return result;
    }

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

    void TCalculator::Save(std::ostream &fout) const {
        DoSave(fout);
    }

} // namespace NGCForest

