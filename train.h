#pragma once

#include "common.h"


namespace NGCForest {

    TCalculatorPtr TrainRandomForest(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, size_t maxDepth, size_t treeCount);
    TCalculatorPtr TrainFullRandomForest(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, size_t maxDepth, size_t treeCount);

} // namespace NGCForest

