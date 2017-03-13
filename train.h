#pragma once

#include "common.h"


namespace NGCForest {

    TCalculatorPtr Train(const std::vector<TFeatures> &x, const std::vector<size_t> &y, size_t classCount, size_t treeCount);

} // namespace NGCForest

