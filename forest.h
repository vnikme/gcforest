#pragma once

#include <memory>
#include <vector>
#include "combiner.h"
#include "common.h"

namespace NGCForest {

    class TTreeImpl;
    using TTreeImplPtr = std::shared_ptr<TTreeImpl>;
    using TForest = std::vector<TTreeImplPtr>;

    class TForestCalculator : public TCalculator {
        public:
            TForestCalculator(TForest &&forest, TCombinerPtr combiner);

        protected:
            virtual void DoCalculate(const TFeatures &features, TFeatures &result) const;

        private:
            TForest Forest;
            TCombinerPtr Combiner;
    };

} // namespace NDecisionTree

