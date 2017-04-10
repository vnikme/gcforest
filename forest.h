#pragma once

#include <memory>
#include <vector>
#include "combiner.h"
#include "common.h"

namespace NGCForest {

    class TTreeImpl;
    using TTreeImplPtr = std::shared_ptr<TTreeImpl>;
    using TForest = std::vector<TTreeImplPtr>;
    using TForests = std::vector<TForest>;
    using TCascadeForest = std::vector<TForests>;

    class TForestCalculator : public TCalculator {
        public:
            TForestCalculator(TForest &&forest, TCombinerPtr combiner);

        protected:
            virtual void DoCalculate(const TFeatures &features, TFeatures &result) const;

        private:
            TForest Forest;
            TCombinerPtr Combiner;
    };


    class TCascadeForestCalculator : public TCalculator {
        public:
            TCascadeForestCalculator(TCascadeForest &&forest, TCombinerPtr combiner);
            TCalculatorPtr GetSlice(size_t k) const;

        protected:
            virtual void DoCalculate(const TFeatures &features, TFeatures &result) const;

        private:
            TCascadeForest CascadeForest;
            TCombinerPtr Combiner;
    };

} // namespace NDecisionTree

