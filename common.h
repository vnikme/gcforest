#pragma once

#include <memory>
#include <vector>


namespace NGCForest {

    using TFeatures = std::vector<double>;
    using TConstFeaturesPtr = std::shared_ptr<const TFeatures>;
    using TMiniBatch = std::vector<TFeatures>;

    class TCalculator {
        private:
            TCalculator(const TCalculator &);
            TCalculator &operator = (const TCalculator &);
            TCalculator(TCalculator &&);
            TCalculator &operator = (TCalculator &&);

        public:
            TCalculator() {}
            virtual ~TCalculator() {}
            TFeatures Calculate(const TFeatures &features) const;
            TMiniBatch Calculate(const TMiniBatch &minibatch) const;

        protected:
            virtual void DoCalculate(const TFeatures &features, TFeatures &result) const = 0;
    };
    using TCalculatorPtr = std::shared_ptr<TCalculator>;

} // namespace NGCForest

