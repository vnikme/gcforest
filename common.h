#pragma once

#include <memory>
#include <vector>
#include <iosfwd>


namespace NGCForest {

    using TFeatures = std::vector<double>;
    using TConstFeaturesPtr = std::shared_ptr<const TFeatures>;
    using TMiniBatch = std::vector<TFeatures>;

    TMiniBatch Transpose(const TMiniBatch &features);

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
            void Save(std::ostream &fout) const;

        protected:
            virtual void DoCalculate(const TFeatures &features, TFeatures &result) const = 0;
            virtual void DoSave(std::ostream &fout) const = 0;
    };
    using TCalculatorPtr = std::shared_ptr<TCalculator>;

} // namespace NGCForest

