#pragma once

#include <memory>
#include <vector>
#include <boost/noncopyable.hpp>


namespace NGCForest {

    using TFeatures = std::vector<double>;
    using TConstFeaturesPtr = std::shared_ptr<const TFeatures>;
    using TMiniBatch = std::vector<TFeatures>;

    class TCalculator : private boost::noncopyable {
        public:
            virtual ~TCalculator() {}
            TFeatures Calculate(const TFeatures &features) const;
            TMiniBatch Calculate(const TMiniBatch &minibatch) const;

        protected:
            virtual void DoCalculate(const TFeatures &features, TFeatures &result) const = 0;
    };

} // namespace NGCForest

