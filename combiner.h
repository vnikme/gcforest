#pragma once

#include <memory>
#include <vector>
#include "common.h"

namespace NGCForest {

    class TCombiner {
        public:
            virtual ~TCombiner() {}
            void Combine(const std::vector<TConstFeaturesPtr> &source, TFeatures &result);

        protected:
            virtual void DoCombine(const std::vector<TConstFeaturesPtr> &source, TFeatures &result) = 0;
    };
    using TCombinerPtr = std::shared_ptr<TCombiner>;


    class TMajorityVote : public TCombiner {
        protected:
            virtual void DoCombine(const std::vector<TConstFeaturesPtr> &source, TFeatures &result);
    };

} // namespace NGCForest

