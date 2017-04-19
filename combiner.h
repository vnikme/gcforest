#pragma once

#include <memory>
#include <vector>
#include "common.h"

namespace NGCForest {

    class TCombiner {
        public:
            virtual ~TCombiner() {}
            void Combine(const std::vector<const TFeatures*> &source, TFeatures &result);

        protected:
            virtual void DoCombine(const std::vector<const TFeatures*> &source, TFeatures &result) = 0;
    };
    using TCombinerPtr = std::shared_ptr<TCombiner>;


    class TMajorityVoteCombiner : public TCombiner {
        protected:
            virtual void DoCombine(const std::vector<const TFeatures*> &source, TFeatures &result);
    };


    class TAverageCombiner : public TCombiner {
        protected:
            virtual void DoCombine(const std::vector<const TFeatures*> &source, TFeatures &result);
    };

} // namespace NGCForest

