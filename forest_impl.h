#pragma once

#include <memory>
#include "common.h"

namespace NGCForest {

    namespace NTreePrivate {

        class TNode;
        using TNodePtr = std::shared_ptr<TNode>;

    } // namespace NTreePrivate


    class TTreeImpl {
        public:
            TTreeImpl(NTreePrivate::TNodePtr root);
            ~TTreeImpl();

            TConstFeaturesPtr DoCalculate(const TFeatures &features);

        private:
            NTreePrivate::TNodePtr Root;
    };

} // namespace NGCForest

