#pragma once

#include <memory>
#include "common.h"

namespace NGCForest {

    class TTreeNode;
    using TTreeNodePtr = std::shared_ptr<TTreeNode>;
    class TTreeNode {
        public:
            TTreeNode(const TConstFeaturesPtr answers);

            size_t GetFeatureIndex() const;
            double GetThreshold() const;
            TTreeNodePtr GetLeftNode() const;
            TTreeNodePtr GetRightNode() const;
            TConstFeaturesPtr GetAnswers() const;

            void SplitNode(size_t featureIndex, double threshold, TTreeNodePtr left, TTreeNodePtr right);

        private:
            size_t FeatureIndex;
            double Threshold;
            TTreeNodePtr Left, Right;
            TConstFeaturesPtr Answers;
    };


    class TTreeImpl {
        public:
            TTreeImpl(TTreeNodePtr root);
            ~TTreeImpl();

            TConstFeaturesPtr DoCalculate(const TFeatures &features);

        private:
            TTreeNodePtr Root;
    };

} // namespace NGCForest

