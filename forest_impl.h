#pragma once

#include <memory>
#include "common.h"

namespace NGCForest {

    class TTreeNode;
    using TTreeNodePtr = std::shared_ptr<TTreeNode>;
    class TTreeNode {
        public:
            TTreeNode();

            size_t GetFeatureIndex() const;
            double GetThreshold() const;
            TTreeNodePtr GetLeftNode() const;
            TTreeNodePtr GetRightNode() const;
            const TFeatures &GetAnswers() const;

            void SplitNode(size_t featureIndex, double threshold, TTreeNodePtr left, TTreeNodePtr right);
            void SetAnswers(TFeatures &&answers);

        private:
            size_t FeatureIndex;
            double Threshold;
            TTreeNodePtr Left, Right;
            TFeatures Answers;
    };


    class TTreeImpl {
        public:
            TTreeImpl();
            virtual ~TTreeImpl();

            const TFeatures &Calculate(const TFeatures &features) const;

        protected:
            virtual const TFeatures &DoCalculate(const TFeatures &features) const = 0;
    };


    class TDynamicTreeImpl : public TTreeImpl {
        public:
            TDynamicTreeImpl(TTreeNodePtr root);

        protected:
            virtual const TFeatures &DoCalculate(const TFeatures &features) const;

        private:
            TTreeNodePtr Root;
    };


    class TObliviousTreeImpl : public TTreeImpl {
        public:
            TObliviousTreeImpl(const std::vector<size_t> &featureIndexes, const std::vector<double> &thresholds, const std::vector<TFeatures> &answers);

        protected:
            virtual const TFeatures &DoCalculate(const TFeatures &features) const;

        private:
            std::vector<size_t> FeatureIndexes;
            std::vector<double> Thresholds;
            std::vector<TFeatures> Answers;
    };

} // namespace NGCForest

