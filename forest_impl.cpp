
#include "forest_impl.h"
#include <vector>


namespace NGCForest {

    // TTreeNode
    TTreeNode::TTreeNode()
        : FeatureIndex(0)
        , Threshold(0.0)
        , Left(TTreeNodePtr())
        , Right(TTreeNodePtr())
        , Answers()
    {
    }

    size_t TTreeNode::GetFeatureIndex() const {
        return FeatureIndex;
    }

    double TTreeNode::GetThreshold() const {
        return Threshold;
    }

    TTreeNodePtr TTreeNode::GetLeftNode() const {
        return Left;
    }

    TTreeNodePtr TTreeNode::GetRightNode() const {
        return Right;
    }

    const TFeatures &TTreeNode::GetAnswers() const {
        return Answers;
    }

    void TTreeNode::SplitNode(size_t featureIndex, double threshold, TTreeNodePtr left, TTreeNodePtr right) {
        FeatureIndex = featureIndex;
        Threshold = threshold;
        Left = left;
        Right = right;
        Answers.clear();
    }

    void TTreeNode::SetAnswers(TFeatures &&answers) {
        Answers = std::move(answers);
    }


    // TTreeImpl
    TTreeImpl::TTreeImpl() {
    }

    TTreeImpl::~TTreeImpl() {
    }

    const TFeatures &TTreeImpl::Calculate(const TFeatures &features) const {
        return DoCalculate(features);
    }


    // TDynamicTreeImpl
    TDynamicTreeImpl::TDynamicTreeImpl(TTreeNodePtr root)
        : Root(root)
    {
    }

    const TFeatures &TDynamicTreeImpl::DoCalculate(const TFeatures &features) const {
        TTreeNodePtr node = Root;
        while (!!node->GetLeftNode()) {
            size_t idx = node->GetFeatureIndex();
            double featureValue = idx < features.size() ? features[idx] : 0.0;
            if (featureValue < node->GetThreshold())
                node = node->GetLeftNode();
            else
                node = node->GetRightNode();
        }
        return node->GetAnswers();
    }


    // TObliviousTreeImpl
    TObliviousTreeImpl::TObliviousTreeImpl(const std::vector<size_t> &featureIndexes, const std::vector<double> &thresholds, const std::vector<TFeatures> &answers)
        : FeatureIndexes(featureIndexes)
        , Thresholds(thresholds)
        , Answers(answers)
    {
    }

    const TFeatures &TObliviousTreeImpl::DoCalculate(const TFeatures &features) const {
        size_t mask = 0;
        for (size_t i = 0; i < FeatureIndexes.size(); ++i) {
            mask <<= 1;
            double val = features[FeatureIndexes[i]];
            if (val >= Thresholds[i])
                mask |= 1;
        }
        return Answers[mask];
    }


} // namespace NGCForest

