
#include "forest_impl.h"
#include <vector>


namespace NGCForest {

    // TTreeNode
    TTreeNode::TTreeNode(const TConstFeaturesPtr answers)
        : FeatureIndex(0)
        , Threshold(0.0)
        , Left(TTreeNodePtr())
        , Right(TTreeNodePtr())
        , Answers(answers)
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

    TConstFeaturesPtr TTreeNode::GetAnswers() const {
        return Answers;
    }

    void TTreeNode::SplitNode(size_t featureIndex, double threshold, TTreeNodePtr left, TTreeNodePtr right) {
        FeatureIndex = featureIndex;
        Threshold = threshold;
        Left = left;
        Right = right;
        TConstFeaturesPtr().swap(Answers);
    }


    // TTreeImpl
    TTreeImpl::TTreeImpl(TTreeNodePtr root)
        : Root(root)
    {
    }

    TTreeImpl::~TTreeImpl() {
    }

    TConstFeaturesPtr TTreeImpl::DoCalculate(const TFeatures &features) {
        TTreeNodePtr node = Root;
        while (!!node->GetLeftNode()) {
            double featureValue = features[node->GetFeatureIndex()];
            if (featureValue < node->GetThreshold())
                node = node->GetLeftNode();
            else
                node = node->GetRightNode();
        }
        return node->GetAnswers();
    }

} // namespace NGCForest

