
#include "forest_impl.h"
#include <vector>


namespace NGCForest {

    namespace NTreePrivate {

        class TNode {
            public:
                size_t GetFeatureIndex() const {
                    return FeatureIndex;
                }

                double GetThreshold() const {
                    return Threshold;
                }

                TNodePtr GetLeftNode() const {
                    return Left;
                }

                TNodePtr GetRightNode() const {
                    return Right;
                }

                TConstFeaturesPtr GetAnswers() const {
                    return Answers;
                }

            private:
                size_t FeatureIndex;
                double Threshold;
                TNodePtr Left, Right;
                TConstFeaturesPtr Answers;
        };

    } // namespace NTreePrivate


    // TTreeImpl
    TTreeImpl::TTreeImpl(NTreePrivate::TNodePtr root)
        : Root(root)
    {
    }

    TTreeImpl::~TTreeImpl() {
    }

    TConstFeaturesPtr TTreeImpl::DoCalculate(const TFeatures &features) {
        NTreePrivate::TNodePtr node = Root;
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

