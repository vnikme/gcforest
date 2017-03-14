#include "evaluation.h"

#include <algorithm>


namespace NGCForest {

    double AUC(std::vector<std::pair<int, double>> data) {
        std::sort(data.begin(), data.end(), [] (std::pair<int, double> a, std::pair<int, double> b) { return a.second < b.second; } );
        double neg = 1e-38, pos = 1e-38;
        size_t tp = 0, fp = 0;
        for (auto v : data) {
            if (v.first == 0) {
                neg += 1;
                ++fp;
            } else {
                pos += 1;
                ++tp;
            }
        }
        double last = data[0].second;
        std::vector<double> tpr(1, 1.0), fpr(1, 1.0);
        for (auto v : data) {
            if (v.second > last + 1e-10) {
                tpr.push_back(tp / pos);
                fpr.push_back(fp / neg);
                last = v.second;
            }
            if (v.first == 0)
                --fp;
            else
                --tp;
        }
        tpr.push_back(tp / pos);
        fpr.push_back(fp / neg);
        std::reverse(tpr.begin(), tpr.end());
        std::reverse(fpr.begin(), fpr.end());
        double auc = 0.0;
        for (size_t i = 1; i < tpr.size(); ++i)
            auc += ((fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]));
        return auc / 2.0;
    }

} // namespace NGCForest

