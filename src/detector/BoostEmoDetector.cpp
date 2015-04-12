/**
 *
 * @file    BoostEmoDetector.cpp
 * @brief
 *
 * @details
 *
 */

#include "BoostEmoDetector.h"
#include "AdaBoostClassifier.h"

namespace emogen {

  BoostEmoDetector::BoostEmoDetector(int boost_type, double trim_weight, int max_depth) : EmoDetector() {
    this->boost_type = boost_type;
    this->trim_weight = trim_weight;
    this->max_depth = max_depth;
  }

  BoostEmoDetector::BoostEmoDetector(int boost_type, double trim_weight, int max_depth,
      std::map<std::string, std::pair<vector<Emotion>, Classifier*> >
      detmap_ext) : EmoDetector(detmap_ext) {
    this->boost_type = boost_type;
    this->trim_weight = trim_weight;
    this->max_depth = max_depth;
  }


  Classifier* BoostEmoDetector::createClassifier() {
    return new AdaBoostClassifier(this->boost_type, this->trim_weight, this->max_depth);
  }

  std::pair<Emotion, float> BoostEmoDetector::predict(cv::Mat& frame) {
    return EmoDetector::predictVotingOneVsAllExt(frame);
  }
}
