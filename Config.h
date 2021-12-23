#pragma once

#include <vector>
#include <string>

enum setSetChoice {
    TRAINING = 1,
    TEST = 2,
    AD = 3,
};

class Config {
    public:
        static int set;
        static double sigma;
        static int preseg_overlap_threshold;
        static double lower_intensity_threshold;
        static double upper_intensity_threshold;
        static bool train_using_multiple_parameters;
        static std::vector<std::string> training_set;
        static std::vector<std::string> test_set;
        static std::vector<std::string> ad_set;
};