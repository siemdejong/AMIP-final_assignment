#include "Config.h"
#include <vector>
#include <string>
#include <iostream>

int Config::set = TRAINING;
double Config::sigma = 0.7; // (0, inf)
int Config::preseg_overlap_threshold = 20; // (0, inf)
double Config::lower_intensity_threshold = 0.15; // [0, 1]
double Config::upper_intensity_threshold = 0.50; // [0, 1] and lower_intensity_threshold + upper_intensity_threshold in [0, 1].
bool Config::train_using_multiple_parameters = false;

// Define training set.
std::vector<std::string> Config::training_set {
    "../../data/final/Pat1.nii.gz",
    "../../data/final/Subj3.nii.gz",
    "../../data/final/Subj4.nii.gz",
    "../../data/final/Subj2.nii.gz",
};

// Define test set.
std::vector<std::string> Config::test_set {
    "../../data/final/Pat2.nii.gz",
    "../../data/final/Subj1.nii.gz",
};

// Define anisotropic diffusion (AD) set.
std::vector<std::string> Config::ad_set {
    "../../data/final/Pat2_AD.nii.gz",
    "../../data/final/Subj1_AD.nii.gz",
};

void verifyConfig() {
    if (Config::sigma <= 0) {
        std::cout << "Sigma should be in (0, inf).\n";
        throw ValueError();
    }
    if (Config::preseg_overlap_threshold <= 0) {
        std::cout << "Basin-presegmentation overlap threshold should be in (0, inf).\n";
        throw ValueError();
    }
    if (Config::lower_intensity_threshold < 0) {
        std::cout << "Lower mean basin intensity threshold should be in [0, 1].\n";
        throw ValueError();
    }
    if (Config::upper_intensity_threshold < 0) {
        std::cout << "Upper mean basin intensity threshold should be in [0, 1].\n";
        throw ValueError();
    }
    if (Config::lower_intensity_threshold + Config::upper_intensity_threshold >= 1) {
        std::cout << "The sum of the lower and upper mean basin intensity should be in [0, 1].\n";
        throw ValueError();
    }
}
