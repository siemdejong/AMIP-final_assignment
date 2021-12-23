#include "Config.h"
#include <vector>
#include <string>

int Config::set = TRAINING;
double Config::sigma = 0.7;
int Config::preseg_overlap_threshold = 20;
double Config::lower_intensity_threshold = 0.15;
double Config::upper_intensity_threshold = 0.50;
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