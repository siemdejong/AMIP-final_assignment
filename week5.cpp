#include <iostream>
#include <utility>
#include <queue>
#include <regex>

// our own image class
#include "image.hpp"

typedef int num;

// Define training set.
vector<string> training_set {
    "../../data/final/Pat1.nii.gz",
    "../../data/final/Subj3.nii.gz",
    "../../data/final/Subj4.nii.gz",
};

// Define test set.
vector<string> test_set {
    "../../data/final/Pat2.nii.gz",
    "../../data/final/Subj1.nii.gz",
    "../../data/final/Subj2.nii.gz",
};

int main() {
    // Settings.
    bool training = true;
    
    // Gaussian kernel with width sigma.
    double sigma = 1.05;
    auto gk = aip::gausskernel(sigma);
    
    
    vector <string> filenames;
    if (training) {
        filenames = training_set;
    } else {
        filenames = test_set;
    };

	for (string filename : filenames) {

        // Read image
        std::cout << "reading image " << filename << "... ";
        aip::imageNd<num> inputImage(filename);
        std::cout << "Done." << std::endl;

        // Gaussian filtering.
        bool doFilter = false;
        if (doFilter) {
            // Filter in x(0), y(1) and z(2) direction.
            std::cout << "Smoothing image x," << std::flush;
            inputImage.filter( gk, 0 );
            std::cout << " y," << std::flush;
            inputImage.filter( gk, 1 );
            std::cout << " z ... " << std::flush;
            inputImage.filter( gk, 2 );
            std::cout << "done\n";
        }
        
        // Watershed
        std::cout << "Performing watershed... ";
        bool test = inputImage.GetWatershedImage();
        if (!test) {
            std::cout << "Could not perform watershed." << std::endl;
        } else {
            std::cout << "Watershed done." << std::endl;
        }

        // Save watershed image.
        string ws_filename = std::regex_replace(filename, std::regex(".nii"), "_components.nii");
        inputImage.saveNII(ws_filename);
        std::cout << "Watershed saved as NII." << std::endl;
        
        std::cout << std::endl;
    }

    return 0 ;
}













