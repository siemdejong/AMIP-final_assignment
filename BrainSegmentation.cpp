/* BrainSegmentation.cpp
 * 
 * Written by Sylvia Spies, Ilja Deurloo, Abel Bregman and Siem de Jong.
 * 
 * Segment the brain from a brain image using the watershed transform.
 */

#include <iostream>
#include <utility>
#include <queue>
#include <regex>
#include <string>


// our own image class
#include "image.hpp"

// Our own configuration class.
#include "Config.h"

typedef int num;

int main() {
    // Configuration options in Config.cpp.
    try {
        verifyConfig();
    } catch(ValueError) {
        std::cout << "ValueError: Please check the configuration files.";
        exit(0);
    }
    
    if (Config::train_using_multiple_parameters) {
        std::cout << "Config::train_using_multiple_parameters is set to true.\n"
                  << "Please verify for what parameters images should be generated.\n";
        system("pause");
    }
    
    int set_to_use = Config::set; // Defines training set to use.
    
    // AD images must not be gaussian filtered.
    bool doFilter = true;
    std::vector<string> filenames;
    switch(set_to_use) {
        case TEST:
            doFilter = true;
            filenames = Config::test_set;
            break;
        case AD:
            doFilter = false;
            filenames = Config::ad_set;
            break;
        default: // In case TRAINING or erroneous setting.
             doFilter = true;
            filenames = Config::training_set;
            break;
    }

    for (string filename : filenames) {

        // Read image
        std::cout << "Reading image " << filename << "... ";
        aip::imageNd<num> input_image(filename);
        aip::imageNd<num> original_image(input_image); // To mask algorithm output on the original image later.
        std::cout << "Done." << std::endl;

        // Gaussian filtering.
        double sigma = Config::sigma; // Choose gaussian kernel width.
        if(doFilter) {
            // Subsequently filter in x = 0, y = 1 and z = 2 direction with separable filter.
            std::cout << "Smoothing image x," << std::flush;
            auto gk = aip::gausskernel(sigma); // Gaussian kernel with width sigma.
            input_image.filter(gk, 0);
            std::cout << " y," << std::flush;
            input_image.filter(gk, 1);
            std::cout << " z ... " << std::flush;
            input_image.filter(gk, 2);
            std::cout << "done\n";
        }

        // Watershed transform
        std::cout << "Performing watershed... ";
        bool test = input_image.GetWatershedImage();
        if(not test) {
            std::cout << "Could not perform watershed." << std::endl;
        } else {
            std::cout << "Watershed done." << std::endl;
        }

        // Save watershed image.
        string ws_filename = std::regex_replace(filename, std::regex(".nii"), "_components.nii");
        input_image.saveNII(ws_filename);
        std::cout << "Watershed saved as NII." << std::endl;

        // Merge watershed pixels with neighbouring basins.
        std::cout << "Performing pixel-basin merging... ";
        bool test_pbm = input_image.pixelBasinMerge();
        if(!test_pbm) {
            std::cout << "Could not perform pixel-basin merging." << std::endl;
        } else {
            std::cout << "Pixel-basin merging completed." << std::endl;
        }

        // Save pixel-basin merged image.
        string pbm_filename = std::regex_replace(filename, std::regex(".nii"), "_PixelBasinMerge.nii");
        input_image.saveNII(pbm_filename);
        // Uncomment to save single slice for watershed transform inspection purposes.
//        input_image.getSlice(0, 90,
//                            std::regex_replace(pbm, std::regex(".nii"), "_")
//                            + to_string(sigma) + ".bmp");
        std::cout << "Pixel-basin merged images saved as NII." << std::endl;

        // Perform basin-basin merge.
        std::cout << "Performing basin-basin merging... ";
        string pre_segmentation_filename;
        switch (set_to_use) { // AD images have suffix "_AD.nii" instead of ".nii" so need to be treated differently.
            case AD:
                pre_segmentation_filename = std::regex_replace(filename, std::regex("_AD.nii"), "Brain.nii");
                break;
            default:
                pre_segmentation_filename = std::regex_replace(filename, std::regex(".nii"), "Brain.nii");
                break;
        }
        aip::imageNd<num> pre_segmentation(pre_segmentation_filename);

        // Option to automatically vary parameters to output many files.
        bool train_using_multiple_parameters = Config::train_using_multiple_parameters;
        if(train_using_multiple_parameters) {
            std::cout << "Careful. I'll generate a lot of images now.\n";
            // a: basin-presegmentation overlap threshold.
            // b and c: lower and upper mean basin intensity thresholds.
            for(int a = 10; a <= 100; a += 10) {
                for (double b = 0; b <= 1; b += 0.05) {
                    for(double c = 0; c < 1 - b; c += 0.05) {
                        // We need a new temporary image for every set of parameters.
                        aip::imageNd<num> temp_image(input_image);
                        cout << "Using (a, b, c) = (" << a << ", " << b << ", " << c << ")\n";
                        
                        // Basin selection.
                        temp_image.basinBasinMerge(pre_segmentation, original_image, a, b, c);
                        temp_image *= original_image; // Cut out selected basins.

                        temp_image.saveBrainSegmentation(filename, a, b, c);
                    }
                }
                
            }
        } else {
            int a = Config::preseg_overlap_threshold; // Basin-presegmentation overlap threshold.
            double b = Config::lower_intensity_threshold; // Lower mean basin intensity thresholds.
            double c = Config::upper_intensity_threshold; // Upper mean basin intensity thresholds.
    
            cout << "Using (a, b, c) = (" << a << ", " << b << ", " << c << ")\n";
            input_image.basinBasinMerge(pre_segmentation, original_image, a, b, c);
            input_image *= original_image; // Cut out selected basins.
            
            input_image.saveBrainSegmentation(filename, a, b, c);
        }

        std::cout << "\n";
    }

    return 0;
}
