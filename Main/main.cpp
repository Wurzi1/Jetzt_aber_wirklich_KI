#include <iostream>
#include <opencv2/opencv.hpp>
#include <armadillo>
#include <vector>
#include <string>
#include "../Processing/neuronal_network.h"
#include "../Processing/image_prep.h"


int main()
{
    /*std::vector<int> dim = {3,4,2};
    neuronal_network n(dim);
    std::string basePath = "C:/Users/Jakob/Downloads/archive/Dataset/Train/";
    std::string pattern = basePath + "*"; // Recursive pattern

    std::vector<cv::String> imagePaths;
    cv::glob(pattern, imagePaths, true); // true = recursive search

    if (imagePaths.empty()) {
        std::cerr << "No images found in: " << basePath << std::endl;
        return -1;
    }

    std::vector<cv::Mat> images;
    for (int i = 0; i < 5; i++) {
        cv::Mat img = cv::imread(imagePaths[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Warning: Could not load image: " << imagePaths[i] << std::endl;
            continue;
       }
        images.push_back(img);
        std::cout << "Loaded: " << imagePaths[i] << std::endl;
    }
    int siz = imagePaths[1].size();
    int soz = std::size(imagePaths[1]);

    std::cout << "Siz: " << siz << "\n";
    std::cout << "Soz: " << soz << "\n";

    arma::Col<uint8_t> img_vec(images[1].rows * images[1].cols);
    std::memcpy(img_vec.memptr(), images[1].data, images[1].rows * images[1].cols * sizeof(uint8_t));
    arma::fcolvec img_fvec = arma::conv_to<arma::fcolvec>::from(img_vec);
    std::cout << "\n" << images[1].type();

    cv::imshow("2", images[1]);
    cv::waitKey(0); // Wait for a key press */

    std::string basePath = "C:/Users/Jakob/Downloads/archive/Dataset/Train/";
    image_prep prp(basePath);

    std::vector<int> dim = {38804, 50, 1};
    neuronal_network ntw(dim);
    ntw.feed_forward(prp.create_input(5, 0));

    std::vector<std::string> paths = prp.recent_input_paths;
    arma::fcolvec labels = prp.create_label(paths);

    return 0;
    
}