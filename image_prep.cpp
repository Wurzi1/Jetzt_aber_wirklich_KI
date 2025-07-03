#include "image_prep.h"

image_prep::image_prep(std::string Base_Path) {
    base_path = Base_Path;

    //get the absolute image paths from the basepath onwards
    std::string pattern = base_path + "*"; // Recursive pattern
    cv::glob(pattern, image_paths, true); // true = recursive search
    if (image_paths.empty()) {
        std::cerr << "No images found in: " << base_path << std::endl;
    }

    unused_image_paths = image_paths;
    amount_images = image_paths.size();
    amount_unused_images = amount_images;
}

arma::fmat image_prep::create_input(int amount, int mode) {
    //load first image just to get dimensions to set matrix size
    cv::Mat test_img = cv::imread(image_paths[0], cv::IMREAD_GRAYSCALE);
    arma::fmat input_node_matrix(test_img.rows * test_img.cols, amount);

    //create vector for y
    current_image_paths.reserve(image_paths[1].size() * amount);
    
    srand(time(NULL));

    for (int i = 0; i < amount; i++) {
        //generate random index to access one unique image path
        int rand_index = rand() % amount_unused_images; // RAND_MAX too low!
        std::string rand_path;
        if (rand_index % 2 == 0) {
            rand_path = unused_image_paths[rand_index];
            unused_image_paths.erase(unused_image_paths.begin() + rand_index);
        }
        else {
            rand_path = unused_image_paths[amount_unused_images-rand_index];
            unused_image_paths.erase(unused_image_paths.begin() + (amount_unused_images - rand_index));
        }
        amount_unused_images--;

        //fill vector for y
        current_image_paths.emplace_back(rand_path);

        //load image with opencv
        cv::Mat cv_img = cv::imread(rand_path, cv::IMREAD_GRAYSCALE );


        //copy opencv image into armadillo column
        arma::Col<uint8_t> img_vec(cv_img.rows * cv_img.cols);
        std::memcpy(img_vec.memptr(), cv_img.data, cv_img.rows * cv_img.cols * sizeof(uint8_t));
        //std::cout << img_vec(arma::span(0,20));

        //covert armadillo column into the right type and add to input matrix
        arma::fcolvec img_fvec = arma::conv_to<arma::fcolvec>::from(img_vec);
        img_fvec /= 255.0f;
        //std::cout << img_fvec(arma::span(0, 20));
        input_node_matrix.col(i) = img_fvec;
    }

    return input_node_matrix;
}

arma::fmat image_prep::create_y(std::vector<std::string> paths) {
    arma::fmat y(1, paths.size());

    for (int i = 0; i < paths.size(); i++) {
        if (paths[i].find("Female") != std::string::npos) {
            y(0, i) = 1.0f;
        }
        else
        {
            y(0, i) = 0.0f;
        }
    }

    return y;
}