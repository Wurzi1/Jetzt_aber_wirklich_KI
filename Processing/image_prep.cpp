#include "../image_prep.h"

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

    input_paths.reserve(image_paths[1].size() * amount);
    
    for (int i = 0; i < amount; i++) {
        //generate random index to access one unique image path
        int rand_index = std::rand() % amount_unused_images;

        //flag index as used
        std::string rand_path = unused_image_paths[rand_index];
        unused_image_paths.erase(unused_image_paths.begin() + rand_index);

        //save path for labeling
        recent_input_paths.emplace_back(rand_path);
        amount_unused_images--;

        //load image with opencv
        cv::Mat cv_img = cv::imread(rand_path, cv::IMREAD_GRAYSCALE );

        //copy opencv image into armadillo column
        arma::Col<uint8_t> img_vec(cv_img.rows * cv_img.cols);
        std::memcpy(img_vec.memptr(), cv_img.data, cv_img.rows * cv_img.cols * sizeof(uint8_t));

        //convert armadillo column into the right type and add to input matrix
        arma::fcolvec img_fvec = arma::conv_to<arma::fcolvec>::from(img_vec);
        input_node_matrix.col(i) = img_fvec;
    }

    return input_node_matrix;
}


//create label values [1=female ; 0=male]
arma::fcolvec image_prep::create_label(std::vector<std::string> paths) {
        arma::fcolvec y(paths.size());
        std::string female = "Female";



    for (int i = 0; i < paths.size(); i++) {
        if (paths[i].find(female) != std::string::npos) {
            y[i] = 1;
        } else {
            y[i] = 0;
        }
    }


    return y;

}