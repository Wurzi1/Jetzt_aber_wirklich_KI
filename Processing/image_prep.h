#include <opencv2/opencv.hpp>
#include <armadillo>
#include <iostream>
#include <vector>
#include <string>
#include <random>


class image_prep
{
public:
	int amount_images;
	int amount_unused_images;
	std::string base_path;
	std::vector<std::string> image_paths;
	std::vector<std::string> unused_image_paths;
	std::vector<std::string> recent_input_paths;

	image_prep(std::string Base_Path);
	arma::fmat create_input(int amount, int mode);
	arma::fcolvec y_male_female(std::vector<std::string> paths);
};

