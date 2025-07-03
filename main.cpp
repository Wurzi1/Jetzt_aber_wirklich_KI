#include <iostream>
#include <opencv2/opencv.hpp>
#include <armadillo>
#include <vector>
#include <string>
#include "neuronal_network.h"
#include "image_prep.h"


int main()
{
    std::string basePath = "C:/Users/Jakob/Downloads/archive/Dataset/Train/";

    std::vector<int> dim = {38804, 10000, 250, 1};
    int epochs = 500;
    float learning_rate = 0.1;
    int m = 25;


    image_prep prp(basePath);
    neuronal_network ntw(dim);
    arma::fmat input = prp.create_input(m, 0);
    arma::fmat y = prp.create_y(prp.current_image_paths);
    ntw.prep_feed_forward(input);

    for (int i = 0; i < epochs; i++) {

        ntw.feed_forward();

        float cost = ntw.cost(y, ntw.nodes.back());
        std::cout << "Cost" << i << ": " << cost << "\n";

        ntw.backprop(y);

    }

    return 0;
    
}