#include <iostream>
#include <cmath>
#include <math.h>
#include <vector>
#include <armadillo>
#include <time.h>


class neuronal_network
{
public:
	std::vector<int> network_dimensions;
	int network_layers;
	std::vector<arma::fmat> nodes;
	std::vector<arma::fcolvec> final_nodes;
	std::vector<arma::fmat> weights;
	std::vector<arma::fcolvec> biases;

	neuronal_network(std::vector<int> Network_Dimensions);

	void sigmoid(arma::fmat &matrix);
	void feed_forward(arma::fmat input_nodes);
	float cost(arma::fmat y, arma::fmat &y_hat);
	void backprop(arma::fmat y);
};

