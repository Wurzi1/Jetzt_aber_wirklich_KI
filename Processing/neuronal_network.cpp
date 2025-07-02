#include "../neuronal_network.h"

neuronal_network::neuronal_network(std::vector<int> Network_Dimensions) {
	network_dimensions = Network_Dimensions;
	network_layers = network_dimensions.size();

	for (int i = 1; i < network_layers; i++) {

		weights.emplace_back(arma::fmat(network_dimensions[i], network_dimensions[i-1], arma::fill::randu)); // or randn for normal distribution

		biases.emplace_back(arma::fcolvec(network_dimensions[i], arma::fill::randu)); // or randn for normal distribution

		final_nodes.emplace_back(arma::fcolvec(network_dimensions[i - 1], arma::fill::zeros));
	}
	final_nodes.emplace_back(arma::fcolvec(network_dimensions[network_layers-1], arma::fill::zeros));
}

void neuronal_network::sigmoid(arma::fmat &matrix) {
	arma::subview_col col = matrix.col(0);
	col = 1 / (1 + arma::exp(-1 * col));
}

void neuronal_network::feed_forward(arma::fmat input_nodes) {
	//get amount of input samples
	int input_amount = input_nodes.n_cols;

	//create feed forward nodes as matrix with column number of input sample amount
	//to calculate "all at once"
	nodes.emplace_back(input_nodes);
	for (int i = 1; i < network_layers; i++) {
		nodes.emplace_back(arma::fmat(network_dimensions[i], input_amount, arma::fill::zeros));
	}

	//matrix feed forward
	for (int i = 0; i < network_layers - 1; i++) {
		nodes[i + 1] = weights[i] * nodes[i];
		nodes[i + 1].each_col() += biases[i];
		sigmoid(nodes[i+1]);

		std::cout << nodes[network_layers-1] << "\n";
	}
}