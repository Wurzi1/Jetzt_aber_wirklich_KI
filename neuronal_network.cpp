#include "neuronal_network.h"

neuronal_network::neuronal_network(std::vector<int> Network_Dimensions) {
	arma::arma_rng::set_seed(time(NULL));
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

		//std::cout << nodes[network_layers-1] << "\n";
	}
}

float neuronal_network::cost(arma::fmat y, arma::fmat &y_hat) {
	//y: n^L x m; y_hat: n^L x m
	std::cout << y << "\n";
	std::cout << y_hat << "\n";
	//loss function applied on every matrix element
	arma::fmat losses0 = - (y % arma::log(y_hat)); 
	arma::fmat losses1 = - ((1 - y) % arma::log(1 - y_hat)); 
	arma::fmat losses = - ((y % arma::log(y_hat)) + (1 - y) % arma::log(1 - y_hat)); 
	arma::fmat losses3 = - losses0 + losses1; 
	std::cout << losses0 << "\n";
	std::cout << losses1 << "\n";
	std::cout << losses << "\n";
	std::cout << losses3 << "\n";

	//number of results
	float m = y_hat.n_rows * y_hat.n_cols;

	//average_cost: n^L x 1
	arma::fcolvec average_cost = (1 / m) * arma::sum(losses, 1); //sum across rows

	//total cost
	float cost = arma::sum(average_cost);
	return cost;
}

void neuronal_network::backprop(arma::fmat y) {

	float m = nodes[network_layers - 1].n_cols;

	std::vector<arma::fmat> dC_dZs;
	std::vector<arma::fmat> dC_dWs;
	std::vector<arma::fmat> dC_dBs;
	std::vector<arma::fmat> dC_dAs;


	for (int i = 0; i < 1; i++) {
		arma::fmat dC_dZ = (1 / m) * (nodes[network_layers - 1 - i] - y);
		dC_dZs.emplace_back(dC_dZ);

		arma::fmat dC_dW = dC_dZ * nodes[network_layers - 2 - i].t();
		dC_dWs.emplace_back(dC_dW);

		arma::fmat dC_dB = arma::sum(dC_dZ, 1);
		dC_dBs.emplace_back(dC_dB);

		arma::fmat dC_dA = weights[network_layers - 2 - i].t() * dC_dZ;
		dC_dAs.emplace_back(dC_dA);
	}



	for (int i = 1; i < network_layers-1; i++) {
		arma::fmat dC_dZ = dC_dAs.back() % (nodes[network_layers - 1 - i] % (1 - nodes[network_layers - 1 - i]));
		dC_dZs.emplace_back(dC_dZ);

		arma::fmat dC_dW = dC_dZ * nodes[network_layers - 2 - i].t();
		dC_dWs.emplace_back(dC_dW);

		arma::fmat dC_dB = arma::sum(dC_dW, 1);
		dC_dBs.emplace_back(dC_dB);

		arma::fmat dC_dA = weights[network_layers - 2 - i].t() * dC_dZ;
		dC_dAs.emplace_back(dC_dA);
	}




	for (int i = 0; i < network_layers - 1; i++) {
		weights[network_layers - 1 - i] -= (0.1f * dC_dWs[i]);
		biases[network_layers - 1 - i] -= (0.1f * dC_dBs[i]);
	}
}