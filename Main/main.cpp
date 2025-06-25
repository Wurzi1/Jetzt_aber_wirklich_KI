//
// Created by danie on 25.06.2025.
//
#include <cmath>
#include <iostream>
int input, hidden, output;
double learningreate;

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

    int  main() {

        input = 3;
        hidden = 3;
        output = 1;
        double inputs [input] = {1, 2 ,4};
        double bias [input] ={0, 0, 0};
        double bias_OUT [output] ={1};
        double weightsOUTPUT[hidden] [output] = {1, 4, 6};
        double finalOutput[output] = {};
        double outputsHIDDEN[hidden] = {};
        double weights [input] [hidden] = {
            {2, 5, 1},
            {1, 3, 8},
            {1 ,1 ,1}
            };



        //Hidden Layer
        for (int i = 0; i < hidden; i++) {
             outputsHIDDEN[i] = 0;
            for (int j = 0; j < input; j++) {
                outputsHIDDEN[i] += inputs[j]*weights[j][i];
            }
                outputsHIDDEN[i] += bias[i];

            std::cout << "RAW: " << outputsHIDDEN[i] << " | ";
            outputsHIDDEN[i] = sigmoid(outputsHIDDEN[i]);
            std::cout << "SIGM: " << outputsHIDDEN[i] << "\n";



        }


        //Output Layer
            for (int i = 0; i < output; i++) {
            finalOutput[i] = 0;
            for (int j = 0; j < hidden; j++) {
            finalOutput[i] += outputsHIDDEN[j]*weightsOUTPUT[j][i];
        }
            finalOutput[i] += bias_OUT[i];
                std::cout << "-----OUTPUT-----" "\n";

            std::cout << "RAW: " << finalOutput[i] << " | ";
            finalOutput[i] = sigmoid(finalOutput[i]);
            std::cout << "SIGM: " << finalOutput[i] << "\n";



            }




    return 0;
}
