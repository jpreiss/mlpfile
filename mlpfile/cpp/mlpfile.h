#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Dense>


namespace mlpfile
{
	enum LayerType
	{
		Input = 1,
		Linear = 2,
		ReLU = 3,
	};

	using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

	struct Layer
	{
		// TODO: use std::variant?
		LayerType type;
		int input_size;
		MatrixXfRow W;
		Eigen::VectorXf b;

		std::string describe() const;
	};

	enum Loss
	{
		// The regression loss `1/2 || model.forward(x) - y ||_2^2`.
		SquaredError,
		// The classification loss `cross_entropy(y, softmax(model.forward(x)))`,
		// where softmax(v) = exp(v) / sum(exp(v)) elementwise,
		// and cross_entropy(p, q) = -sum(p log(q)) elementwise.
		SoftmaxCrossEntropy,
	};

	struct Model
	{
		std::vector<Layer> layers;

		// Reads a model from our file format (see block comment at top of file).
		static Model load(char const *path);

		// Generates a random NN with Xavier-uniform initialization. Mainly
		// intended for unit test, etc, where the NN function doesn't matter.
		static Model random(int input, std::vector<int> hidden, int output);

		int input_dim() const;

		int output_dim() const;

		// Computes the forward pass of the neural network.
		Eigen::VectorXf forward(Eigen::VectorXf x);

		// Computes the Jacobian of the neural network.
		//
		// Uses reverse-mode (backprop-style) differentiation, which is faster
		// if the NN's output is smaller than its input. It would be easy to
		// implement forward-mode too if needed, for the opposite case.
		MatrixXfRow jacobian(Eigen::VectorXf const &x);

		// Does a step of gradient descent for regression loss on one point.
		//
		// Updates the parameters in-place for a step of gradient descent on
		// one of the loss functions from the Loss enum.
		void grad_update(Eigen::VectorXf x, Eigen::VectorXf y, Loss loss, float rate);

		// Pretty-prints a description of the network architecture.
		std::string describe() const;
	};
}  // namespace mlpfile
