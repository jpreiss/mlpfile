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

	struct Model
	{
		std::vector<Layer> layers;

		// Reads a model from our file format (see block comment at top of file).
		static Model load(char const *path);

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

		// Pretty-prints a description of the network architecture.
		std::string describe() const;
	};
}  // namespace mlpfile
