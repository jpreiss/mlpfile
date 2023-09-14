#include <cstdio>
#include <memory>

#include "mlpfile.h"


static std::vector<Eigen::VectorXf> fwdpass_stack(
	mlpfile::Model &m, Eigen::VectorXf x)
{
	std::vector<Eigen::VectorXf> stack = { x };
	if (x.rows() != m.input_dim()) {
		throw std::runtime_error("incorrect input size");
	}

	// Forward pass. TODO: Deduplicate copy-paste.
	for (mlpfile::Layer const &layer : m.layers) {
		if (layer.type == mlpfile::Linear) {
			stack.push_back(layer.W * stack.back() + layer.b);
		}
		else if (layer.type == mlpfile::ReLU) {
			stack.push_back(stack.back().array().max(0));
		}
		else {
			throw std::runtime_error("unrecognized type");
		}
	}

	return stack;
}


namespace mlpfile
{
	static std::string const layer_type_names[] = {
		"Reserved",
		"Reserved",
		"Linear",
		"ReLU",
	};

	std::string Layer::describe() const
	{
		std::string s = layer_type_names[type];
		switch (type) {
		case mlpfile::Linear:
			s += ": " + std::to_string(W.cols())
				+ " -> " + std::to_string(W.rows());
			break;
		default:
			break;
		}
		return s;
	}

	Model Model::load(char const *path)
	{
		Model model;

		FILE *fp = fopen(path, "r");
		if (!fp) {
			throw std::runtime_error("Could not open file.");
		}

		auto readu32 = [fp]()
		{
			uint32_t u32;
			size_t r = fread(&u32, sizeof(u32), 1, fp);
			if (r != 1) {
				throw std::runtime_error("File format error.");
			}
			return u32;
		};

		uint32_t n_layers = readu32();
		if (n_layers == 0) {
			throw std::runtime_error("Model has no layers.");
		}
		model.layers.resize(n_layers);

		uint32_t size = readu32();
		model._input_dim = size;

		// Pass 1: Metadata
		for (uint32_t i = 0; i < n_layers; ++i) {
			Layer &layer = model.layers[i];
			layer.type = (LayerType)readu32();
			if (layer.type == Linear) {
				uint32_t next_size = readu32();
				if (next_size == 0) {
					throw std::runtime_error("Linear layer output size 0 is not valid.");
				}
				layer.W = MatrixXfRow(next_size, size);
				layer.b = Eigen::VectorXf(next_size);
				size = next_size;
			}
			else if (layer.type == ReLU) {
				// do nothing.
			}
		}

		// Pass 2: Data
		for (uint32_t i = 0; i < n_layers; ++i) {
			if (model.layers[i].type == Linear) {
				Layer &layer = model.layers[i];
				uint32_t total = layer.W.rows() * layer.W.cols();
				size_t rW = fread(&layer.W(0, 0), sizeof(float), total, fp);
				if (rW != total) {
					throw std::runtime_error("Not enough data in file.");
				}
				size_t rb = fread(&layer.b[0], sizeof(float), layer.W.rows(), fp);
				if ((Eigen::Index)rb != layer.W.rows()) {
					throw std::runtime_error("Not enough data in file.");
				}
			}
		}

		// now the file should be empty
		char dummy;
		size_t rend = fread(&dummy, 1, 1, fp);
		if (rend != 0) {
			throw std::runtime_error("More data than expected at end of file.");
		}

		return model;
	}

	// Generates a random NN with Xavier-uniform initialization. Mainly
	// intended for unit test, etc, where the NN function doesn't matter.
	Model Model::random(int input, std::vector<int> hidden, int output)
	{
		Model m;
		m._input_dim = input;
		int size = input;
		std::vector<int> widths = hidden;
		widths.push_back(output);
		for (int h : widths) {
			Layer lay{Linear};
			float glorot_scale = std::sqrt(6.0f / (size + h));
			lay.W = glorot_scale * MatrixXfRow::Random(h, size);
			lay.b = Eigen::VectorXf::Zero(h);
			m.layers.push_back(lay);
			m.layers.push_back(Layer{ReLU});
			size = h;
		}
		// Remove the last ReLU.
		m.layers.pop_back();
		return m;
	}

	int Model::input_dim() const
	{
		return _input_dim;
	}

	int Model::output_dim() const
	{
		for (int i = layers.size() - 1; i >= 0; --i) {
			Layer const &layer = layers[i];
			if (layer.type == Linear) {
				return layer.b.rows();
			}
		}
		// Empty or all-ReLU model: Degenerate case, but allowed.
		return _input_dim;
	}

	Eigen::VectorXf Model::forward(Eigen::VectorXf x)
	{
		if (x.rows() != _input_dim) {
			throw std::runtime_error("incorrect input size");
		}

		for (Layer const &layer : layers) {
			if (layer.type == mlpfile::Linear) {
				x = layer.W * x + layer.b;
			}
			else if (layer.type == mlpfile::ReLU) {
				x = x.array().max(0);
			}
			else {
				throw std::runtime_error("unrecognized type");
			}
		}
		return x;
	}

	MatrixXfRow Model::jacobian(Eigen::VectorXf const &x)
	{
		std::unique_ptr<MatrixXfRow> J;

		std::vector<Eigen::VectorXf> stack = fwdpass_stack(*this, x);

		// Backward pass
		for (int i = (int)layers.size() - 1; i >= 0; --i) {
			if (layers[i].type == mlpfile::ReLU) {
				// ReLU
				assert (J != nullptr);
				Eigen::VectorXf const &f = stack.back();
				// TODO: Would it be possible to do this without a loop, like
				// we can in NumPy by using broadcasting?
				for (int j = 0; j < f.rows(); ++j) {
					assert (f[j] >= 0.0f);
					if (f[j] == 0.0f) {
						J->col(j) *= 0.0f;
					}
				}
			}
			else if (layers[i].type == mlpfile::Linear) {
				if (J == nullptr) {
					J.reset(new MatrixXfRow(layers[i].W));
				}
				else {
					*J = (*J) * layers[i].W;
				}
			}
			else {
				throw std::runtime_error("unrecognized type");
			}
			stack.pop_back();
		}
		assert (stack.size() == 0);
		// TODO: validate dimensionality
		return *J;
	}

	void Model::grad_update(Eigen::VectorXf x, Eigen::VectorXf y, LossGrad loss, float rate)
	{
		std::vector<Eigen::VectorXf> stack = fwdpass_stack(*this, x);

		Eigen::VectorXf grad = loss(stack.back(), y);

		// Backward pass
		for (int i = (int)layers.size() - 1; i >= 0; --i) {
			stack.pop_back();

			if (layers[i].type == mlpfile::ReLU) {
				// Eigen doesn't have logical indexing.
				grad = (stack.back().array() > 0.0f).select(grad, 0.0f);
				assert (grad.rows() == stack.back().rows());
			}
			else if (layers[i].type == mlpfile::Linear) {
				Eigen::VectorXf backprop_grad = grad.transpose() * layers[i].W;
				assert (backprop_grad.rows() == layers[i].W.cols());
				// OGD inplace
				auto expr = (rate * grad) * stack.back().transpose();
				assert (expr.rows() == layers[i].W.rows());
				assert (expr.cols() == layers[i].W.cols());
				layers[i].W = layers[i].W - expr;
				layers[i].b = layers[i].b - (rate * grad);
				grad = backprop_grad;
			}
			else {
				throw std::runtime_error("unrecognized type");
			}
		}
		assert (stack.size() == 0);
		// TODO: validate dimensionality
	}

	std::string Model::describe() const
	{
		return
			"mlpfile::Model with " + std::to_string(layers.size()) + " Layers, "
			+ std::to_string(_input_dim) + " -> " + std::to_string(output_dim());
	}

	Eigen::VectorXf squared_error(Eigen::VectorXf y, Eigen::VectorXf target)
	{
		return y - target;
	}

	Eigen::VectorXf softmax_cross_entropy(Eigen::VectorXf y, Eigen::VectorXf target)
	{
		auto e = y.array().exp();
		auto softmax = e / e.sum();
		return softmax.matrix() - target;
	}

}  // namespace mlpfile
