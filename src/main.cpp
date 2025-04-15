#include "network.h"

//
// FURTHER NOTICE
//
// Typically, more complex neural networks with larger datasets implement the following concepts:
//
// Because this network is really simple and the dataset is small, I didn't implement these concepts, BUT these are essential
// and in an actual project, you should implement them.
//
// 1. Batch Processing:
//    - Instead of updating weights after every individual data point, the training data is divided into smaller subsets, called "batches".
//    - Gradients are computed for each batch, and weights are updated after processing the entire batch.
//    - This approach is more computationally efficient, smooths the updates, and helps the model generalize better.
//    - For instance, with 1,000 training samples and a batch size of 100:
//      - The model processes 100 samples at a time, computes the gradients, and updates weights.
//      - This results in only 10 updates per epoch, as opposed to 1,000 updates.
//
// 2. Shuffling Data:
//    - To prevent the model from learning unintended patterns based on the data order, the training data is usually shuffled at the beginning of each epoch.
//    - This helps improve generalization and prevent overfitting to the specific sequence of samples.
//
// 3. Validation Set:
//    - A separate subset of data, known as the validation set, is typically used to evaluate the model’s performance during training.
//    - This allows us to monitor overfitting and ensure the model generalizes well to unseen data.
//    - The validation loss is computed after each epoch but is not used in weight updates.
//
// 4. Optimizers:
//    - Optimizers update the model’s weights to minimize the loss function.
//    - Common optimizers include SGD (Stochastic Gradient Descent), Adam, RMSprop, and Adagrad.
//    - In this code, a simple SGD optimizer is used. Although more advanced optimizers can improve performance in larger, more complex networks, 
//	they are not strictly necessary for a small network like this.
//
// Network Setup:
// - The data used here represents the classic XOR problem, where the model learns to output 1 for (0,1) and (1,0) inputs, and 0 for (0,0) and (1,1) inputs.
//
// Also you would usually implement neural networks in cuda since it is much much faster than cpu, but I didn't do that here.
//

int main()
{
	std::vector<NetInput> data;

	// initialize XOR data: input -> target
	data = {
		NetInput({0, 1}, {1}),
		NetInput({1, 0}, {1}),
		NetInput({0, 0}, {0}),
		NetInput({1, 1}, {0})
	};
	
	Network net;
	net.train(data, 100'000);  // train the network for 100,000 epochs

	// test the trained network
	std::cout << "\nTesting network:\n" << std::endl;
	for (size_t i = 0; i < data.size(); i++)
	{
		Array prediction = net.forward(data[i].input);
		Array target = data[i].target;

		for (int j = 0; j < prediction.getSize(); j++)
			std::cout << "Data " << (i + 1) 
					  << ": Prediction = " << prediction[j] 
					  << " Target = " << target[j] << std::endl;
	}

	return 0;
}
