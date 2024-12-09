# CS 3110 FINAL PROJECT - Fall 2024
### Sasha Jazmín Abuin and Hailey Schoppe

## Abbreviations
* DP: differential privacy
* GD: gradient descent
* zCDP: zero-concentrated differential privacy
  
## Problem Statement and Solution

**Problem Statement**

When calculating gradient descent as an optimization technique for machine learning training models, batch and mini-batch gradient descent have distinct tradeoffs in terms of accuracy and efficiency. Additionally, to maintain the privacy of the training data, various methods of differential privacy can be used to add noise and protect the data. These include epsilon-delta differential privacy with the Gaussian mechanism, Rényi differential privacy (RDP) using the Rényi divergence and the Gaussian Rényi algorithm, and Zero Concentrated differential privacy (zCDP), which also uses Gaussian. 

We will be testing whether batch or mini-batch gradient descent has a better base accuracy, as well as which one maintains higher accuracy while adding noise to the data through differential privacy. Furthermore, we will test what combinations of differential privacy algorithms and gradient descent functions produce the highest accuracies, and at which values of hyperparameters such as batch size, learning rate, epsilon and delta (for epsilon-delta DP), epsilon_bar and alpha (for RDP), or rho (for zCDP). 


**Solution**

To tackle our problem, we have decided to compare full-batch gradient descent and mini-batch gradient descent algorithms with the variants of differential privacy that have been studied this semester. More specifically we compared full-batch gradient descent and mini-batch gradient descent algorithms for (epsilon, delta)-DP, Rényi-DP, and zCDP. 

For calculating the gradient loss, we opted to use the vectorized version provided to us, which is important for utility and provides faster execution. 

**Batch Gradient Implementation (Hailey Schoppe)**

Batch gradient descent uses the entire training set to compute the gradients at each step, meaning it updates the model parameters after processing the entire training set. In each iteration, the gradient of the function is calculated based on the training examples, and the parameters are updated to minimize the prediction error. 

There are numerous advantages and disadvantages to Batch gradient descent. In favor of BGD, it features a consistent pattern of convergence, as well as high accuracy by leveraging the entire dataset in each iteration (or epoch). On the other hand, it does not scale well with large datasets, as the computational intensity is extreme because of its consideration of the full dataset each iteration. 

Batch Gradient Descent features a learning rate, which I notate as n throughout this project. The learning rate in BGD guides the number of steps taken towards finding the optimal solution. If the learning rate is too high the model could overshoot the minimum. Contrastingly, if set too low, the convergence takes far too long.

For my implementation, I utilized the vectorized functions provided within our project proposal feedback. I took the loss and gradient functions for logistic regression, which work together to compute how theta (the aforementioned parameters) need to be altered to reduce loss. We use this function to create the actual gradient in each iteration. Therefore, within the loop of epochs (iterations), the first line creates the gradient. Now, with the parameters calculated, we go ahead to updating the theta. We do this by subtracting the learning rate, which we represent as n, multiplied by the gradient (grad), from theta. Once the epochs are completed we return the batch gradient descent function theta.

Adding noise takes a very similar process to creating the inherent batch gradient descent. We once again calculate the gradient using the gradient and loss methods of LogisticRegression(). While adding noise, we do not need to take the sum of the gradient in batch descent, as we are looking at the entire gradient per iteration. After determining the gradient, we then use L2_clip to clip the gradient by a variable we set, b, and then use b as the sensitivity within the gaussian mechanism. We then confirm that the gradient remains an array before subtracting the learning rate multiplied by the noisy gradient from theta. Then, once all of the iterations are complete, we return theta. For epsilon-delta differential privacy, we use standard gaussian; for Renyi DP, we use gaussian_rdp; and for Zero Concentrated DP we use gaussian_zCDP. 

**Mini-batch gradient Descent Implementation (Sasha J. Abuin)**

Mini-batch gradient descent is a variant of the gradient descent algorithm, that instead of calculating the gradient of the loss function with respect to the entire data set, it does it for subsets of the data. Splitting the data into subsets creates the “mini-batches”, thus the name of the algorithm. It can be said that this variant is a combination of stochastic gradient descent and batch gradient descent. 

There are many advantages to using mini-batch gradient descent, but it has its disadvantages too. Below is a table that explores the strengths and weaknesses of this algorithm: 

|Advantages|Disadvantages|
|----------|-------------|
|Model parameters get updated more frequently, thus leading to faster convergence if the batch size parameter is properly set|Can be less accurate than batch gradient descent depending on the dataset or if the hyperparameters are not set properly|
|Performs well with big datasets|Tradeoff between fast convergence and noisy updates|
|Can provide better accuracy if compared to Stochastic Gradient Descent|Have to pick “learning rate” hyperparameter.
|Parameter updates can be less noisy if the batch size parameter is properly set|We have to pick the value for the “batch_size” hyperparameter. If too small: higher variance when updating parameters. If too big: slower convergence|

For the implementation of the algorithm, five steps were taken:
1. Define a function that splits data into mini-batches (subsets of the whole dataset) 
2. Define the loss function that measures how good our model is (original function taken from in-class exercises)
3. Define the vectorized version of the gradient function. The gradient is a vector that indicates the rate of change of the loss in each direction (Implementation taken from project feedback; check citations section)
4. Define an avg_grad function that computes the average gradient over each mini-batch (original function taken from in-class exercises)
5. Define a mini_batch_gradient_descent function that computes the gradient using mini-batches for each iteration with the different variants of differential privacy (modified versions of functions provided in in-class exercises)
   
## Results
The table below shows the average accuracy for 20 runs of each implementation. The values of our privacy parameters and hyperparameters are the following:
* Epsilon = 1.0
* Delta = 1e^-5
* Epsilon_bar = 0.1
* Alpha = 20
* Rho = 1
* Learning rate for batch-GD = 1
* Learning rate for mini-batch-GD = 0.01
* Batch size (mini-batch-GD only) = 64

||Batch-GD|Mini-Batch-GD|
|-|-------|-------------|
|(Epsilon,Delta)-DP|0.4780904467049978|0.7285106147722247|
|Rényi-DP|0.5123949579831933|0.8061311366651924|
|zCDP|0.630827067669173|0.8177797434763379|

*In the code, for each implementation we have tested various values as parameters for differential privacy*

Overall, we can see that mini-batch GD has provided more accurate results overall, with the highest accuracy coming from zCDP. This could be for multiple reasons, as for example the combinations of hyperparameters. However, mini-batch GD also has the advantage of the parameters being updated more frequently, which could result in better accuracy and less noise being added during the calculation of the gradients. 

## Link to video presentation

## Citations
[1] A. Agrawall, "Mini-batch Gradient Descent," Inside Learning Machines, [Online]. Available: https://insidelearningmachines.com/mini_batch_gradient_descent/.

[2] A. Agrawall, "Batch vs Stochastic vs Mini-batch Gradient Descent Techniques," Medium, 2-May-2020. [Online]. Available: https://medium.com/@amannagrawall002/batch-vs-stochastic-vs-mini-batch-gradient-descent-techniques-7dfe6f963a6f.

[3] Deepgram, "Batch gradient descent," Deepgram AI Glossary. [Online]. Available: https://deepgram.com/ai-glossary/batch-gradient-descent. 

[4]J. Near, "CS3110 Data Privacy Exercises," University of Vermont, GitHub, [Online]. Available: https://github.com/jnear/cs3110-data-privacy/tree/main/exercises.

[5] S. Sunblaze, "Logistic Regression in DPML Benchmark," GitHub, [Online]. Available: https://github.com/sunblaze-ucb/dpml-benchmark/blob/master/lossfunctions/logistic_regression.py#L12.


