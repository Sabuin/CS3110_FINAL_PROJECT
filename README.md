# CS 3110 FINAL PROJECT - Fall 2024
### Sasha Jazmín Abuin and Hailey Schoppe

## Abbreviations
* DP: differential privacy
* GD: gradient descent
* zCDP: zero-concentrated differential privacy
  
## Problem Statement and Solution
TODO: 
* Problem statement
* A technical description of your solution, with emphasis on anything that makes your solution unique; your description should be sufficient to enable the reader to reproduce your implementation
* A description of the results - if you’ve evaluated your implementation on real data, describe how well it works

**Problem Statement**

When calculating gradient descent as an optimization technique for machine learning training models, batch and mini-batch gradient descent have distinct tradeoffs in terms of accuracy and efficiency. Additionally, to maintain the privacy of the training data, various methods of differential privacy can be used to add noise and protect the data. These include epsilon-delta differential privacy with gaussian mechanism, Renyi differential privacy (RDP) using the Renyi divergence and a gaussian renyi algorithm, and Zero Concentrated differential privacy (zCDP), also using gaussian. 

We will be testing whether batch or mini-batch gradient descent has a better base accuracy, as well as which maintains higher accuracy while adding noise to the data through differential privacy. Furthermore, we will test which combination of differential privacy algorithm and gradient descent function produces the highest accuracies, and at which values of inputs such as batch size, epsilon and delta (for epsilon-delta DP), epsilon_bar and alpha (for RDP), or rho (for zCDP).


**Solution**

To tackle our problem, we have decided to compare full-batch gradient descent and mini-batch gradient descent algorithms with the variants of differential privacy that have been studied this semester. More specifically we compared full-batch gradient descent and mini-batch gradient descent algorithms for (epsilon, delta)-DP, Rényi-DP, and zCDP. 

**Batch Gradient Implementation (Hailey Schoppe)**

Batch gradient descent uses the entire training set to compute the gradients at each step, meaning it updates the model parameters after processing the entire training set. In each iteration, the gradient of the function is calculated based on the training examples, and the parameters are updated to minimize the prediction error. 

There are numerous advantages and disadvantages to Batch gradient descent. In favor of BGD, it features a consistent pattern of convergence, as well as high accuracy by leveraging the entire dataset in each iteration (or epoch). On the other hand, it does not scale well with large datasets, as the computational intensity is extreme because of its consideration of the full dataset each iteration. 

Batch Gradient Descent features a learning rate, which I notate as n throughout this project. The learning rate in BGD guides the number of steps taken towards finding the optimal solution. If the learning rate is too high the model could overshoot the minimum. Contrastingly, if set too low, the convergence takes far too long.

For my implementation, I utilized the vectorized functions provided within our project proposal feedback. I took the loss and gradient functions for logistic regression, which work together to compute how theta (the aforementioned parameters) need to be altered to reduce loss. We use this function to create the actual gradient in each iteration. Therefore, within the loop of epochs (iterations), the first line creates the gradient. From here, we do not need to calculate the average gradient as we did with stochastic gradient descent, because the gradient method internally computes the average gradient by dividing the sum by the number of training examples (X.shape[0]). Therefore, with the parameters calculated, we go ahead to updating the theta. We do this by subtracting the learning rate, which we represent as n, multiplied by the gradient (grad), from theta. Once the epochs are completed we return the batch gradient descent function theta.

Adding noise takes a very similar process to creating the inherent batch gradient descent. We once again calculate the gradient using the gradient and loss methods of LogisticRegression(). Again, this function returns the averaged gradient, so we do not need to take the average again. For epsilon differential privacy, we then go straight to adding noise to the gradient with laplace mechanism. We use a sensitivity of __ and an epsilon of the total epsilon divided by the number of iterations to meet the privacy budget. We then confirm that the gradient remains an array before subtracting the learning rate multiplied by the noisy gradient from theta. Then, once all of the iterations are complete, we return theta.

The process of adding noise is marginally different for epsilon-delta, Renyi, and Zero Concentrated differential privacy. Each of these require L2 clipping, as they utilize versions of the gaussian mechanism to add noise. After determining the gradient, we then use L2_clip to clip the gradient by a variable we set, b, and then use b as the sensitivity within the gaussian mechanism. For epsilon-delta differential privacy, we use standard gaussian; for Renyi DP, we use gaussian_rdp; and for Zero Concentrated DP we use gaussian_zCDP. Following adding noise, the steps are the same as detailed above with the epsilon differential privacy.

**Mini-batch gradient Descent Implementation (Sasha J. Abuin)**

Mini-batch gradient descent is a variant of the gradient descent algorithm, that instead of calculating the gradient of the loss function with respect to the entire data set, it does it for subsets of the data. Splitting the data into subsets is known as “mini-batches”, thus the name of the algorithm. It can be said that this variant is a combination of stochastic gradient descent and batch gradient descent. 

There are many advantages to using mini-batch gradient descent, but it has its disadvantages too. Below is a table that explores the strengths and weaknesses of this algorithm: 

|Advantages|Disadvantages|
|----------|-------------|
|Model parameters get updated more frequently, thus leading to faster convergence if the batch size parameter is properly set|Can be less accurate than batch gradient descent at times|
|Performs well with big datasets|Tradeoff between fast convergence and noisy updates|
|Can provide better accuracy if compared to Stochastic Gradient Descent|Have to pick “learning rate” hyperparameter.
|Parameter updates can be less noisy if the batch size parameter is properly set|We have to pick the value for the “batch_size” hyperparameter. If too small: higher variance when updating parameters. If too big: slower convergence|

For calculating the gradient loss, we opted to use the vectorized version provided to us, which is important for utility and provides faster execution. 
For the implementation of the algorithm, five steps were taken:
1. Define a function that splits data into mini-batches (subsets of the whole dataset) 
2. Define the loss function that measures how good our model is (original function taken from in-class exercises)
3. Define the vectorized version of the gradient function. The gradient is a vector that indicates the rate of change of the loss in each direction (Implementation taken from project feedback; check citations section)
4. Define an avg_grad function that computes the average gradient over each mini-batch (original function taken from in-class exercises)
5. Define gradient_descent function that computes gradient using mini-batches for each iteration with the different variants of differential privacy (modified versions of functions provided in in-class exercises)
   
## Results


## Citations
[1] A. Agrawall, "Mini-batch Gradient Descent," Inside Learning Machines, [Online]. Available: https://insidelearningmachines.com/mini_batch_gradient_descent/.

[2] A. Agrawall, "Batch vs Stochastic vs Mini-batch Gradient Descent Techniques," Medium, 2-May-2020. [Online]. Available: https://medium.com/@amannagrawall002/batch-vs-stochastic-vs-mini-batch-gradient-descent-techniques-7dfe6f963a6f.

[3]J. Near, "CS3110 Data Privacy Exercises," University of Vermont, GitHub, [Online]. Available: https://github.com/jnear/cs3110-data-privacy/tree/main/exercises.

[3] S. Sunblaze, "Logistic Regression in DPML Benchmark," GitHub, [Online]. Available: https://github.com/sunblaze-ucb/dpml-benchmark/blob/master/lossfunctions/logistic_regression.py#L12.


