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
bla bla bla

**Solution**

To tackle our problem, we have decided to compare full-batch gradient descent and mini-batch gradient descent algorithms with all of the variants of differential privacy that have been studied this semester. More specifically we compared full-batch gradient descent and mini-batch gradient descent algorithms for epsilon-DP, (epsilon, delta)-DP, Rényi-DP, and zCDP.

We have also added comparisons between non-vectorized and vectorized calculations of the gradients. 

For easier understanding, here is a vague overview of the functions of main importance in our implementation:

|Function Name                        |Function description|
|-------------------------------------|--------------------|
|split_to_mini_batches(...))|Splits the data into "mini batches" (creates subsets of the data)|
|gradient()| Calculates the gradient of the logistic loss|
|gradient_vectorized(...)|Vectorized version of the calculation of the gradient. Vectorization makes it more efficient (especially when working with large datasets)|
|epsilon_delta_noisy_gradient_descent(...)|Mini-batch GD with (ε,δ)-DP using gradient()|
|vectorized_delta_noisy_gradient_descent(...)|Mini-batch GD with (ε,δ)-DP using gradient_vectorized()|
|mini_batch_noisy_gradient_RDP(...)|Mini-batch GD with Rényi-DP using gradient()|
|vectorized_mini_batch_noisy_gradient_descent_RDP(...)|Mini-batch GD with Rényi-DP using gradient_vectorized()|
|mini_batch_noisy_gradient_descent_zCDP|Mini-batch GD with zCDP using gradient()|
|vectorized_mini_batch_noisy_gradient_descent_zCDP|Mini-batch GD with zCDP using gradient_vectorized()|

## Results


## Citations
