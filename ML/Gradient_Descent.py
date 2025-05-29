class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x = init # initial guess
        n_iterations = iterations
        print(n_iterations)
        learning_rate = learning_rate 
        print(learning_rate)

        for i in range(n_iterations):
            gradient = 2 * x # derivative of function
            x = x - learning_rate * gradient
        
        return round(x, 5)
