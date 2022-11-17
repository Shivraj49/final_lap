import numpy as np
def gradientFunction(x):
    return 2*x + 6

def gradientDescent (start, gradient, learnRate, maxIteration, tolerance=0.01):
    steps = [start]
    X = start
    
    for i in range(maxIteration):
        diff = -learnRate * gradient(X)
        if np.abs(diff) <= tolerance:
            break
        X = X + diff
        steps.append(X)
    return steps, learnRate, X, len(steps)
history, learnRate, result, steps = gradientDescent(2, gradientFunction, 0.1, 100)

print("\nSteps in Gradient Descent: ", history)
print("\nLearning Rate is: ", learnRate)
print("\nNumber of steps required to reach Local Minima: ", steps)
print("\nThe Local Minimum occurs at", result)