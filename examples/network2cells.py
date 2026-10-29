import numpy as np
X = np.array([
    [0,0], 
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],        
    [1],
    [1],
    [0]
])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

np.random.seed(43)

n_input = 2
n_hidden = 2
n_output = 1

weights_1 = np.random.uniform(-1, 1, (n_input, n_hidden))
bias_1 = np.zeros((1, n_hidden))

weights_2 = np.random.uniform(-1, 1, (n_hidden, n_output))
bias_2 = np.zeros((1, n_output))

learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):
    hidden_input = np.dot(X, weights_1) + bias_1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output,weights_2) + bias_2
    final_output = sigmoid(final_input)

    error = y - final_output

    delta_output = error * sigmoid_derivative(final_output)
    error_hidden = delta_output.dot(weights_2.T)
    delta_hidden = error_hidden * sigmoid_derivative(hidden_output)

    weights_2 += hidden_output.T.dot(delta_output) * learning_rate
    bias_2 += np.sum(delta_output, axis=0, keepdims=True) * learning_rate

    weights_1 += X.T.dot(delta_hidden) * learning_rate
    bias_1 += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        loss = np.mean(error ** 2)
        print(f"Epoch {epoch:5d} | Loss: {loss:.4f}")

print()
print("training results")
print(f"{'x1':>5} {'x2':>5} {'raw':>5} {'rounded':>10} {'expected':>10}")

for i in range(len(X)):
    x1, x2 = X[i]
    raw = final_output[i][0]
    rounded = round(raw)
    expected = y[i][0]
    result = "OK" if rounded == expected else "FAIL"
    print(f"{'x1':>3} {'x2':>3} {'raw':>14.4f} {'rounded':>11} {'expected':>9} {result}")