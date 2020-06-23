import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


inputs = np.linspace(-10, 10, 1000)
outputs = sigmoid(inputs)

plt.plot(inputs, outputs)
plt.xlabel('inputs')
plt.ylabel('outputs')
plt.title('Sigmoid function')
plt.show()