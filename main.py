import numpy as np
import matplotlib.pyplot as plt

SIZE = 64
LAYERS = 4
class MLP():
    def __init__(self):
        self.V0 = [np.random.randn(2, SIZE)] + [np.random.randn(SIZE, SIZE) * 0.2 for _ in range(LAYERS - 2)]
        self.B0 = np.zeros((LAYERS-1, SIZE))
        self.V1 = np.random.randn(SIZE, 1) * 0.1
        self.B1 = 0.0
        self.alpha = 0.1

        # gradients
        self.gV0 = [np.zeros_like(v) for v in self.V0]
        self.gB0 = np.zeros((LAYERS-1, SIZE))
        self.gV1 = np.zeros((SIZE, 1))
        self.gB1 = 0

        self.params = np.zeros((LAYERS+1, SIZE))

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_d(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_d(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forvard(self, x):
        # print(x.shape, self.V1.shape)
        # self.params[0][:2] = x
        # x = x @ self.V0 + self.B0

        # self.params[1] = x
        # x = self.relu(x)

        # x = x @ self.V1 + self.B1
        # self.params[2] = x
        # x = self.relu(x)
        # return x[0]

        self.params[0][:2] = x
        for i in range(LAYERS-1):
            x = x @ self.V0[i] + self.B0[i]
            self.params[i+1] = x
            x = self.relu(x)
        
        x = x @ self.V1 + self.B1
        self.params[LAYERS][:1] = x

        x = self.sigmoid(x)
        return x[0]


    def gradient(self, y):
        n = LAYERS - 1
        z_out = self.params[LAYERS][0]
        d = 2 * (self.sigmoid(z_out) - y) * self.sigmoid_d(z_out)

        self.gB1 = d
        self.gV1 = (d * self.relu(self.params[n])).reshape(SIZE, 1)

        delta = d * self.V1[:, 0] * self.relu_d(self.params[n])
        for i in range(n - 1, -1, -1):
            a_prev = self.params[0][:2] if i == 0 else self.relu(self.params[i])
            self.gV0[i] = np.outer(a_prev, delta)
            self.gB0[i] = delta
            if i > 0:
                delta = (self.V0[i] @ delta) * self.relu_d(self.params[i])
    
    def backvard(self):
        for i in range(LAYERS - 1):
            self.V0[i] -= self.alpha * self.gV0[i]
            self.B0[i] -= self.alpha * self.gB0[i]

        self.V1 -= self.alpha * self.gV1
        self.B1 -= self.alpha * self.gB1
        
    def loss(self, x, y):
        return np.linalg.norm(x-y)

    def train(self, X, Y):
        gV0 = [np.zeros_like(v) for v in self.V0]
        gB0 = [np.zeros_like(b) for b in self.B0]
        gV1 = np.zeros_like(self.V1)
        gB1 = 0.0
        error = 0.0

        for a, b in zip(X, Y):
            t = self.forvard(a)
            error += (t - b) ** 2

            self.gradient(b)
            for i in range(LAYERS - 1):
                gV0[i] += self.gV0[i]
                gB0[i] += self.gB0[i]
            gV1 += self.gV1
            gB1 += self.gB1

        n = len(X)
        self.gV0 = [g / n for g in gV0]
        self.gB0 = [g / n for g in gB0]
        self.gV1 = gV1 / n
        self.gB1 = gB1 / n
        self.backvard()

        return error / n
    
# print(X_train.shape)
# testing
model = MLP()
epoch = 1300
batch = 64

# weight dumping
def train(X_train, y_train):
    for _ in range(epoch):
        t = np.random.permutation(len(X_train))
        error = 0
        for i in range(0, len(X_train), batch):
            idx = t[i:i+64]
            error += model.train(X_train[idx], y_train[idx])

    layers = [(model.V0[i], model.B0[i]) for i in range(LAYERS - 1)]
    layers.append((model.V1, np.array([model.B1])))
    return layers
        
