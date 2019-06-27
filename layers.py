import numpy as np
class ConvLayer():
    def __init__(self, kernel, feature_size, l2=0.0, stride=1):
        self.kernel = kernel
        self.feature_size = feature_size
        self.l2 = l2
        self.stride = stride
        self.v = np.array([0.0, 0.0, 0.0])
        self.type = 1

    def __str__(self):
        _str = 'C[K:{}-F:{}-L2:{:.4f}]'.format(self.kernel, self.feature_size, self.l2)
        return _str


class PoolLayer():
    def __init__(self, kernel=2, stride=2):
        self.kernel = kernel
        self.stride = stride
        self.type = 2
    def __str__(self):
        _str = 'P[K:{}-S:{}]'.format(self.kernel, self.stride)
        return _str

if __name__ == '__main__':
    print(1)