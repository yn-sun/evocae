import numpy as np
from layers import ConvLayer, PoolLayer
import copy


class CAE():

    def __init__(self):
        self.score = 0.00
        self.b_score = 99999999.0

        self.w = 0.72984
        self.c1 = 1.193
        self.c2 = 1.193
        self.units = []
        self.conv_feature_size_range = [20, 100]
        self.conv_feature_size_maxv = self.conv_feature_size_range[1] - self.conv_feature_size_range[0]
        self.conv_kernel_range = [2, 5]
        self.conv_kernel_maxv = self.conv_kernel_range[1] - self.conv_kernel_range[0]
        self.conv_l2_range = [0.0001, 0.01]
        self.conv_l2_maxv = self.conv_l2_range[1] - self.conv_l2_range[0]


    def reset_state(self):
        self.score = 0.00

    def set_pbest(self, p_cae):
        units = p_cae.units
        units = copy.deepcopy(units)
        cae = CAE()
        cae.set_units(units)
        cae.score = p_cae.score
        self.b_score = cae.score
        self.p_best = cae

    def set_units(self, new_units):
        self.units = new_units

    def random_a_conv(self):
        kernel = self.random_conv_kernel_sze()
        feature_size = self.random_conv_feature_size()
        l2 = self.random_l2()
        conv = self.create_a_conv(kernel, feature_size, l2, 1)
        return conv
    def random_a_pool(self):
        pool = self.create_a_pool(kernel_size=2, stride=2)
        return pool

    def random_conv_feature_size(self):
        feature_size = self.randint(self.conv_feature_size_range[0], self.conv_feature_size_range[1])
        return feature_size

    def random_conv_kernel_sze(self):
        kernel = self.randint(self.conv_kernel_range[0], self.conv_kernel_range[1])
        return kernel

    def random_l2(self):
        l2 = (self.conv_l2_range[1] - self.conv_l2_range[0]) *np.random.random() + self.conv_l2_range[0]
        return l2
    def get_length(self):
        return len(self.units)

    def create_a_conv(self, kernel, feature_size, l2, stride):
        conv = ConvLayer(kernel, feature_size, l2, stride)
        return conv

    def create_a_pool(self, kernel_size, stride):
        pool = PoolLayer(kernel=kernel_size, stride=stride)
        return pool

    def randint(self, low, high):
        return np.random.random_integers(low, high).item()

    def rand(self):
        return np.random.random()

    def update(self, g_best):
        p_best_units = self.p_best.units
        g_best_units = g_best.units
        current_u = self.units

        g_best_conv_list = []
        for i in range(len(g_best_units)-1):
            g_best_conv_list.append(g_best_units[i])

        p_best_conv_list = []
        for i in range(len(p_best_units)-1):
            p_best_conv_list.append(p_best_units[i])

        current_conv_list = []
        v_list = []
        for i in range(len(current_u)-1):
            current_conv_list.append(current_u[i])
            v_list.append(current_u[i].v)

        new_unit_list = []
        min_length = min(len(g_best_conv_list), len(p_best_conv_list))
        for i in range(min_length):
            gbest_unit = g_best_conv_list[i]
            gbest_kernel = gbest_unit.kernel
            gbest_feature_size = gbest_unit.feature_size
            gbest_l2 = gbest_unit.l2

            pbest_unit = p_best_conv_list[i]
            pbest_kernel = pbest_unit.kernel
            pbest_feature_size = pbest_unit.feature_size
            pbest_l2 = pbest_unit.l2

            current_unit = current_conv_list[i]
            current_unit_kernel = current_unit.kernel
            current_unit_feature_size = current_unit.feature_size
            current_unit_l2 = current_unit.l2

            v_current = v_list[i]
            kernel_old_v = v_current[0]
            feature_size_old_v = v_current[1]
            l2_old_v = v_current[2]

            kernel_new_v = self.w*kernel_old_v + self.c1*self.rand()*(gbest_kernel-current_unit_kernel) + self.c2*self.rand()*(pbest_kernel-current_unit_kernel)
            kernel_new_v = self.adjust_kernel_v(kernel_new_v)
            feature_size_new_v = self.w*feature_size_old_v + self.c1*self.rand()*(gbest_feature_size-current_unit_feature_size) + self.c2*self.rand()*(pbest_feature_size-current_unit_feature_size)
            feature_size_new_v = self.adjust_feature_size_v(feature_size_new_v)
            l2_new_v = self.w*l2_old_v + self.c1*self.rand()*(gbest_l2-current_unit_l2) + self.c2*self.rand()*(pbest_l2-current_unit_l2)
            l2_new_v = self.adjust_l2_v(l2_new_v)


            new_v_list = [kernel_new_v, feature_size_new_v, l2_new_v]
            new_kernel = self.adjust_kernel(current_unit_kernel + kernel_new_v)
            new_feature_size = self.adjust_feature_size(current_unit_feature_size + feature_size_new_v)
            new_l2 = self.adjust_l2(current_unit_l2 + l2_new_v)

            new_unit = self.create_a_conv(new_kernel, new_feature_size, new_l2, stride=1)
            new_unit.v = new_v_list
            new_unit_list.append(new_unit)

        if min_length < len(p_best_conv_list):
            for i in range(min_length, len(p_best_conv_list)):
                pbest_unit = p_best_conv_list[i]
                pbest_kernel = pbest_unit.kernel
                pbest_feature_size = pbest_unit.feature_size
                pbest_l2 = pbest_unit.l2

                current_unit = current_conv_list[i]
                current_unit_kernel = current_unit.kernel
                current_unit_feature_size = current_unit.feature_size
                current_unit_l2 = current_unit.l2

                v_current = v_list[i]
                kernel_old_v = v_current[0]
                feature_size_old_v = v_current[1]
                l2_old_v = v_current[2]

                kernel_new_v = self.w*kernel_old_v + self.c2*self.rand()*(pbest_kernel-current_unit_kernel)
                kernel_new_v = self.adjust_kernel_v(kernel_new_v)
                feature_size_new_v = self.w*feature_size_old_v + self.c2*self.rand()*(pbest_feature_size-current_unit_feature_size)
                feature_size_new_v = self.adjust_feature_size_v(feature_size_new_v)
                l2_new_v = self.w*l2_old_v + self.c2*self.rand()*(pbest_l2-current_unit_l2)
                l2_new_v = self.adjust_l2_v(l2_new_v)

                new_v_list = [kernel_new_v, feature_size_new_v, l2_new_v]
                new_kernel = self.adjust_kernel(current_unit_kernel + kernel_new_v)
                new_feature_size = self.adjust_feature_size(current_unit_feature_size + feature_size_new_v)
                new_l2 = self.adjust_l2(current_unit_l2 + l2_new_v)

                new_unit = self.create_a_conv(new_kernel, new_feature_size, new_l2, stride=1)
                new_unit.v = new_v_list
                new_unit_list.append(new_unit)
        new_unit_list.append(current_u[-1])
        self.units = new_unit_list





    def adjust_kernel(self, kernel):
        if kernel < self.conv_kernel_range[0]:
            kernel = self.conv_kernel_range[0]
        elif kernel > self.conv_kernel_range[1]:
            kernel = self.conv_kernel_range[1]
        return int(kernel)

    def adjust_feature_size(self, feature_size):
        if feature_size < self.conv_feature_size_range[0]:
            feature_size = self.conv_feature_size_range[0]
        elif feature_size > self.conv_feature_size_range[1]:
            feature_size = self.conv_feature_size_range[1]
        return int(feature_size)
    def adjust_l2(self, l2):
        if l2 < self.conv_l2_range[0]:
            l2 = self.conv_l2_range[0]
        elif l2 > self.conv_l2_range[1]:
            l2 = self.conv_l2_range[1]
        return l2


    def adjust_kernel_v(self, kernel_new_v):
        if np.abs(kernel_new_v) > self.conv_kernel_maxv:
            kernel_new_v =  (kernel_new_v/np.abs(kernel_new_v))*self.conv_kernel_maxv
        return kernel_new_v
    def adjust_feature_size_v(self, feature_size_new_v):
        if np.abs(feature_size_new_v) > self.conv_feature_size_maxv:
            feature_size_new_v = (feature_size_new_v/np.abs(feature_size_new_v))*self.conv_feature_size_maxv
        return feature_size_new_v
    def adjust_l2_v(self, l2_new_v):
        if np.abs(l2_new_v) > self.conv_l2_maxv:
            l2_new_v = (np.abs(l2_new_v)/l2_new_v)*self.conv_l2_maxv
        return l2_new_v

    def init(self, max_length):
        num = np.random.randint(2, max_length+1)
        #a conv at the ehad and a pool at the tail
        head = self.random_a_conv()
        tail = self.random_a_pool()

        self.units.append(head)
        for _ in range(num - 2):
            conv = self.random_a_conv()
            self.units.append(conv)
        self.units.append(tail)

    def __str__(self):
        _str = []
        _str.append('len:{}'.format(self.get_length()))
        _str.append('score:{:.2E}'.format(self.score))
        for u in self.units:
            _str.append(str(u))
        return ' '.join(_str)

if __name__ == '__main__':
    g_best = CAE()
    u1 = g_best.create_a_conv(kernel=1, feature_size=1, l2=0.01, stride=1)
    u2 = g_best.create_a_conv(kernel=2, feature_size=2, l2=0.02, stride=1)
    u3 = g_best.create_a_conv(kernel=3, feature_size=3, l2=0.03, stride=1)
    p1 = g_best.create_a_pool(kernel_size=2, stride=2)
    g_best.set_units([ u1, p1])

    p_best = CAE()
    u1 = p_best.create_a_conv(kernel=4, feature_size=4, l2=0.04, stride=1)
    u2 = p_best.create_a_conv(kernel=5, feature_size=5, l2=0.05, stride=1)
    p1 = p_best.create_a_pool(kernel_size=6, stride=6)
    p_best.set_units([u1, u2, p1])

    current = CAE()
    u1 = current.create_a_conv(kernel=7, feature_size=7, l2=0.07, stride=1)
    u2 = current.create_a_conv(kernel=8, feature_size=8, l2=0.08, stride=1)
    p1 = current.create_a_pool(kernel_size=9, stride=9)
    current.set_units([u1, u2, p1])
    current.set_pbest(p_best)
    for i in range(5):
        current.update(g_best)
        for j in range(2):
            print(current.units[j].v)
        print(current)







