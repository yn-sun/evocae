from ae import CAE
import os
import get_data
import tensorflow as tf
from measure import *
import numpy as np
import copy
from ae import CAE
from utils import *
import copy
class Population():
    def __init__(self, params):
        self.gbest_score = 999999.0
        self.gbest = None
        self.indi = []
        self.params = params

    def set_gbest(self, g_cae):
        units = g_cae.units
        units = copy.deepcopy(units)
        cae = CAE()
        cae.set_units(units)
        cae.score = g_cae.score
        self.gbest_score = cae.score
        self.gbest = cae

    def init_population(self):
        for _ in range(self.params['pop_size']):
            cae = CAE()
            cae.init(self.params['cae_length'])
            self.indi.append(cae)

    def get_pop_size(self):
        return len(self.indi)

class PSO():
    def __init__(self, params):
        self.params = params


    def init_population(self):
        pops = Population(self.params)
        pops.init_population()
        self.pops = pops

    def evaluate_fitness(self, pops, gen_no):
        f= FitnessAssignment(pops, self.params)
        f.evalue_all(gen_no)

    def begin_to_evolve(self):
        print('Begin to ...')
        self.init_population()
        for i in range(self.params['total_generation']):
            print('Begin {}/{} generation...'.format(i, self.params['total_generation']))
            self.evaluate_fitness(self.pops, i)
            self.update(i)


    def update(self, gen_no):
        # for the first generation, just update the pbest, gbest, and no update
        if self.pops.gbest is None:
            for i in range(self.pops.get_pop_size()):
                cae = self.pops.indi[i]
                cae.set_pbest(cae)
                if cae.score < self.pops.gbest_score:
                    self.pops.set_gbest(cae)

            for i in range(self.pops.get_pop_size()):
                cae = self.pops.indi[i]
                log_particle_info(i, 'The {} generation...'.format(gen_no))
                log_particle_info(i, 'g_best:' + str(self.pops.gbest))
                log_particle_info(i, 'p_best:' + str(cae.p_best))
                log_particle_info(i, 'before:' + str(cae))
                cae.update(self.pops.gbest)
                log_particle_info(i, 'after:' + str(cae))
                self.pops.indi[i] = cae

        else:
            for i in range(self.pops.get_pop_size()):
                cae = self.pops.indi[i]
                log_particle_info(i, 'The {} generation...'.format(gen_no))
                log_particle_info(i, 'g_best:' + str(self.pops.gbest))
                log_particle_info(i, 'p_best:' + str(cae.p_best))
                log_particle_info(i, 'before:' + str(cae))
                cae.update(self.pops.gbest)
                log_particle_info(i, 'after:' + str(cae))
                self.pops.indi[i] = cae

            for i in range(self.pops.get_pop_size()):
                cae = self.pops.indi[i]
                if cae.score < cae.b_score:
                    cae.set_pbest(cae)
                if cae.score < self.pops.gbest_score:
                    self.pops.set_gbest(cae)



if __name__ == '__main__':
    #cuda2
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_data = get_data.get_train_data()
    #test_data, test_label = get_data.get_test_data()
    validate_data = get_data.get_validate_data()

    params = {}
    params['train_data'] = train_data
    #params['train_label'] = train_label
    params['validate_data'] = validate_data
    #params['validate_label'] = validate_label
    #params['test_data'] = test_data
    #params['test_label'] = test_label
    params['pop_size'] = 50
    params['num_class'] = 10
    params['cae_length'] = 5
    params['x_prob'] = 0.9
    params['x_eta'] = 20
    params['m_prob'] = 0.1
    params['m_eta'] = 20
    params['total_generation'] = 50

    params['batch_size'] = 128
    params['epochs'] = 5
    params['input_size'] = train_data.shape[2]
    params['channel'] = train_data.shape[3]
    pso = PSO(params)
    pso.begin_to_evolve()
