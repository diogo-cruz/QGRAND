import numpy as np
from scipy.special import comb
from itertools import combinations, product
from .QGRAND import QGRAND

from datetime import datetime
from time import time, sleep

class WeightStatistics:

    def __init__(self,
                n,
                k_list,
                gate_list,
                runs = 1,
                max_t = 2,
                p = np.array([1e-3])):

        self.n, self.k_list, self.gate_list = n, np.array(k_list), np.array(gate_list)
        self.runs, self.max_t, self.p = runs, max_t, p

        n_gates = self.gate_list.shape[0]
        self.infidelity = np.zeros((self.k_list.shape[0], n_gates, runs, n, p.shape[0]))
        self.iterations = np.zeros_like(self.infidelity)
        self.correctable_fraction = np.zeros((self.k_list.shape[0], n_gates, runs, n))

        self.args = list(product([n], k_list, gate_list, range(runs), [max_t], p))

        @staticmethod
        def _get_data_for_arg(args):
            n, k, n_gates, run, max_t, p = args
            start = time()
            trial = QGRAND(n = n, k = k, num_gates = n_gates)
            trial.get_encoding(fast=True)
            middle = time()
            trial.get_failure_rate_fast(prob=p, max_t=max_t, fill_syndromes=False)
            end = time()
            # stab_size = [n - stab.count('I') for stab in trial.stabilizers]
            # print(np.mean(stab_size), stab_size)
            print("n: {}\t k: {}\t gates: {}\t run: {}".format(n, k, n_gates, run))
            print("Encoding time:", middle-start)
            print("Failure time:", end-middle)
            return trial.failure_rate, trial.correctable_fraction, trial.mean_iterations

        def get_data(self):
            self.results = list(map(self._get_data_for_arg, self.args))