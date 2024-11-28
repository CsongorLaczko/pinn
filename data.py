import time
from pathlib import Path

import wandb
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import Matern

from solvers.fem.fem import *
from solvers.pinn.util import get_absolute_path, read_config_file, np_load, format_time, set_seeds


def _generate_error(right_hand, solutions, c, v, N):
    pred = np.asarray(solutions)
    hh = np.linspace(0, 1, N)
    h = hh[1] - hh[0]
    error = np.sqrt(np.sum((right_hand[:, 1:-1] - (
            -c * (pred[:, :-2] + pred[:, 2:] - 2 * pred[:, 1:-1]) / (h ** 2) + v * pred[:, 1:-1])) ** 2)) / \
            right_hand.shape[0]
    return error


class DataGenerator:
    def __init__(self, bc_type='DD', args=read_config_file('runner_config.yml')['data'], group_id='coef'):
        self.args = args
        self.group_id = group_id if group_id is not None else wandb.util.generate_id(2)
        self.args['bc_type'] = bc_type

    def get_data(self):
        self._load_or_generate_inputs()
        self._load_or_generate_outputs()
        return self.inputs, self.outputs

    def _load_or_generate_inputs(self):
        if self.args['load']:
            data = np_load(f'data/{self.group_id}/' + self.args['input_file_name'], self.args['bc_type'] + '.npz')
            if data is not None:
                self.inputs = data['right_hand'], data['bc'], data['c'], data['v']
                right_hand, bc_values, c_val, v_val = self.inputs
                N = right_hand.shape[1]
                return right_hand, bc_values, c_val, v_val, N

        self._generate_inputs()
        right_hand, bc_values, c_val, v_val = self.inputs
        N = right_hand.shape[1]
        return right_hand, bc_values, c_val, v_val, N

    def _generate_inputs(self):
        start_time = time.time()

        self._generate_right_hand()
        self._generate_bc_values(-10, 10)
        self._generate_coefficients()

        right_hand, bc, c, v = self._match_dimensions()

        self.inputs = right_hand, bc, c, v

        if self.args['plot']:
            self._plot_inputs(np.linspace(0, self.args['L'], self.args['N']), right_hand)
            self._plot_cv(np.linspace(0, self.args['L'], self.args['N']), c, v)
        if self.args['save']:
            self._save_inputs(right_hand, bc, c, v)

        if self.args['print_time']:
            print(f"Generated inputs in {format_time(time.time() - start_time)}.")

    def _generate_right_hand(self):
        start_time = time.time()

        kernel = 5.0 * Matern(length_scale=0.1, nu=self.args['nu'])
        x = np.linspace(0, self.args['L'], self.args['N'])[:, np.newaxis]
        cov_matrix = kernel(x)

        self.right_hand_train = np.random.multivariate_normal(mean=np.zeros(len(x)), cov=cov_matrix,
                                                              size=self.args['num_f'])
        self.right_hand_valid = np.random.multivariate_normal(mean=np.zeros(len(x)), cov=cov_matrix,
                                                              size=self.args['num_f'])

        if self.args['print_time']:
            print(f"Generated right hand in {format_time(time.time() - start_time)}.")

    def _generate_bc_values(self, lower_bound, upper_bound):
        start_time = time.time()

        self.bc_train = np.random.uniform(lower_bound, upper_bound, (self.args['num_bc'], 2))
        self.bc_valid = np.random.uniform(lower_bound, upper_bound, (self.args['num_bc'], 2))

        if self.args['print_time']:
            print(f"Generated boundary conditions in {format_time(time.time() - start_time)}.")

    def _generate_coefficients(self):
        start_time = time.time()

        kernel = 5.0 * Matern(length_scale=0.1, nu=4.5)
        x = np.linspace(0, self.args['L'], self.args['N'])[:, np.newaxis]
        cov_matrix = kernel(x)

        self.c_train = np.random.multivariate_normal(mean=np.zeros(len(x)), cov=cov_matrix,
                                                     size=self.args['num_c'])

        self.c_train = np.ones((self.args['num_c'], self.args['N']))
        self.c_valid = np.ones((self.args['num_c'], self.args['N']))

        random_factors_train = np.random.uniform(0.1, 5, (self.args['num_v']))
        random_factors_valid = np.random.uniform(0.1, 5, (self.args['num_v']))
        self.v_train = np.ones((self.args['num_v'], self.args['N'])) * random_factors_train[:, np.newaxis]
        self.v_valid = np.ones((self.args['num_v'], self.args['N'])) * random_factors_valid[:, np.newaxis]

        if self.args['print_time']:
            print(f"Generated coefficients in {format_time(time.time() - start_time)}.")

    def _match_dimensions(self):
        start_time = time.time()

        if self.args['method'] == 'descartes':
            ratio = read_config_file('runner_config.yml')['trainer']['split_ratio']
            train_size_f = int(self.args['num_bc'] * ratio)
            valid_size_f = int(self.args['num_bc'] - train_size_f)
            train_size_bc = int(self.args['num_f'] * ratio)
            valid_size_bc = int(self.args['num_f'] - train_size_bc)

            right_hand_train = np.repeat(self.right_hand_train, train_size_f, axis=0)
            right_hand_valid = np.repeat(self.right_hand_valid, valid_size_f, axis=0)
            bc_train = np.tile(self.bc_train, (train_size_bc, 1))
            bc_valid = np.tile(self.bc_valid, (valid_size_bc, 1))

            self.right_hand = np.concatenate((right_hand_train, right_hand_valid), axis=0)
            self.bc = np.concatenate((bc_train, bc_valid), axis=0)

            num_of_samples = self.right_hand.shape[0]
            train_size_c = int(num_of_samples * ratio)
            valid_size_c = num_of_samples - train_size_c
            train_size_v = int(num_of_samples * ratio)
            valid_size_v = num_of_samples - train_size_v

            c_train = np.repeat(self.c_train, train_size_c // self.args['num_c'] + 1, axis=0)[:train_size_c]
            c_valid = np.repeat(self.c_valid, valid_size_c // self.args['num_c'] + 1, axis=0)[:valid_size_c]
            v_train = np.repeat(self.v_train, train_size_v // self.args['num_v'] + 1, axis=0)[:train_size_v]
            v_valid = np.repeat(self.v_valid, valid_size_v // self.args['num_v'] + 1, axis=0)[:valid_size_v]

            self.c = np.concatenate((c_train, c_valid), axis=0)
            self.v = np.concatenate((v_train, v_valid), axis=0)
        else:
            raise ValueError('Invalid method')

        if self.args['print_time']:
            print(f"Matched dimensions in {format_time(time.time() - start_time)}.")
        return self.right_hand, self.bc, self.c, self.v

    def _plot_inputs(self, x, right_hand):
        plt.figure(figsize=(10, 6))
        iterations = min(right_hand.shape[0], 20)
        for i in range(iterations):
            plt.plot(x, right_hand[i], label=f'Sample {i + 1}')
        plt.xlabel('Position')
        plt.ylabel('Value')
        plt.title(f"Samples from Gaussian Process with Matern Kernel (nu={self.args['nu']})")
        plt.xlim([0, 1])
        plt.show()

    def _plot_cv(self, x, c, v):
        plt.figure(figsize=(10, 6))
        plt.plot(x, c[0], label='c')
        plt.plot(x, v[0], label='v')
        plt.xlabel('Position')
        plt.ylabel('Value')
        plt.title(f"Samples from Gaussian Process with Matern Kernel (nu={self.args['nu']})")
        plt.xlim([0, 1])
        plt.legend()
        plt.show()

    def _save_inputs(self, right_hand, bc, c, v):
        file_name = get_absolute_path(f'data/{self.group_id}/')
        Path(file_name).mkdir(parents=True, exist_ok=True)
        file_name = file_name + self.args['input_file_name'] + '_' + self.args['bc_type']
        np.savez(file_name, right_hand=right_hand, bc=bc, c=c, v=v)

    def _load_or_generate_outputs(self):
        if self.args['load']:
            data = np_load(f'data/{self.group_id}/' + self.args['output_file_name'], self.args['bc_type'] + '.npy')
            if data is not None:
                self.outputs = data
                return
        self._generate_outputs()

    def _generate_outputs(self):
        start_time = time.time()

        right_hand, bc_values, c_val, v_val = self.inputs
        N = right_hand.shape[1]
        lbc_type, rbc_type = self._get_bc_types()
        solutions = self._generate_solutions(right_hand, bc_values, c_val, v_val, N, lbc_type, rbc_type)

        if self.args['save']:
            self._save_outputs(solutions)

        if self.args['plot']:
            x = np.linspace(0, self.args['L'], N)
            self._plot_outputs(x, solutions, lbc_type, rbc_type, bc_values)

        self.outputs = np.array(solutions)

        if self.args['print_time']:
            print(f"Generated outputs in {format_time(time.time() - start_time)}.")

    def _generate_solutions(self, right_hand, bc_values, c_values, v_values, N, lbc_type, rbc_type):
        start_time = time.time()

        solutions = [
            FEM_hamiltonian_value(c, v, f, BC(lbc_type, bc[0]), BC(rbc_type, bc[1]), 0, self.args['L'], N)
            for f, bc, c, v in zip(right_hand, bc_values, c_values, v_values)]

        if self.args['print_time']:
            print(f"Generated {len(solutions)} solutions in {format_time(time.time() - start_time)}.")

        return solutions

    @staticmethod
    def _plot_outputs(x, solutions, lbc_type, rbc_type, bc_values):
        plt.figure(figsize=(10, 6))
        iterations = min(np.shape(solutions)[0], 50)
        for i in range(iterations):
            plt.plot(x, solutions[i], label=f'Solution {i + 1}')
        plt.xlabel('Position')
        plt.ylabel('Value')
        plt.title(f'FEM solutions (left: {lbc_type}, {bc_values[0, 0]:.2f}; right: {rbc_type}, {bc_values[0, 1]:.2f})')
        plt.xlim([0, 1])
        plt.show()

    def _save_outputs(self, solutions):
        file_name = get_absolute_path(f'data/{self.group_id}/')
        Path(file_name).mkdir(parents=True, exist_ok=True)
        file_name = file_name + self.args['output_file_name'] + '_' + self.args['bc_type']
        np.save(file_name, solutions)

    def _get_bc_types(self):
        bc_type_map = {
            'DD': (BCType.DIRICHLET, BCType.DIRICHLET),
            'DN': (BCType.DIRICHLET, BCType.NEUMANN),
            'ND': (BCType.NEUMANN, BCType.DIRICHLET),
            'NN': (BCType.NEUMANN, BCType.NEUMANN),
        }
        return bc_type_map.get(self.args['bc_type'], (BCType, BCType))


if __name__ == '__main__':
    set_seeds()
    data_gen = DataGenerator(bc_type='NN')
    input_data, output_data = data_gen.get_data()
