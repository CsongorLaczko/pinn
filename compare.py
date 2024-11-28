import cProfile
import time

import torch
from torch.utils.data import TensorDataset, DataLoader

from solvers.fem.fem import *
from solvers.pinn.model import Network
from solvers.pinn.trainer import Trainer
from solvers.pinn.util import read_config_file, get_absolute_path


class Evaluator:
    def __init__(self, args=read_config_file('runner_config.yml'), group_id='coef-50'):
        self.args = args
        self.group_id = group_id
        self.bc_types = ['DD', 'DN', 'ND', 'NN']
        self.epoch_nums = {}

        self.prediction = {}
        self.truth = {}
        self.error = {}
        self.validation_dataloader = {}

        data_args = args['data']
        self.discretization = data_args['N']
        self.x = torch.linspace(0, 1, self.discretization)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_data()
        self._load_models()

    def _load_models(self):
        start_time = time.perf_counter()
        checkpoints = {
            bc_type: torch.load(get_absolute_path('model/' + self.group_id + '/checkpoint', bc_type + '.pth')) for
            bc_type in self.bc_types}
        self.models = {bc_type: Network().to(self.device) for bc_type in self.bc_types}
        for bc_type in self.bc_types:
            self.models[bc_type].load_state_dict(checkpoints[bc_type]['model_state_dict'])
            self.models[bc_type].eval()
            self.epoch_nums[bc_type] = checkpoints[bc_type]['epoch']
        self.model_load_time = time.perf_counter() - start_time

    def _load_data(self):
        folder = self.group_id
        input_data = {bc_type: np.load(get_absolute_path('data/' + folder + '/input', bc_type + '.npz')) for bc_type in
                      self.bc_types}
        output_data = {bc_type: np.load(get_absolute_path('data/' + folder + '/output', bc_type + '.npy')) for bc_type
                       in self.bc_types}

        right_hand = {bc_type: input_data[bc_type]['right_hand'] for bc_type in self.bc_types}
        bc = {bc_type: input_data[bc_type]['bc'] for bc_type in self.bc_types}
        c = {bc_type: input_data[bc_type]['c'] for bc_type in self.bc_types}
        v = {bc_type: input_data[bc_type]['v'] for bc_type in self.bc_types}

        input_data = {bc_type: np.concatenate((right_hand[bc_type], bc[bc_type], c[bc_type], v[bc_type]), axis=1) for
                      bc_type in self.bc_types}

        input_data_validate = {}
        output_data_validate = {}
        validation_dataset = {}

        for bc_type in self.bc_types:
            _, input_data_validate[bc_type], _, output_data_validate[bc_type] = Trainer.split_data(input_data[bc_type],
                                                                                                   output_data[bc_type],
                                                                                                   ratio=
                                                                                                   self.args['trainer'][
                                                                                                       'split_ratio'])
            validation_dataset[bc_type] = TensorDataset(torch.Tensor(input_data_validate[bc_type]),
                                                        torch.Tensor(output_data_validate[bc_type]))
            self.validation_dataloader[bc_type] = DataLoader(validation_dataset[bc_type],
                                                             batch_size=self.args['trainer']['batch_size'])

    def _format_data(self, X, y):
        X = X.squeeze()
        y = y.squeeze()

        N = self.discretization
        f, lbc, rbc, c, v = X[:, 0:N], X[:, N], X[:, N + 1], X[:, N + 2: 2 * N + 2], X[:, 2 * N + 2:]

        k = self.x.shape[0]
        b = f.shape[0]

        x = self.x.repeat(b)
        lbc = lbc.unsqueeze(1).expand(-1, k).reshape(-1)
        rbc = rbc.unsqueeze(1).expand(-1, k).reshape(-1)
        f = f.unsqueeze(1).expand(-1, k, -1).reshape(-1, f.shape[1])
        c = c.unsqueeze(1).expand(-1, k, -1).reshape(-1, c.shape[1])
        v = v.unsqueeze(1).expand(-1, k, -1).reshape(-1, v.shape[1])
        y = y.reshape(-1)

        return x, f, lbc, rbc, c, v, y

    def _move_to_device(self, x, f, lbc, rbc, c, v, y):
        start_time = time.perf_counter()
        x = x.to(self.device)
        f = f.to(self.device)
        lbc = lbc.to(self.device)
        rbc = rbc.to(self.device)
        c = c.to(self.device)
        v = v.to(self.device)
        y = y.to(self.device)
        move_time = time.perf_counter() - start_time
        return x, f, lbc, rbc, c, v, y, move_time

    def evaluate(self):
        for bc_type in self.bc_types:
            pinn_solution = []
            fem_solution = []

            pinn_time = 0
            fem_time = 0
            move_time_total = 0
            format_time_total = 0

            for batch_nr, (X, y) in enumerate(self.validation_dataloader[bc_type]):
                N = self.discretization
                f, lbc, rbc, c, v = X[:, 0:N], X[:, N], X[:, N + 1], X[:, N + 2: 2 * N + 2], X[:, 2 * N + 2:]
                f, lbc, rbc, c, v = f.numpy(), lbc.numpy(), rbc.numpy(), c.numpy(), v.numpy()

                lbc_type = None
                rbc_type = None

                if bc_type == 'DD':
                    lbc_type = BCType.DIRICHLET
                    rbc_type = BCType.DIRICHLET
                elif bc_type == 'DN':
                    lbc_type = BCType.DIRICHLET
                    rbc_type = BCType.NEUMANN
                elif bc_type == 'ND':
                    lbc_type = BCType.NEUMANN
                    rbc_type = BCType.DIRICHLET

                start_time = time.perf_counter()
                fem_solution.append(
                    [FEM_hamiltonian_value(c, v, f, BC(lbc_type, lbc), BC(rbc_type, rbc), 0, 1, self.discretization)
                     for f, lbc, rbc, c, v in zip(f, lbc, rbc, c, v)]
                )
                fem_time += time.perf_counter() - start_time

                start_time = time.perf_counter()
                x, f, lbc, rbc, c, v, y = self._format_data(X, y)
                format_time_total += time.perf_counter() - start_time

                x, f, lbc, rbc, c, v, y, move_time = self._move_to_device(x, f, lbc, rbc, c, v, y)
                move_time_total += move_time

                start_time = time.perf_counter()
                pinn_solution.append(self.models[bc_type](f, x, lbc, rbc, c, v).detach().cpu().numpy())
                pinn_time += time.perf_counter() - start_time

            self.prediction[bc_type] = np.concatenate(pinn_solution, axis=0)
            total_pinn_time = move_time_total + format_time_total + pinn_time
            total_pinn_time_with_load = self.model_load_time + total_pinn_time

            print(f"BC Type: {bc_type}, Move to Device Time: {move_time_total:.8f} seconds, "
                  f"Format Data Time: {format_time_total:.8f} seconds, "
                  f"PINN Inference Time: {pinn_time:.8f} seconds, FEM Time: {fem_time:.8f} seconds, "
                  f"Total PINN Time: {total_pinn_time:.8f} seconds, "
                  f"Total PINN Time with Model Load: {total_pinn_time_with_load:.8f} seconds")

        print(f"Model Load Time: {self.model_load_time:.8f} seconds")


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats('profile_results.prof')