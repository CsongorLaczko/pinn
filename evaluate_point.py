import time

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

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
        self.eval_points = 4
        self.x = torch.linspace(0, 1, self.eval_points)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_data()
        self._load_models()

    def _load_models(self):
        checkpoints = {
            bc_type: torch.load(get_absolute_path('model/' + self.group_id + '/checkpoint', bc_type + '.pth')) for
            bc_type in
            self.bc_types}
        self.models = {bc_type: Network().to(self.device) for bc_type in self.bc_types}
        for bc_type in self.bc_types:
            self.models[bc_type].load_state_dict(checkpoints[bc_type]['model_state_dict'])
            self.models[bc_type].eval()
            self.epoch_nums[bc_type] = checkpoints[bc_type]['epoch']

    def _load_data(self):
        folder = self.group_id
        input_data = {bc_type: np.load(get_absolute_path('data/' + folder + '/input', bc_type + '.npz')) for bc_type in
                      self.bc_types}
        output_data = {bc_type: np.load(get_absolute_path('data/' + folder + '/output', bc_type + '.npy')) for bc_type
                       in
                       self.bc_types}

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

        # f, lbc, rbc = X[:, :-2], X[:, -2], X[:, -1]
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
        x = x.to(self.device)
        f = f.to(self.device)
        lbc = lbc.to(self.device)
        rbc = rbc.to(self.device)
        c = c.to(self.device)
        v = v.to(self.device)
        y = y.to(self.device)
        return x, f, lbc, rbc, c, v, y

    def evaluate(self):
        wandb.init(project='pinn', config=read_config_file('runner_config.yml'),
                   name=f"Eval_{wandb.util.generate_id(4)}",
                   group=self.group_id, job_type='evaluation',
                   mode='disabled')

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        start_time = time.time()
        for bc_type in self.bc_types:
            prediction = []
            truth = []

            for batch_nr, (X, y) in enumerate(self.validation_dataloader[bc_type]):
                x, f, lbc, rbc, c, v, y = self._format_data(X, y)
                x, f, lbc, rbc, c, v, y = self._move_to_device(x, f, lbc, rbc, c, v, y)

                prediction.append(self.models[bc_type](f, x, lbc, rbc, c, v).detach().cpu().numpy())

                y_cpu = y.detach().cpu().numpy()
                total_elements = y_cpu.shape[0]
                step_size = 128
                num_elements = total_elements // step_size

                # Create an index array
                indices = torch.arange(0, total_elements, step_size).repeat_interleave(2) + torch.tensor([0, 1]).repeat(
                    num_elements)

                # Index the y tensor
                y_indexed = y_cpu[indices]

                truth.append(y_indexed)

            self.prediction[bc_type] = np.concatenate(prediction, axis=0)
            self.truth[bc_type] = np.concatenate(truth, axis=0)
            # self.error[bc_type] = torch.nn.MSELoss()(torch.Tensor(self.prediction[bc_type]),
            #                                          torch.Tensor(self.truth[bc_type]))

        print(f"Time taken: {time.time() - start_time}")

        for ax, bc_type in zip(axs.flatten(), self.bc_types):
            # ax.set_title(f'{bc_type} - MSE: {self.error[bc_type]:.6f}, Epoch: {self.epoch_nums[bc_type]}')
            ax.set_xlabel('x')
            ax.set_ylabel('u(x)')
            for j in range(min(20, len(self.truth[bc_type]) // self.eval_points)):
                ax.plot(self.x.cpu(), self.prediction[bc_type][self.eval_points * j:self.eval_points * (j + 1)],
                        'r',
                        label='Prediction')

        plt.tight_layout()
        plt.show()

        wandb.log({"img": [wandb.Image(fig, caption="Prediction")]})
        wandb.finish()


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate()
