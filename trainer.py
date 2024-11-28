import shutil
import signal
import time
from pathlib import Path

import wandb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from solvers.pinn.data import DataGenerator
from solvers.pinn.model import Network
from solvers.pinn.util import *


def plot_results(x, y, u, diff, f, discretization):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(x[:discretization], y[:discretization], 'g', label='Ground Truth')
    axs[0].plot(x[:discretization], u[:discretization], 'r', label='Prediction')
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('Position')
    axs[0].set_ylabel('Value')

    axs[1].plot(x[:discretization], diff[:discretization], 'b', label='Calculated equation value')
    axs[1].plot(x[:discretization], f[0, :], 'y', label='Right hand side of the equation')
    axs[1].legend(loc='upper right')
    axs[1].set_xlabel('Position')
    axs[1].set_ylabel('Value')

    plt.tight_layout()
    plt.show()


class Trainer:
    def __init__(self, bc_type='DD', args=read_config_file('runner_config.yml')['trainer'], group_id='coef',
                 run_id=wandb.util.generate_id(4)):
        self.args = args
        set_seeds(self.args['seed'])

        self.start_epoch = 0
        self.bc_type = bc_type
        self.group_id = group_id
        self.run_id = run_id

        data_args = read_config_file('runner_config.yml')['data']
        self.discretization = data_args['N']
        self.x = torch.linspace(0, 1, self.discretization)
        self.x.requires_grad = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_list = []
        self.best_test_loss = float('inf')
        self.best_model = None
        self.network = Network().to(self.device)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=float(self.args['lr']),
                                           weight_decay=float(self.args['weight_decay']))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                    factor=float(self.args['lr_sch_factor']),
                                                                    patience=int(self.args['lr_sch_patience']),
                                                                    threshold=float(self.args['lr_sch_threshold']))
        torch.autograd.set_detect_anomaly(True)

        if self.args['verbose']:
            self.network.print_parameter_count()

        signal.signal(signal.SIGINT, self._save_checkpoint_on_exit)
        self.run = None

        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )

    def _load_data(self):
        self.data_gen = DataGenerator(bc_type=self.bc_type, group_id=self.group_id)
        input_data, output_data = self.data_gen.get_data()
        right_hand, bc, c, v = input_data
        input_data = np.concatenate((right_hand, bc, c, v), axis=1)

        input_data_train, input_data_validate, output_data_train, output_data_validate = self.split_data(input_data,
                                                                                                         output_data,
                                                                                                         ratio=
                                                                                                         self.args[
                                                                                                             'split_ratio'])

        train_dataset = TensorDataset(torch.Tensor(input_data_train), torch.Tensor(output_data_train))
        validation_dataset = TensorDataset(torch.Tensor(input_data_validate), torch.Tensor(output_data_validate))

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True)
        self.validation_dataloader = DataLoader(validation_dataset, batch_size=self.args['batch_size'])

    @staticmethod
    def split_data(input_data, output_data, ratio):
        input_data_train, input_data_validate, output_data_train, output_data_validate = train_test_split(input_data,
                                                                                                          output_data,
                                                                                                          train_size=ratio)
        return input_data_train, input_data_validate, output_data_train, output_data_validate

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if self.args['verbose']:
            print(f"Loaded checkpoint from {checkpoint_path}", flush=True)
        self.start_epoch = checkpoint['epoch']
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.loss_list = checkpoint['loss']
        self.best_test_loss = checkpoint['best_test_loss']

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

    def net_u(self, f, x, lbc, rbc, c, v):
        return self.network(f, x, lbc, rbc, c, v)

    def net_diff(self, f, x, lbc, rbc, c, v):
        u = self.net_u(f, x, lbc, rbc, c, v)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

        return u_x, -c[::128].reshape(-1) * u_xx + v[::128].reshape(-1) * u

    def loss(self, f, x, lbc, rbc, c, v, y):
        u = self.net_u(f, x, lbc, rbc, c, v)
        u_x, diff = self.net_diff(f, x, lbc, rbc, c, v)

        loss_u = nn.MSELoss()(u, y)
        loss_f = nn.MSELoss()(diff, f[0:-1:self.discretization, :].reshape(-1))

        loss_bc = 0
        if self.bc_type == 'DD':
            loss_bc = nn.MSELoss()(u[::self.discretization], lbc[::self.discretization]) + nn.MSELoss()(
                u[self.discretization - 1::self.discretization], rbc[::self.discretization])
        elif self.bc_type == 'DN':
            loss_bc = nn.MSELoss()(u[::self.discretization], lbc[::self.discretization]) + nn.MSELoss()(
                u_x[self.discretization - 1::self.discretization], rbc[::self.discretization])
        elif self.bc_type == 'ND':
            loss_bc = nn.MSELoss()(u_x[::self.discretization], lbc[::self.discretization]) + nn.MSELoss()(
                u[self.discretization - 1::self.discretization], rbc[::self.discretization])
        elif self.bc_type == 'NN':
            loss_bc = nn.MSELoss()(u_x[::self.discretization], lbc[::self.discretization]) + nn.MSELoss()(
                u_x[self.discretization - 1::self.discretization], rbc[::self.discretization])

        loss = float(self.args['alpha']) * loss_u + float(self.args['beta']) * loss_f + float(
            self.args['gamma']) * loss_bc

        return loss, loss_u, loss_f, loss_bc, u, diff

    def train(self):
        self.run = wandb.init(project='pinn', config=read_config_file('runner_config.yml'),
                              id=self.run_id, resume='allow',
                              group=self.group_id, job_type=self.bc_type)
        if wandb.run.resumed:
            checkpoint_path = get_absolute_path('model/' + self.group_id + '/checkpoint', self.bc_type + '.pth')
            if os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)

        self._load_data()

        start_time = time.time()

        with self.profiler:
            for epoch in range(self.start_epoch, self.args['max_epochs']):
                if self.args['print_memory']:
                    print_memory_usage()

                loss, loss_u, loss_f, loss_bc = self._train_epoch()
                validation_loss, validation_loss_u, validation_loss_f, validation_loss_bc = self._validation_epoch()

                self._print_epoch_results(epoch, loss, validation_loss, validation_loss_u, validation_loss_f,
                                          validation_loss_bc, start_time)
                self.loss_list.append(validation_loss)
                self.start_epoch = epoch
                self.scheduler.step(validation_loss)

                if validation_loss < self.best_test_loss:
                    self.best_test_loss = validation_loss
                    self.best_model = self.network
                if epoch % int(self.args['save_every']):
                    self._save_checkpoint()

                wandb.log({'train': {'loss': loss, 'loss_u': loss_u, 'loss_f': loss_f, 'loss_bc': loss_bc},
                           'validation': {'loss': validation_loss, 'loss_u': validation_loss_u,
                                          'loss_f': validation_loss_f,
                                          'loss_bc': validation_loss_bc},
                           'lr': self.optimizer.param_groups[0]['lr'], 'best_val_loss': self.best_test_loss},
                          step=epoch + 1
                          )

                if self.optimizer.param_groups[0]['lr'] <= 1e-7:
                    self._save_final_model()
                    break

        print(f"Training finished. Elapsed time: {format_time(time.time() - start_time)}", flush=True)

    def _train_epoch(self):
        self.network.train()
        losses = []
        losses_u = []
        losses_f = []
        losses_bc = []

        for batch_nr, (X, y) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            x, f, lbc, rbc, c, v, y = self._format_data(X, y)
            x, f, lbc, rbc, c, v, y = self._move_to_device(x, f, lbc, rbc, c, v, y)

            loss, loss_u, loss_f, loss_bc, u, diff = self.loss(f, x, lbc, rbc, c, v, y)

            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            losses_u.append(loss_u.item())
            losses_f.append(loss_f.item())
            losses_bc.append(loss_bc.item())

            if self.profiler:
                self.profiler.step()

        return np.mean(losses), np.mean(losses_u), np.mean(losses_f), np.mean(losses_bc)

    def _validation_epoch(self):
        self.network.eval()
        losses = []
        losses_u = []
        losses_f = []
        losses_bc = []

        for batch_nr, (X, y) in enumerate(self.validation_dataloader):
            x, f, lbc, rbc, c, v, y = self._format_data(X, y)
            x, f, lbc, rbc, c, v, y = self._move_to_device(x, f, lbc, rbc, c, v, y)

            loss, loss_u, loss_f, loss_bc, u, diff = self.loss(f, x, lbc, rbc, c, v, y)
            loss = loss.cpu().detach().numpy()

            losses.append(loss.item())
            losses_u.append(loss_u.item())
            losses_f.append(loss_f.item())
            losses_bc.append(loss_bc.item())

            self.optimizer.zero_grad()

            x = x.cpu().detach()
            f = f.cpu().detach()
            y = y.cpu().detach()
            u = u.cpu().detach()
            diff = diff.cpu().detach()

            if self.args['plot'] and batch_nr % 500 == 0:
                plot_results(x, y, u, diff, f, self.discretization)

        return np.mean(losses), np.mean(losses_u), np.mean(losses_f), np.mean(losses_bc)

    def _print_epoch_results(self, epoch, loss, validation_loss, validation_loss_u, validation_loss_f,
                             validation_loss_bc, start_time):
        if self.args['verbose']:
            print(
                f"{self.bc_type} | E{epoch + 1:04d} | LR: {self.optimizer.param_groups[0]['lr']:.0e} | TrL: {loss:.6f}; VaL: {validation_loss:.6f} (Lu: {validation_loss_u:.6f}, Lf: {validation_loss_f:.6f}, Lbc: {validation_loss_bc:.6f}) | Time: {format_time(time.time() - start_time)}",
                flush=True)

    def _move_to_device(self, x, f, lbc, rbc, c, v, y):
        x = x.to(self.device)
        f = f.to(self.device)
        lbc = lbc.to(self.device)
        rbc = rbc.to(self.device)
        c = c.to(self.device)
        v = v.to(self.device)
        y = y.to(self.device)
        return x, f, lbc, rbc, c, v, y

    def _save_checkpoint_on_exit(self, signal, frame):
        self._save_checkpoint()
        wandb.finish()
        print("Training interrupted. Checkpoint saved. Run finished", flush=True)
        exit(0)

    def _save_checkpoint(self):
        checkpoint_path = get_absolute_path(f'model/{self.group_id}')
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_path + '/checkpoint_' + self.bc_type + '.pth'
        torch.save({
            'epoch': self.start_epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.loss_list,
            'best_test_loss': self.best_test_loss
        }, checkpoint_path)
        shutil.copy2(checkpoint_path, wandb.run.dir)
        if not self.run:
            wandb.save(checkpoint_path)

    def _save_final_model(self):
        if self.best_model is not None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'model/{self.group_id}/')
            Path(model_path).mkdir(parents=True, exist_ok=True)
            model_path = model_path + 'model_' + str(self.bc_type) + '.pth'
            torch.save(self.best_model.state_dict(), model_path)
            shutil.copy2(model_path, wandb.run.dir)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    wandb.finish()
