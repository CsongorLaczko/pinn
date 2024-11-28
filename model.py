import torch
from torch import nn

from solvers.pinn.util import read_config_file


class Network(nn.Module):
    def __init__(self, config_file='runner_config.yml'):
        super().__init__()
        config = read_config_file(config_file)
        layers = config['model']['layers']
        input_dim = config['data']['N'] * 3 + 3
        self.linears = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip([input_dim, *layers], layers)])
        self.linears.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)

        # Print whole architecture
        # print("Architecture:")
        # print(self.linears)

    def forward(self, f, x, lbc, rbc, c, v):
        # print("Starting forward pass")

        x = x.unsqueeze(1)
        lbc = lbc.unsqueeze(1)
        rbc = rbc.unsqueeze(1)

        X = torch.concatenate((x, lbc, rbc, f, c, v), 1)

        for ix in range(len(self.linears) - 1):
            # print("Input shape: ", X.shape)
            X = nn.Tanh()(self.linears[ix](X))
            # print("Output shape: ", X.shape)

        # print("Input shape: ", X.shape)
        X = self.linears[-1](X)
        # print("Output shape: ", X.shape)
        return X.reshape(-1)

    def print_parameter_count(self):
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}",
              flush=True)


class Network_without_coeff(nn.Module):
    def __init__(self, config_file='runner_config.yml'):
        super(Network_without_coeff, self).__init__()
        lifts = read_config_file(config_file)['model']['layers']
        dim = read_config_file(config_file)['data']['N']
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip([dim+3, *lifts], lifts)])
        self.linears.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)

    def forward(self, f, x, lbc, rbc):
        x = x.unsqueeze(1)
        lbc = lbc.unsqueeze(1)
        rbc = rbc.unsqueeze(1)

        x = torch.concatenate((x, lbc, rbc, f), 1)

        for ix in range(len(self.linears) - 1):
            x = nn.Tanh()(self.linears[ix](x))

        x = self.linears[-1](x)
        return x.reshape(-1)

    def print_parameter_count(self):
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}",
              flush=True)
