import torch

from solvers.pinn.model import Network
from solvers.pinn.util import get_absolute_path, read_config_file

k = 2
N = 128
b = 100


def format_data(f, x, lbc, rbc, c, v):
    x = x.repeat(b)
    lbc = lbc.unsqueeze(1).expand(-1, k).reshape(-1)
    rbc = rbc.unsqueeze(1).expand(-1, k).reshape(-1)
    f = f.unsqueeze(1).expand(-1, k, -1).reshape(-1, f.shape[1])
    c = c.unsqueeze(1).expand(-1, k, -1).reshape(-1, c.shape[1])
    v = v.unsqueeze(1).expand(-1, k, -1).reshape(-1, v.shape[1])

    return f, x, lbc, rbc, c, v


class Exporter:
    def __init__(self, args=read_config_file('runner_config.yml'), group_id='coef-50'):
        self.args = args
        self.group_id = group_id
        self.bc_types = ['DD', 'DN', 'ND', 'NN']
        self.epoch_nums = {}

        data_args = args['data']
        self.discretization = data_args['N']
        self.x = torch.linspace(0, 1, self.discretization)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        print('Loaded {} models'.format(len(self.models)))

    def move_to_device(self, f, x, lbc, rbc, c, v):
        x = x.to(self.device)
        f = f.to(self.device)
        lbc = lbc.to(self.device)
        rbc = rbc.to(self.device)
        c = c.to(self.device)
        v = v.to(self.device)
        return f, x, lbc, rbc, c, v


if __name__ == "__main__":
    exporter = Exporter()

    f = torch.randn(b, N)
    x = torch.randn(k)
    lbc = torch.randn(b)
    rbc = torch.randn(b)
    c = torch.randn(b, N)
    v = torch.randn(b, N)

    f, x, lbc, rbc, c, v = format_data(f, x, lbc, rbc, c, v)
    f, x, lbc, rbc, c, v = exporter.move_to_device(f, x, lbc, rbc, c, v)

    example_input = (f, x, lbc, rbc, c, v)
    for bc_type in exporter.bc_types:
        traced_model = torch.jit.trace(exporter.models[bc_type], example_input)
        traced_model.save(f"model_{bc_type}.pt")
