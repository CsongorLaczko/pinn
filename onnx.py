from doctest import debug

import numpy as np
import onnx
import onnxruntime as ort
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


def export_onnx_models(exporter, bc_type, f, x, lbc, rbc, c, v):
    torch.onnx.export(
        exporter.models[bc_type],  # Model being exported
        (f, x, lbc, rbc, c, v),  # Model input (or a tuple for multiple inputs)
        f"model_{bc_type}.onnx",  # File name for the exported model
        export_params=True,  # Store trained parameter weights inside the model file
        opset_version=12,  # Specify the ONNX version to export to (12 is commonly used)
        do_constant_folding=True,  # Optimize by executing constant folding
        input_names=['f', 'x', 'lbc', 'rbc', 'c', 'v'],  # The model's input names
        output_names=['y'],  # The model's output names
        dynamic_axes={
            'f': {0: 'b'},  # 'b' is dynamic for the first dimension
            'x': {0: 'k'},  # 'k' is dynamic for the first dimension
            'lbc': {0: 'b'},  # 'b' is dynamic for the first dimension
            'rbc': {0: 'b'},  # 'b' is dynamic for the first dimension
            'c': {0: 'b'},  # 'b' is dynamic for the first dimension
            'v': {0: 'b'},  # 'b' is dynamic for the first dimension
            'y': {0: 'b'}  # 'b' is dynamic for the first dimension
        }
    )
    print(f"Exported model {bc_type} to ONNX format.")


def load_and_test_onnx_model(model_path, input_data):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print(f"Loaded and checked model: {model_path}")

    ort_session = ort.InferenceSession(model_path)

    input_names = [input.name for input in ort_session.get_inputs()]
    # inputs = {name: data for name, data in zip(input_names, input_data)}
    inputs = {name: data.cpu().numpy() for name, data in zip(input_names, input_data)}
    outputs = ort_session.run(None, inputs)

    return outputs[0]


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

    for bc_type in exporter.bc_types:
        # Get the output from the original PyTorch model
        with torch.no_grad():
            original_output = exporter.models[bc_type](f, x, lbc, rbc, c, v).cpu().numpy()

        # Export the model to ONNX
        export_onnx_models(exporter, bc_type, f, x, lbc, rbc, c, v)

        # Load and test the ONNX model
        onnx_output = load_and_test_onnx_model(f"model_{bc_type}.onnx", (f, x, lbc, rbc, c, v))

        # Compare the outputs
        if np.allclose(original_output, onnx_output, atol=1e-5):
            print(f"Model {bc_type} outputs are the same before and after export/import.")
        else:
            print(f"Model {bc_type} outputs differ before and after export/import.")
            breakpoint()
