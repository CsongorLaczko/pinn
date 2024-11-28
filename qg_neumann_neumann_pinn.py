import os
import torch
from solvers.interfaces.qg_neumann_neumann import *
from solvers.pinn.model import Network_without_coeff
from solvers.fem.fem import *


class QGNNPINNEdge(QGEdge):
    models = {}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for type in ['DD', 'DN', 'ND', 'NN']:
        file_name = os.path.join(script_dir, 'model/coef-50/checkpoint_' + type + '.pth')
        models[type] = Network_without_coeff()
        models[type].load_state_dict(torch.load(file_name)['model_state_dict'])
        models[type].eval()
        models[type].to('cuda')

    def __init__(self, out_, in_, x, h, c_value, f_value, A, lbc, rbc, original_N):
        self.out_ = out_
        self.in_ = in_
        self.x = x
        self.h = h
        self.lbc = lbc
        self.rbc = rbc
        self.lweight = 1
        self.rweight = 1
        self.c_value = c_value
        self.f_value = f_value
        self.A = A
        self.Al = [self.A[1][0], self.A[0][1]]
        self.Ar = [self.A[2][-2], self.A[1][-1]]
        self.A = apply_bcs_to_stiffness_matrix(self.A, self.lbc, self.rbc)
        self.b = load_vector_value(self.f_value, self.h, self.lbc, self.rbc)
        self.u = np.zeros((4,))
        self.original_N = original_N

        if self.lbc.type == BCType.DIRICHLET:
            if self.rbc.type == BCType.DIRICHLET:
                self.type = 'DD'
            else:
                self.type = 'DN'
        else:
            if self.rbc.type == BCType.DIRICHLET:
                self.type = 'ND'
            else:
                self.type = 'NN'

    def solve(self):
        with torch.no_grad():
            x = torch.Tensor(self.x)
            x = torch.Tensor([x[0], x[1], x[-2], x[-1]]).to('cuda')
            f = torch.Tensor(self.f_value).to('cuda')
            lbc = torch.Tensor([self.lbc.value]).to('cuda')
            rbc = torch.Tensor([self.rbc.value]).to('cuda')
            # c = torch.Tensor(self.c_value).to('cuda')
            # v = torch.Tensor(self.v_value).to('cuda')

            f = f.unsqueeze(0)
            # c = c.unsqueeze(0)
            # v = v.unsqueeze(0)

            k = x.shape[0]
            b = f.shape[0]

            x = x.repeat(b)
            lbc = lbc.unsqueeze(1).expand(-1, k).reshape(-1)
            rbc = rbc.unsqueeze(1).expand(-1, k).reshape(-1)

            f = f.unsqueeze(1).expand(-1, k, -1).reshape(-1, f.shape[1])
            # c = c.unsqueeze(1).expand(-1, k, -1).reshape(-1, c.shape[1])
            # v = v.unsqueeze(1).expand(-1, k, -1).reshape(-1, v.shape[1])

            # self.u = QGNNPINNEdge.models[self.type](f, x, lbc, rbc, c, v).cpu().detach().numpy().flatten()
            self.u = QGNNPINNEdge.models[self.type](f, x, lbc, rbc).cpu().detach().numpy().flatten()

    def solve_full(self):
        with torch.no_grad():
            x = torch.Tensor(torch.linspace(0, 1, self.original_N+1).to('cuda'))
            # x = torch.Tensor(self.x).to('cuda')
            f = torch.Tensor(self.f_value).to('cuda')
            lbc = torch.Tensor([self.lbc.value]).to('cuda')
            rbc = torch.Tensor([self.rbc.value]).to('cuda')
            # c = torch.Tensor(self.c_value).to('cuda')
            # v = torch.Tensor(self.v_value).to('cuda')

            f = f.unsqueeze(0)
            # c = c.unsqueeze(0)
            # v = v.unsqueeze(0)

            k = x.shape[0]
            b = f.shape[0]

            x = x.repeat(b)
            lbc = lbc.unsqueeze(1).expand(-1, k).reshape(-1)
            rbc = rbc.unsqueeze(1).expand(-1, k).reshape(-1)

            f = f.unsqueeze(1).expand(-1, k, -1).reshape(-1, f.shape[1])
            # c = c.unsqueeze(1).expand(-1, k, -1).reshape(-1, c.shape[1])
            # v = v.unsqueeze(1).expand(-1, k, -1).reshape(-1, v.shape[1])

            # self.u = QGNNPINNEdge.models[self.type](f, x, lbc, rbc, c, v).cpu().detach().numpy().flatten()
            self.u = QGNNPINNEdge.models[self.type](f, x, lbc, rbc).cpu().detach().numpy().flatten()
            self.x = torch.linspace(0, 1, self.original_N+1)

    def update_bc_values(self, lbc_value, rbc_value):
        self.lbc.value = lbc_value / self.lweight
        self.rbc.value = rbc_value / self.rweight
        apply_bcs_to_load_vector(self.b, self.f_value, self.h, self.lbc, self.rbc)

    def set_weights(self, lweight, rweight):
        self.lweight = lweight
        self.rweight = rweight

    def approx_Neumann(self, endpoint):
        if endpoint == self.out_:
            return -(np.dot(self.Al, self.u[:2]) - self.h * self.f_value[0] / 2)
        else:
            return -(np.dot(self.Ar, self.u[-2:]) - self.h * self.f_value[-1] / 2)


class QGNeumannNeumannPINN(QGNeumannNeumann):
    def __init__(self, adj, bcs, N, change_threshold, maxiter, theta, use_weights, adaptive):
        self.N = 127
        self.original_N = N
        self.x0 = 0
        self.xend = 1
        self.x = np.linspace(self.x0, self.xend, self.N + 1)
        self.h = (self.xend - self.x0) / self.N
        self.adj = np.array(adj)
        self.bcs = bcs
        self.change_threshold = change_threshold / N
        self.maxiter = maxiter
        self.theta = theta
        self.adaptive = adaptive
        self.vertex_values = np.zeros(len(self.adj))
        self.dirichlet_edges, self.neumann_edges, self.vertex2edge = self._generate_edges()
        if use_weights:
            self.vertex_weights = self._degrees()
        else:
            self.vertex_weights = np.ones(len(self.adj))
        self._set_bc_weights()

    def _new_edges(self, out_, in_, lbc, rbc):
        params = self.adj[out_][in_]
        c_value = np.vectorize(params['c'])(self.x)
        v_value = np.vectorize(params['v'])(self.x)
        f_value = np.vectorize(params['f'])(self.x)
        A = stiffness_matrix_hamiltonian_value(c_value, v_value, self.h, lbc, rbc)
        dirichlet_edge = QGNNPINNEdge(out_, in_, self.x, self.h, c_value, f_value, A, lbc, rbc, self.original_N)
        neumann_edge = QGNNPINNEdge(out_, in_, self.x, self.h, c_value, np.zeros_like(self.x), A, BC(BCType.NEUMANN),
                                    BC(BCType.NEUMANN), self.original_N)
        return dirichlet_edge, neumann_edge

    def solve(self):
        iter = super().solve()
        for edge in self.dirichlet_edges:
            edge.solve_full()
        return iter
