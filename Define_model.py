import torch
import torch.nn as nn
import torch.optim as optim
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import scipy.io
import numpy as np    # Define a simple coupling block for the normalizing flow


class CouplingLayer(Fm.GLOWCouplingBlock):
    def __init__(self, dims_in, dims_c=[]):
        super().__init__(dims_in, dims_c, subnet_constructor=self.subnet)

    @staticmethod
    def subnet(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, dims_out)
        )

# Create the coupling flow architecture
def create_flow(input_dim):
    nodes = [Ff.InputNode(input_dim, name='input')]

    for i in range(5):  # 5 coupling layers
        nodes.append(Ff.Node([nodes[-1].out0], CouplingLayer, {}, name=f'coupling_{i}'))

    nodes.append(Ff.OutputNode([nodes[-1].out0], name='output'))
    return Ff.ReversibleGraphNet(nodes)

# Koopman model with invertible flow as encoder
class KoopmanModel(nn.Module):
    def __init__(self, input_dim, control_dim):
        super(KoopmanModel, self).__init__()
        self.flow = create_flow(input_dim)  # Invertible encoder using normalizing flow
        self.Koopman = matrix = torch.empty(input_dim, input_dim+control_dim)
        nn.init.xavier_uniform_(self.Koopman)
        #self.Koopman = nn.Parameter(torch.eye(koopman_dim))  # Koopman matrix

    def encode(self, x):
        """ Encode system state x into Koopman observables (invertible) """
        return self.flow(x)[0]

    def decode(self, phi):
        """ Decode Koopman observables back to original state (invertible) """
        return self.flow(phi, rev=True)[0]

    def join(self, phi_x, u):
        return torch.cat((phi_x, u), dim=1)

    def forward(self, x, u):
        """ Forward pass for Koopman dynamics """
        phi_x = self.encode(x)  # Encode state to Koopman observables
        phi_x_u = self.join(phi_x, u)
        phi_x_next = torch.matmul(self.Koopman, phi_x_u.T).T  # Linear Koopman dynamics
        return phi_x, phi_x_next