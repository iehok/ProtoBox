"""
This code is based on the repo of "Modeling Fine-Grained Entity Types 
with Box Embeddings" by Onoe et al., (2021).

  Paper: https://arxiv.org/pdf/2101.00345
  Code : https://github.com/yasumasaonoe/Box4Types

"""

from code.box.box_wrapper import log1mexp
from typing import Optional

import torch
import torch.nn as nn


class LinearProjection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True
    ):
        super(LinearProjection, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(inputs)
        return outputs


class SimpleFeedForwardLayer(nn.Module):
    """2-layer feed forward"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        activation: Optional[nn.Module] = None
    ):
        super(SimpleFeedForwardLayer, self).__init__()
        self.linear_projection1 = nn.Linear(input_dim, (input_dim + output_dim) // 2, bias=bias)
        self.linear_projection2 = nn.Linear((input_dim + output_dim) // 2, output_dim, bias=bias)
        self.activation = activation if activation else nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.activation(self.linear_projection1(inputs))
        inputs = self.activation(self.linear_projection2(inputs))
        return inputs


class HighwayNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int,
        activation: Optional[nn.Module] = None
    ):
        super(HighwayNetwork, self).__init__()
        self.n_layers = n_layers
        self.nonlinear = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        self.gate = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        for layer in self.gate:
            layer.bias = torch.nn.Parameter(0. * torch.ones_like(layer.bias))
        self.final_linear_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU() if activation is None else activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer_idx in range(self.n_layers):
            gate_values = self.sigmoid(self.gate[layer_idx](inputs))
            nonlinear = self.activation(self.nonlinear[layer_idx](inputs))
            inputs = gate_values * nonlinear + (1. - gate_values) * inputs
        return self.final_linear_layer(inputs)


class BCEWithLogProbLoss(nn.BCELoss):
    def _binary_cross_entropy(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Computes binary cross entropy.

        This function takes log probability and computes binary cross entropy.

        Args:
          input: Torch float tensor. Log probability. Same shape as `target`.
          target: Torch float tensor. Binary labels. Same shape as `input`.
          weight: Torch float tensor. Scaling loss if this is specified.
          reduction: Reduction method. 'mean' by default.
        """
        loss = -target * input - (1 - target) * log1mexp(input)

        if weight is not None:
            loss = loss * weight

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, input, target, weight=None, reduction='mean'):
        return self._binary_cross_entropy(
            input,
            target,
            weight=weight,
            reduction=reduction
        )
