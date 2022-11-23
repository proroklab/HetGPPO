#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
from torch import Tensor


def clamp_with_norm(tensor: Tensor, max_norm: float):
    norm = torch.linalg.vector_norm(tensor, dim=-1)
    new_tensor = (tensor / norm.unsqueeze(-1)) * max_norm
    tensor[norm > max_norm] = new_tensor[norm > max_norm]
    return tensor


class TorchDiagGaussian:
    """Wrapper class for PyTorch Normal distribution."""

    def __init__(
        self,
        inputs,
        u_range,
    ):
        self.inputs = inputs
        mean, log_std = torch.chunk(self.inputs, 2, dim=1)
        self.dist = torch.distributions.normal.Normal(mean, torch.exp(log_std))
        self.u_range = u_range

    def sample(self):
        sample = self.dist.sample()
        return self._squash(sample)

    def deterministic_sample(self):
        sample = self.dist.mean
        return self._squash(sample)

    def _squash(self, sample):
        sample = -self.u_range + (sample + 1.0) * (2 * self.u_range) / 2.0
        return sample.clamp(-self.u_range, self.u_range)


def compute_action_corridor(
    pos0_x: float,
    pos0_y: float,
    vel0_x: float,
    vel0_y: float,
    pos1_x: float,
    pos1_y: float,
    vel1_x: float,
    vel1_y: float,
    model,
    u_range: float,
    n_agents: int = 2,
    obs_size: int = 4,
    num_actions_per_agent: int = 2,
    deterministic: bool = True,
    circular_action_constraint: bool = True,
):
    observation = torch.zeros((1, n_agents, obs_size))
    agent_input = [[vel0_x, vel0_y, pos0_x, pos0_y], [vel1_x, vel1_y, pos1_x, pos1_y]]
    for j in range(n_agents):
        for i, val in enumerate(agent_input[j]):
            observation[:, j, i] = float(val)
    pos = observation[..., 2:]
    vel = observation[..., :2]
    logits = model(observation, pos, vel)[0].detach()[0]

    input_lens = [2 * num_actions_per_agent] * n_agents

    logits = logits.view(1, -1)
    split_inputs = torch.split(logits, input_lens, dim=1)
    action_dists = []
    for agent_inputs in split_inputs:
        action_dists.append(TorchDiagGaussian(agent_inputs, u_range))
    actions = []
    for action_dist in action_dists:
        if deterministic:
            action = action_dist.deterministic_sample()[0]
        else:
            action = action_dist.sample()[0]
        if circular_action_constraint:
            action = clamp_with_norm(action, u_range)
        actions.append(action.tolist())
    return actions
