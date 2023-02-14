#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import Any, Optional

import torch
import torch_geometric
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from torch import Tensor
from torch import nn
from torch_geometric.nn import MessagePassing, GINEConv, GraphConv, GATv2Conv
from torch_geometric.transforms import BaseTransform


def get_activation_fn(name: Optional[str] = None):
    """Returns a framework specific activation function, given a name string.

    Args:
        name (Optional[str]): One of "relu" (default), "tanh", "elu",
            "swish", or "linear" (same as None)..

    Returns:
        A framework-specific activtion function. e.g. tf.nn.tanh or
            torch.nn.ReLU. None if name in ["linear", None].

    Raises:
        ValueError: If name is an unknown activation function.
    """
    # Already a callable, return as-is.
    if callable(name):
        return name

    # Infer the correct activation function from the string specifier.
    if name in ["linear", None]:
        return None
    if name == "relu":
        return nn.ReLU
    elif name == "tanh":
        return nn.Tanh
    elif name == "elu":
        return nn.ELU

    raise ValueError("Unknown activation ({}) for framework=!".format(name))


def get_edge_index_from_topology(topology_type: str, n_agents: int):
    assert n_agents > 0

    if n_agents == 1:
        edge_index = torch.empty((2, 1)).long()
        edge_index[:, 0] = torch.Tensor([0, 0])
    # Connected to all
    elif topology_type == "full":
        edge_index = torch.empty((2, (n_agents**2))).long()

        index = 0
        for i in range(n_agents):
            for j in range(n_agents):
                edge_index[:, index] = torch.Tensor([i, j])
                index += 1
        assert index == n_agents**2
    # Connected in a ring
    elif topology_type == "ring":
        edge_index = torch.empty((2, n_agents * 2)).long()

        index = 0
        if n_agents > 2:
            for i in range(n_agents - 1):
                edge_index[:, index] = torch.Tensor([i, i + 1])
                index += 1
                edge_index[:, index] = torch.Tensor([i + 1, i])
                index += 1
        edge_index[:, index] = torch.Tensor([n_agents - 1, 0])
        index += 1
        edge_index[:, index] = torch.Tensor([0, n_agents - 1])
        assert index == (n_agents * 2) - 1
    # Connected in a line
    elif topology_type == "line":
        edge_index = torch.empty((2, (n_agents - 2) * 2 + 2)).long()

        index = 0
        for i in range(n_agents - 1):
            edge_index[:, index] = torch.Tensor([i, i + 1])
            index += 1
            edge_index[:, index] = torch.Tensor([i + 1, i])
            index += 1
        assert index == (n_agents - 2) * 2 + 2
    else:
        assert False
    return edge_index


class RelVel(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        (row, col), vel, pseudo = data.edge_index, data.vel, data.edge_attr

        cart = vel[row] - vel[col]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if pseudo is not None:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        return data


def batch_from_rllib_to_ptg(
    x,
    pos: Tensor = None,
    vel: Tensor = None,
    edge_index: Tensor = None,
    comm_radius: float = -1,
    rel_pos: bool = True,
    distance: bool = True,
    rel_vel: bool = True,
) -> torch_geometric.data.Batch:
    batch_size = x.shape[0]
    n_agents = x.shape[1]

    x = x.view(-1, x.shape[-1])
    if pos is not None:
        pos = pos.view(-1, pos.shape[-1])
    if vel is not None:
        vel = vel.view(-1, vel.shape[-1])

    assert (edge_index is None or comm_radius < 0) and (
        edge_index is not None or comm_radius > 0
    )

    b = torch.arange(batch_size, device=x.device)

    graphs = torch_geometric.data.Batch()
    graphs.ptr = torch.arange(0, (batch_size + 1) * n_agents, n_agents)
    graphs.batch = torch.repeat_interleave(b, n_agents)
    graphs.pos = pos
    graphs.vel = vel
    graphs.x = x
    graphs.edge_attr = None

    if edge_index is not None:
        n_edges = edge_index.shape[1]
        # Tensor of shape [batch_size * n_edges]
        # in which edges corresponding to the same graph have the same index.
        batch = torch.repeat_interleave(b, n_edges)
        # Edge index for the batched graphs of shape [2, n_edges * batch_size]
        # we sum to each batch an offset of batch_num * n_agents to make sure that
        # the adjacency matrices remain independent
        batch_edge_index = edge_index.repeat(1, batch_size) + batch * n_agents
        graphs.edge_index = batch_edge_index
    else:
        assert pos is not None
        graphs.edge_index = torch_geometric.nn.pool.radius_graph(
            graphs.pos, batch=graphs.batch, r=comm_radius, loop=False
        )

    graphs = graphs.to(x.device)

    if pos is not None and rel_pos:
        graphs = torch_geometric.transforms.Cartesian(norm=False)(graphs)
    if pos is not None and distance:
        graphs = torch_geometric.transforms.Distance(norm=False)(graphs)
    if vel is not None and rel_vel:
        graphs = RelVel()(graphs)

    return graphs


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_layers: int,
        hidden_dim: int = None,
        activation_fn: Any = None,
        use_norm: bool = False,
    ):
        assert n_layers >= 1
        super().__init__()

        if n_layers == 1:
            assert hidden_dim is None
        if hidden_dim is None:
            assert n_layers == 1

        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn)

        layers = [nn.Linear(in_dim, out_dim if n_layers == 1 else hidden_dim)]
        if activation_fn is not None:
            layers.append(activation_fn())
        if use_norm:
            layers.append(nn.LayerNorm(out_dim if n_layers == 1 else hidden_dim))

        if n_layers > 1:
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if activation_fn is not None:
                    layers.append(activation_fn())
                if use_norm:
                    layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.Linear(hidden_dim, out_dim))
            if activation_fn is not None:
                layers.append(activation_fn())
            if use_norm:
                layers.append(nn.LayerNorm(out_dim))

        self._model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    def reset_parameters(self):
        for layer in self._model():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class MatPosConv(MessagePassing):
    propagate_type = {"x": Tensor, "edge_attr": Tensor}

    def __init__(self, in_dim, out_dim, edge_features, **cfg):
        super().__init__(aggr=cfg["aggr"])

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_features = edge_features
        self.activation_fn = get_activation_fn(cfg["activation_fn"])

        self.message_encoder = nn.Sequential(
            torch.nn.Linear(self.in_dim + self.edge_features, self.out_dim),
            self.activation_fn(),
            torch.nn.Linear(self.out_dim, self.out_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
        )
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        msg = self.message_encoder(torch.cat([x_j, edge_attr], dim=-1))
        return msg


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, edge_features, **cfg):
        super().__init__()

        gnn_types = {"GraphConv", "GATv2Conv", "GINEConv", "MatPosConv"}
        aggr_types = {"add", "mean", "max"}

        self.aggr = cfg["aggr"]
        self.gnn_type = cfg["gnn_type"]

        assert self.aggr in aggr_types
        assert self.gnn_type in gnn_types

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_features = edge_features
        self.activation_fn = get_activation_fn(cfg["activation_fn"])

        if self.gnn_type == "GraphConv":
            self.gnn = GraphConv(
                self.in_dim,
                self.out_dim,
                aggr=self.aggr,
            )
        elif self.gnn_type == "GATv2Conv":
            # Default adds self loops
            self.gnn = GATv2Conv(
                self.in_dim,
                self.out_dim,
                edge_dim=self.edge_features,
                fill_value=0.0,
                share_weights=True,
                add_self_loops=True,
                aggr=self.aggr,
            )
        elif self.gnn_type == "GINEConv":
            self.gnn = GINEConv(
                nn=nn.Sequential(
                    torch.nn.Linear(self.in_dim, self.out_dim),
                    self.activation_fn(),
                ),
                edge_dim=self.edge_features,
                aggr=self.aggr,
            )
        elif self.gnn_type == "MatPosConv":
            self.gnn = MatPosConv(
                self.in_dim,
                self.out_dim,
                edge_features=self.edge_features,
                **cfg,
            )
        else:
            assert False

    def forward(self, x, edge_index, edge_attr):
        if self.gnn_type == "GraphConv":
            out = self.gnn(x, edge_index)
        elif (
            self.gnn_type == "GATv2Conv"
            or self.gnn_type == "GINEConv"
            or self.gnn_type == "MatPosConv"
        ):
            out = self.gnn(x, edge_index, edge_attr)
        else:
            assert False

        return out


class GPPOBranch(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        edge_features,
        n_agents,
        centralised,
        edge_index,
        comm_radius_processed,
        out_features_2=None,
        **cfg,
    ):
        super().__init__()

        self.n_agents = n_agents

        self.in_features = in_features
        self.edge_features = edge_features
        self.out_features = out_features
        self.out_features2 = out_features_2
        self.double_output = out_features_2 is not None
        self.centralised = centralised
        self.edge_index = edge_index
        self.comm_radius = comm_radius_processed

        self.hidden_size = 128

        self.activation_fn = get_activation_fn(cfg["activation_fn"])

        self.hetero_encoders = cfg["heterogeneous"]
        self.hetero_gnns = cfg["heterogeneous"]
        self.hetero_decoders = cfg["heterogeneous"]

        if self.centralised:
            # Will not get edge features
            self.centralised_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        torch.nn.Linear(
                            self.in_features * self.n_agents,
                            256,
                        ),
                        self.activation_fn(),
                        torch.nn.Linear(
                            256,
                            self.hidden_size * self.n_agents,
                        ),
                    )
                    for _ in range(self.n_agents if self.hetero_gnns else 1)
                ]
            )
            self.gnns = None
        else:
            self.gnns = nn.ModuleList(
                [
                    GNN(
                        in_dim=self.in_features,
                        out_dim=self.hidden_size,
                        edge_features=self.edge_features,
                        **cfg,
                    )
                    for _ in range(self.n_agents if self.hetero_gnns else 1)
                ]
            )
            self.centralised_mlps = None

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        self.in_features + self.hidden_size, self.hidden_size
                    ),
                    self.activation_fn(),
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    self.activation_fn(),
                )
                for _ in range(self.n_agents if self.hetero_decoders else 1)
            ]
        )

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(self.hidden_size, self.out_features),
                )
                for _ in range(self.n_agents if self.hetero_decoders else 1)
            ]
        )
        if self.double_output:
            self.heads2 = nn.ModuleList(
                [
                    nn.Sequential(
                        torch.nn.Linear(self.hidden_size, self.out_features2),
                    )
                    for _ in range(self.n_agents if self.hetero_decoders else 1)
                ]
            )
        self.share_init_hetero_networks()

    def forward(self, obs, pos, vel):
        batch_size = obs.shape[0]
        device = obs.device
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)

        if self.centralised:
            embedding = obs.view(batch_size, self.n_agents * self.in_features)

            if self.hetero_gnns:
                embedding = torch.stack(
                    [
                        centralised_mlp(embedding).view(
                            batch_size,
                            self.n_agents,
                            self.hidden_size,
                        )[:, i]
                        for i, centralised_mlp in enumerate(self.centralised_mlps)
                    ],
                    dim=1,
                )
            else:
                embedding = self.centralised_mlps[0](embedding).view(
                    batch_size,
                    self.n_agents,
                    self.hidden_size,
                )

        else:
            graph = batch_from_rllib_to_ptg(
                x=obs,
                pos=pos,
                vel=vel,
                edge_index=self.edge_index,
                comm_radius=self.comm_radius,
            )

            if self.hetero_gnns:
                embedding = torch.stack(
                    [
                        gnn(graph.x, graph.edge_index, graph.edge_attr).view(
                            batch_size,
                            self.n_agents,
                            self.hidden_size,
                        )[:, i]
                        for i, gnn in enumerate(self.gnns)
                    ],
                    dim=1,
                )

            else:
                embedding = self.gnns[0](
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                ).view(batch_size, self.n_agents, self.hidden_size)

        if self.hetero_decoders:
            embedding = torch.stack(
                [
                    decoder(torch.cat([obs[:, i], embedding[:, i]], dim=-1))
                    for i, decoder in enumerate(self.decoders)
                ],
                dim=1,
            )
        else:
            embedding = self.decoders[0](torch.cat([obs, embedding], dim=-1))

        if self.hetero_decoders:
            out = torch.stack(
                [head(embedding[:, i]) for i, head in enumerate(self.heads)],
                dim=1,
            )
            if self.double_output:
                out2 = torch.stack(
                    [head2(embedding[:, i]) for i, head2 in enumerate(self.heads2)],
                    dim=1,
                )
        else:
            out = self.heads[0](embedding)
            if self.double_output:
                out2 = self.heads2[0](embedding)

        return out, (out2 if self.double_output else None)

    def share_init_hetero_networks(self):
        for child in self.children():
            assert isinstance(child, nn.ModuleList)
            for agent_index, agent_model in enumerate(child.children()):
                if agent_index == 0:
                    state_dict = agent_model.state_dict()
                else:
                    agent_model.load_state_dict(state_dict)


class GPPO(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **cfg):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        topology_types = {"full", "ring", "line"}

        self.n_agents = len(obs_space.original_space)
        self.outputs_per_agent = int(num_outputs / self.n_agents)
        self._cur_value = None

        self.pos_dim = cfg["pos_dim"]
        self.pos_start = cfg["pos_start"]
        self.vel_start = cfg["vel_start"]
        self.vel_dim = cfg["vel_dim"]
        self.trainer = cfg["trainer"]

        self.share_action_value = cfg["share_action_value"]
        self.share_observations = cfg["share_observations"]
        self.centralised_critic = cfg["centralised_critic"]
        self.use_mlp = cfg["use_mlp"]

        self.use_beta = cfg["use_beta"]
        self.add_agent_index = cfg["add_agent_index"]

        self.topology_type = cfg.get("topology_type", None)
        self.comm_radius_processed = cfg.get("comm_radius", -1)

        assert self.topology_type in topology_types or self.comm_radius_processed > 0

        if self.use_mlp:
            assert self.share_observations and self.centralised_critic
        if self.centralised_critic and self.share_action_value:
            assert self.use_mlp

        self.obs_shape = obs_space.original_space[0].shape[0]
        # Remove position
        self.obs_shape -= self.pos_dim
        if self.add_agent_index:
            self.obs_shape += 1

        self.edge_features_dim = (
            self.vel_dim + self.pos_dim + (1 if self.pos_dim > 0 else 0)
        )

        # Communication yes or no
        if self.share_observations:
            if self.topology_type is not None:
                self.edge_index = get_edge_index_from_topology(
                    topology_type=self.topology_type, n_agents=self.n_agents
                )
            else:
                self.edge_index = None
        else:
            self.edge_index = torch.empty(2, self.n_agents).long()
            for i in range(self.n_agents):
                self.edge_index[:, i] = torch.Tensor([i, i])
            self.comm_radius_processed = -1

        if self.edge_index is not None:
            self.edge_index, _ = torch_geometric.utils.remove_self_loops(
                self.edge_index
            )

        if not self.share_action_value:
            self.gnn = GPPOBranch(
                in_features=self.obs_shape,
                out_features=self.outputs_per_agent,
                double_output=False,
                out_features_2=None,
                edge_features=self.edge_features_dim,
                n_agents=self.n_agents,
                centralised=self.use_mlp,
                edge_index=self.edge_index,
                comm_radius_processed=self.comm_radius_processed,
                **cfg,
            )
            self.gnn_value = GPPOBranch(
                in_features=self.obs_shape,
                out_features=1,
                double_output=False,
                out_features_2=None,
                edge_features=self.edge_features_dim,
                n_agents=self.n_agents,
                centralised=self.use_mlp or self.centralised_critic,
                edge_index=self.edge_index,
                comm_radius_processed=self.comm_radius_processed,
                **cfg,
            )
        else:
            self.gnn = GPPOBranch(
                in_features=self.obs_shape,
                out_features=self.outputs_per_agent,
                double_output=True,
                out_features_2=1,
                edge_features=self.edge_features_dim,
                n_agents=self.n_agents,
                centralised=self.use_mlp,
                edge_index=self.edge_index,
                comm_radius_processed=self.comm_radius_processed,
                **cfg,
            )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # print(f"Obs len: {len(input_dict['obs'])}\n Single obs shape: {input_dict['obs'][0].shape}")
        batch_size = input_dict["obs"][0].shape[0]
        device = input_dict["obs"][0].device

        obs = torch.stack(input_dict["obs"], dim=1)
        if self.add_agent_index:
            agent_index = (
                torch.arange(self.n_agents, device=device)
                .repeat(batch_size, 1)
                .unsqueeze(-1)
            )
            obs = torch.cat((obs, agent_index), dim=-1)
        pos = (
            obs[..., self.pos_start : self.pos_start + self.pos_dim]
            if self.pos_dim > 0
            else None
        )
        vel = (
            obs[..., self.vel_start : self.vel_start + self.vel_dim]
            if self.vel_dim > 0
            else None
        )
        obs_no_pos = torch.cat(
            [
                obs[..., : self.pos_start],
                obs[..., self.pos_start + self.pos_dim :],
            ],
            dim=-1,
        ).view(
            batch_size, self.n_agents, self.obs_shape
        )  # This acts like an assertion

        if not self.share_action_value:
            outputs, _ = self.gnn(obs=obs_no_pos, pos=pos, vel=vel)
            values, _ = self.gnn_value(obs=obs_no_pos, pos=pos, vel=vel)
        else:
            outputs, values = self.gnn(obs=obs_no_pos, pos=pos, vel=vel)

        values = values.view(batch_size, self.n_agents)
        if self.trainer == "PPOTrainer":
            assert self.n_agents == 1
            values = values.squeeze(-1)  # If using default ppo trainer with one agent
        self._cur_value = values

        outputs = outputs.view(batch_size, self.n_agents * self.outputs_per_agent)

        assert not outputs.isnan().any()

        return outputs, state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value
