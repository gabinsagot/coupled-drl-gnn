import torch
from torch_geometric.data import Data

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_features(graph: Data) -> Data:
    # construct features
    velocity = graph.x[:, 0:2]
    pressure = graph.x[:, 3].unsqueeze(1)
    levelset = graph.x[:, 4].unsqueeze(1)
    nodetype = graph.x[:, 5].unsqueeze(1)

    graph.x = torch.cat(
        (
            velocity,
            pressure,
            levelset,
            graph.pos[:, :2],
            nodetype,
        ),
        dim=1,
    )

    # hide Vz in target
    target_velocity = graph.y[:, 0:2]
    target_pressure = graph.y[:, 3].unsqueeze(1)
    graph.y = torch.cat((target_velocity, target_pressure), dim=1)

    return graph
