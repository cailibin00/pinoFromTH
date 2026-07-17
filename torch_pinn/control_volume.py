from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .geometry import PhysicalParams


@dataclass
class FaceBatch:
    right: torch.Tensor
    left: torch.Tensor
    top: torch.Tensor
    bottom: torch.Tensor
    weights_theta: torch.Tensor
    weights_radius: torch.Tensor
    centers: torch.Tensor
    delta_radius: torch.Tensor
    delta_theta: torch.Tensor


class ControlVolumeGrid:
    def __init__(
        self,
        params: PhysicalParams,
        cells_r: int,
        cells_theta: int,
        gauss_order: int,
    ):
        if cells_r < 1 or cells_theta < 1:
            raise ValueError("Control-volume dimensions must be positive")
        self.params = params
        self.cells_r = cells_r
        self.cells_theta = cells_theta
        self.gauss_order = gauss_order
        r_edges = np.linspace(params.r_min, params.r_max, cells_r + 1)
        t_edges = np.linspace(params.theta_min, params.theta_max, cells_theta + 1)
        records = []
        for i in range(cells_r):
            for j in range(cells_theta):
                records.append((r_edges[i], r_edges[i + 1], t_edges[j], t_edges[j + 1]))
        self.bounds = np.asarray(records, dtype=np.float64)
        self.centers = np.column_stack(
            [
                0.5 * (self.bounds[:, 0] + self.bounds[:, 1]),
                0.5 * (self.bounds[:, 2] + self.bounds[:, 3]),
            ]
        )
        self.gauss_nodes, self.gauss_weights = np.polynomial.legendre.leggauss(
            gauss_order
        )

    def __len__(self) -> int:
        return len(self.bounds)

    def all_indices(self) -> np.ndarray:
        return np.arange(len(self), dtype=np.int64)

    def sample_indices(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if size >= len(self):
            return self.all_indices()
        return rng.choice(len(self), size=size, replace=False)

    def center_tensor(
        self,
        indices: Iterable[int] | np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        index = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices)
        return torch.as_tensor(self.centers[index], device=device, dtype=dtype)

    def face_batch(
        self,
        indices: Iterable[int] | np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ) -> FaceBatch:
        index = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices)
        bounds = self.bounds[index]
        r_left, r_right = bounds[:, 0], bounds[:, 1]
        t_bottom, t_top = bounds[:, 2], bounds[:, 3]
        r_mid = 0.5 * (r_left + r_right)
        t_mid = 0.5 * (t_bottom + t_top)
        half_r = 0.5 * (r_right - r_left)
        half_t = 0.5 * (t_top - t_bottom)
        radius_nodes = r_mid[:, None] + half_r[:, None] * self.gauss_nodes[None, :]
        theta_nodes = t_mid[:, None] + half_t[:, None] * self.gauss_nodes[None, :]

        right = np.stack(
            [np.broadcast_to(r_right[:, None], theta_nodes.shape), theta_nodes], axis=-1
        )
        left = np.stack(
            [np.broadcast_to(r_left[:, None], theta_nodes.shape), theta_nodes], axis=-1
        )
        top = np.stack(
            [radius_nodes, np.broadcast_to(t_top[:, None], radius_nodes.shape)], axis=-1
        )
        bottom = np.stack(
            [radius_nodes, np.broadcast_to(t_bottom[:, None], radius_nodes.shape)], axis=-1
        )
        weights_theta = half_t[:, None] * self.gauss_weights[None, :]
        weights_radius = half_r[:, None] * self.gauss_weights[None, :]

        tensor = lambda value: torch.as_tensor(value, device=device, dtype=dtype)
        return FaceBatch(
            right=tensor(right),
            left=tensor(left),
            top=tensor(top),
            bottom=tensor(bottom),
            weights_theta=tensor(weights_theta),
            weights_radius=tensor(weights_radius),
            centers=tensor(self.centers[index]),
            delta_radius=tensor((r_right - r_left)[:, None]),
            delta_theta=tensor((t_top - t_bottom)[:, None]),
        )
