import torch
import numpy as np
from collections import deque
from scipy.special import loggamma
from sklearn.neighbors import NearestNeighbors


def to_spherical(x):
    """
    Convert cartesian coordinates to spherical
    Input: x: torch.Tensor of size (..., D)
    Output: radii: corresponding radii of size (...)
            angles: corresponding angle vectors of size (..., D - 1)
    """
    denom = torch.flip(
        torch.cumsum(
            torch.flip(x.square(), dims=(-1, )), dim=-1
        ), dims=(-1, )
    ).sqrt()
    angles = torch.atan2(denom[..., 1:], x[..., :-1])
    angles[..., -1] = torch.atan2(x[..., -1], x[..., -2])
    radii = denom[..., 0]
    return radii, angles


class EntropyLogger(object):
    def __init__(self, queue_size, thr=1e-8):
        self.queue_size = queue_size
        self.angles = deque(maxlen=queue_size)
        self.radii = deque(maxlen=queue_size)
        self.thr = thr

    def add_weights(self, weights):
        radius, angle = to_spherical(weights)
        self.angles.append(angle)
        self.radii.append(radius.item())

    def get_radius(self):
        return torch.tensor(list(self.radii)).mean().item()

    def get_metrics(self):
        if len(self.angles) < self.queue_size:
            spherical_entropy, log_radius = np.nan, np.nan
        else:
            dim = self.angles[0].shape[0]
            N = self.queue_size

            angles = torch.stack(list(self.angles), dim=0)
            nb = NearestNeighbors(n_neighbors=2).fit(angles)
            knn_distances, _ = nb.kneighbors(angles)
            knn_distances = knn_distances[..., 1]
            knn_distances = np.clip(knn_distances, a_min=self.thr, a_max=None)
            angles_entropy = dim * np.log(knn_distances).mean() + dim / 2 * np.log(np.pi) - \
                         loggamma(dim / 2 + 1) + np.log(N - 1) + np.euler_gamma

            mults = torch.arange(dim - 1, 0, -1).reshape(1, -1)
            log_jacobian = (
                mults * angles[:, :-1].sin().clip(min=self.thr).log()
            ).sum().item() / N

            spherical_entropy = angles_entropy + log_jacobian
            radius = self.get_radius()
            log_radius = dim * np.log(radius)

        return spherical_entropy, log_radius
