#!/usr/bin/env python3

import numpy as np
import torch
from matplotlib import pyplot as plt
import oineus as oin

def sample_data():
    # sample points from the unit circle
    # return points as differentiable torch tensor
    np.random.seed(1)

    num_points = 120
    noise_std_dev = 0.1

    angles = np.random.uniform(low=0, high=2*np.pi, size=num_points)
    x = np.cos(angles)
    y = np.sin(angles)

    x += np.random.normal(loc=0, scale=noise_std_dev, size=num_points)
    y += np.random.normal(loc=0, scale=noise_std_dev, size=num_points)

    pts = np.vstack((x, y)).T
    pts = torch.Tensor(pts)
    pts.requires_grad_(True)

    return pts


def topological_loss(pts: torch.Tensor, dim: int=1, n: int=2):
    pts_as_numpy = pts.clone().detach().numpy().astype(np.float64)
    fil, longest_edges = oin.get_vr_filtration_and_critical_edges(pts_as_numpy, max_dim=2, max_radius=9.0, n_threads=1)
    top_opt = oin.TopologyOptimizer(fil)

    # dgm = top_opt.compute_diagram(include_inf_points=False)
    # print(dgm.in_dimension(dim))

    eps = top_opt.get_nth_persistence(dim, n)
    print(f"{eps= }")

    indices, values = top_opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, dim)

    critical_sets = top_opt.singletons(indices, values)
    crit_indices, crit_method_values = top_opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)

    crit_indices = np.array(crit_indices, dtype=np.int32)

    dgm_method_edges = longest_edges[indices, :]
    dgm_method_edges_x, dgm_method_edges_y = dgm_method_edges[:, 0], dgm_method_edges[:, 1]

    dgm_method_values = torch.Tensor(values)

    crit_method_edges = longest_edges[crit_indices, :]
    crit_method_edges_x, crit_method_edges_y = crit_method_edges[:, 0], crit_method_edges[:, 1]

    crit_method_values = torch.Tensor(crit_method_values)

    if len(crit_method_edges_x) > 0:
        dgm_loss = torch.sum(torch.abs(torch.sum((pts[dgm_method_edges_x, :] - pts[dgm_method_edges_y, :])**2, axis=1) - dgm_method_values ** 2))
        top_loss = torch.sum(torch.abs(torch.sum((pts[crit_method_edges_x, :] - pts[crit_method_edges_y, :])**2, axis=1) - crit_method_values ** 2))
    else:
        top_loss = torch.zeros(())
        dgm_loss = torch.zeros(())

        top_loss.requires_grad_(True)
        dgm_loss.requires_grad_(True)
    return top_loss, dgm_loss


if __name__ == "__main__":
    draw = False
    use_critical_sets = False

    pts = sample_data()

    lr = 0.2

    opt = torch.optim.SGD([pts], lr=lr)

    for step_idx in range(5):
        opt.zero_grad()

        loss, dgm_loss = topological_loss(pts, dim=1, n=1)

        print(f"{step_idx = }, {loss = }, {dgm_loss = }")

        if draw:
            np_pts = pts.clone().detach().numpy()
            plt.scatter(np_pts[:, 0], np_pts[:, 1], color="green")

        if loss.item() == 0:
            break

        if use_critical_sets:
            loss.backward()
        else:
            dgm_loss.backward()

        opt.step()

        if draw:
            np_pts = pts.clone().detach().numpy()
            plt.scatter(np_pts[:, 0], np_pts[:, 1], color="red")
            plt.show()



