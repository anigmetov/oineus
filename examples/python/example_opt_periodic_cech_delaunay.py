import torch

import oineus.diff as od


def get_pts():
    axis = (torch.arange(6, dtype=torch.float64) + 0.5) / 6
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    background = torch.stack((x.ravel(), y.ravel()), dim=1)
    index = torch.arange(len(background), dtype=torch.float64)
    background += 0.005 * torch.stack((torch.sin(index), torch.cos(2 * index)), dim=1)
    pair = torch.tensor([[0.01, 0.5], [0.99, 0.5]], dtype=torch.float64)
    return torch.cat((pair, background)).requires_grad_()


def wrap(pts, bbox_min, bbox_max):
    return torch.remainder(pts - bbox_min, bbox_max - bbox_min) + bbox_min


bbox_min = torch.zeros(2, dtype=torch.float64)
bbox_max = torch.ones(2, dtype=torch.float64)
pts = get_pts()
opt = torch.optim.SGD([pts], lr=1.5)

for step_idx in range(5):
    opt.zero_grad()
    fil = od.cech_delaunay_filtration(
        pts, periodic=True, bbox_min=bbox_min, bbox_max=bbox_max
    )
    dgm0 = od.persistence_diagram(fil)[0]
    loss = dgm0[:, 1].min()
    loss.backward()
    opt.step()
    with torch.no_grad():
        pts.copy_(wrap(pts, bbox_min, bbox_max))
    print(step_idx, loss.item())
