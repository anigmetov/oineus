#!python3

import numpy as np

import oineus as oin

# Example of computing kernel diagrams
def sample_data(num_points, noise_std_dev=0.1):
    # sample points from the unit circle
    # return points as differentiable torch tensor
    np.random.seed(1)

    angles = np.random.uniform(low=0, high=2*np.pi, size=num_points)
    x = np.cos(angles)
    y = np.sin(angles)

    x += np.random.normal(loc=0, scale=noise_std_dev, size=num_points)
    y += np.random.normal(loc=0, scale=noise_std_dev, size=num_points)

    pts_2 = np.vstack((x, y)).T

    # Generate a random 3x3 matrix
    A = np.random.rand(3, 3)

    # Apply QR decomposition
    Q, R = np.linalg.qr(A)

    # Step 2: Extract the first two rows to get a 2x3 matrix
    B = Q[:2, :]
    print(B.shape, pts_2.shape)
    pts_3 =  pts_2 @ B

    return pts_2, pts_3, B

# points in plane, orthogonal embedding in space, embedding matrix
pts_2, pts_3, B = sample_data(10)

max_radius = 3.0
fil_2, edges_2 = oin.get_vr_filtration_and_critical_edges(pts_2, max_dim = 2, max_radius=max_radius, n_threads=1)
fil_3, edges_3 = oin.get_vr_filtration_and_critical_edges(pts_3, max_dim = 2, max_radius = max_radius, n_threads=1)

fil_min = oin.min_filtration(fil_2, fil_3)

id_domain = fil_3.size() + fil_min.size() + 1
id_codomain = id_domain + 1

# id_domain: id of vertex at the top of the cylinder,
# i.e., we multiply fil_3 with id_domain
# id_codomain: id of vertex at the bottom of the cylinder
# i.e, we multiply fil_min with id_codomain

v0 = oin.Simplex(id_domain, [id_domain])
v1 = oin.Simplex(id_codomain, [id_codomain])
fil_cyl = oin.mapping_cylinder(fil_3, fil_min, v0, v1)

# to get a subcomplex, we multiply each fil_3 with id_domain
fil_3_prod = oin.multiply_filtration(fil_3, v0)

hdim = 1
# ker_dgms, coker_dgms = oin.kernel_cokernel_diagrams(fil_3_prod, fil_cyl)
ker_dgms = oin.kernel_diagrams(fil_3_prod, fil_cyl)

for pt in ker_dgms.in_dimension(hdim, as_numpy=False):
    b, d = pt.birth_index, pt.death_index

    # birth: comes from codomain (cylinder) complex
    # can be either in fil_3 or in fil_cyl
    c_birth = fil_cyl.get_cell(b)
    birth_simplex = c_birth.cell_1
    if c_birth.cell_2 == v1:
        true_birth_simplex = fil_min.cell_by_uid(birth_simplex.get_uid())
    else:
        true_birth_simplex = fil_3.cell_by_uid(birth_simplex.get_uid())

    # death: comes from included complex L, i.e., fil_3
    if d < fil_cyl.size():
        # point is finite
        c_death = fil_cyl.get_cell(d)
        death_simplex = c_death.cell_1
        true_death_simplex = fil_3.cell_by_uid(death_simplex.get_uid())
        print(f"{true_birth_simplex = }, {true_death_simplex = }")

    # true_birth_simplex.sorted_id and true_death_simplex.sorted_id are indices
    # in the tensors of critical values for fil_min and fil_3, resp.
    # so we can differentiate them
