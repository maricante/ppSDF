import torch
from bpSDF import bpSDF


class ippSDF(bpSDF):
    """
    Implicit function model for Signed Distance Function (SDF) representation.
    args:
        n_func: number of basis functions per segment
        n_seg: number of segments
        qd: weight for distance loss
        qn: weight for normal loss
        qt: weight for tension loss
        sigma: noise level
        device: 'cpu' or 'cuda'
    """
    def __init__(self, n_func=4, n_seg=5,
                 qd = 1.0, qn = 1.0, qt = 6e-2,
                 sigma = 1,
                 device='cpu'):
        super().__init__(n_func, n_seg, device)

        # HYPERPARAMETERS
        self.sigma = sigma
        self.qd = qd
        self.qn = qn
        self.qt = qt

        # PRECISION MATRIX INITIALIZATION
        self.P = torch.eye(len(self.w), device=self.device) / 1e-2

        # REGULARIZATION POINT DISTANCES
        # self.ray_distances = torch.cat([torch.arange(-100, -0.01, 0.03),
        #                                 torch.arange(0.05, 100, 0.03)]).to(self.device)
        self.ray_distances = torch.arange(0.1, 100, 0.05).to(self.device)
        self.N_reg = 6

    def init_w_sphere(self, radius, center=torch.zeros(1, 3)):
        """
        Initialize weights to sphere prior.
        args:
            radius: radius of sphere
            center: center of sphere
        """
        axis = torch.linspace(self.domain_min, self.domain_max, self.grid_res)
        center = center.type(torch.float32).to(self.device)
        X = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'),
                        dim=-1).reshape(-1, 3).to(self.device)
        y = torch.sqrt(torch.sum((X-center) ** 2, dim=1, keepdim=True)) - radius
        idx = torch.randperm(X.shape[0])[:100000]
        X = X[idx, :].to(self.device)
        y = y[idx, :].to(self.device)
        Xs = X.split(100, dim=0)
        ys = y.split(100, dim=0)
        i = 1
        for Xi, yi in zip(Xs, ys):
            print("Batch {} / {}".format(i, len(Xs)))
            Psi, _, _ = self.generate_bf(Xi.to(self.device))
            self.update(Psi, yi)
            i += 1

    def update_pos(self, p):
        """
        Update weights with distance loss.
        args:
            p: surface points
        """
        Psi, _, _ = self.generate_bf(p)
        self.update(Psi * self.qd, 
                    torch.zeros([Psi.shape[0], 1], device=self.device))

    def update_grad(self, p, n):
        """
        Update weights with normal loss.
        args:
            p: surface points
            n: surface normals
        """
        _, dPsi, _ = self.generate_bf(p, order=1)
        dPsi = torch.vstack((dPsi[:, :, 0], dPsi[:, :, 1], dPsi[:, :, 2])).to(self.device)
        n = torch.vstack((n[:, 0].unsqueeze(-1),
                          n[:, 1].unsqueeze(-1),
                          n[:, 2].unsqueeze(-1))).to(self.device)
        self.update(dPsi * self.qn / 3, n * self.qn / 3)

    def regularize_ray(self, p, n):
        """
        Regularize using tension loss on points sampled from normal rays.
        args:
            p: surface points
            n: surface normals
        """
        p_dist = n.expand(len(self.ray_distances), -1, -1) * \
            self.ray_distances.view(-1, 1, 1).repeat(1, len(p), 3)

        p_reg = p.expand(len(self.ray_distances), -1, -1) + p_dist
        p_reg = p_reg.reshape(-1, 3)
        p_reg = p_reg[torch.logical_and(torch.all(p_reg > self.domain_min, dim=1),
                                        torch.all(p_reg < self.domain_max, dim=1)), :]

        idx = torch.randperm(len(p_reg))[:self.N_reg * p.shape[0]]
        p_reg = p_reg[idx, :]
        _, _, ddPsi_tension = self.generate_bf(p_reg, order=2)

        ddPsi_tension = torch.vstack((ddPsi_tension[:, :, 0] * self.qt / 9,
                            ddPsi_tension[:, :, 1] * self.qt / 9,
                            ddPsi_tension[:, :, 2] * self.qt / 9,
                            2 * ddPsi_tension[:, :, 3] * self.qt / 9,
                            2 * ddPsi_tension[:, :, 4] * self.qt / 9,
                            2 * ddPsi_tension[:, :, 5] * self.qt / 9,
                            )).to(self.device)
        self.update(ddPsi_tension, 
                    torch.zeros([ddPsi_tension.shape[0], 1], device=self.device).to(self.device))

    def update(self, Psi, y):
        """
        Pefrorm incremental update.
        args:
            Psi: basis functions
            y: target values
        """
        K = torch.linalg.solve(
                torch.eye(Psi.shape[0], device=self.device)
                    + self.sigma**(-2) * Psi @ self.P @ Psi.T,
                1/self.sigma**2 * self.P @ Psi.T,
                left = False)
        self.w += K @ (y - Psi @ self.w)
        self.P -= K @ Psi @ self.P
