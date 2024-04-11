import os
import numpy as np
import torch
import time
import skimage
import open3d as o3d
import matplotlib.pyplot as plt
import time

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class SDFmodel():
    def __init__(self,
                 n_func=4,
                 n_seg=5,
                 qd = 1.0,
                 qn = 1.0,
                 qt = 6e-2,
                 sigma = 1,
                 device='cpu'):
        self.n_func = n_func
        self.n_seg = n_seg
        self.device = device

        # HYPERPARAMETERS
        self.sigma = sigma
        self.qd = qd
        self.qn = qn
        self.qt = qt

        # INPUT SPACE
        self.nbDim = 128
        self.domain_min = -1.0
        self.domain_max = 1.0

        # BASIS FUNCTION MATRICES
        self.B = self.compute_B()
        self.C = self.compute_C()

        # HELPER VARIABLES
        self.nw = self.C.size(-1)
        self.i = torch.arange(self.n_func, device=self.device)
        self.segment_size = 1.0/self.n_seg

        # INITIALIZATION OF PRIOR VARIABLES
        self.w = torch.zeros([self.nw ** 3, 1]).to(self.device)
        self.iB = torch.eye(len(self.w), device=self.device) / 1e-2

        # REGULARIZATION POINTS
        self.ray_distances = torch.cat([torch.arange(-100, -0.01, 0.03),
                                        torch.arange(0.05, 100, 0.03)]).to(self.device)

    def init_w_sphere(self, radius, center=torch.zeros(1, 3)):
        axis = torch.linspace(self.domain_min, self.domain_max, self.nbDim)
        center = center.type(torch.float32).to(self.device)
        X = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), dim=-1).reshape(-1, 3).to(self.device)
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
        torch.save(self.w, 'priors/sphere_weights_{}fun_{}seg.pt'.format(self.n_func, self.n_seg))

    def update_pos(self, p):
        Psi, _, _ = self.generate_bf(p)
        self.update(Psi * self.qd, 
                    torch.zeros([Psi.shape[0], 1], device=self.device))

    def update_grad(self, p, n):
        _, dPsi, _ = self.generate_bf(p, order=1)
        dPsi = torch.vstack((dPsi[:, :, 0], dPsi[:, :, 1], dPsi[:, :, 2])).to(self.device)
        n = torch.vstack((n[:, 0].unsqueeze(-1),
                          n[:, 1].unsqueeze(-1),
                          n[:, 2].unsqueeze(-1))).to(self.device)
        self.update(dPsi * self.qn / 3, n * self.qn / 3)
            
    def regularize_ray(self, p, n):
        p_dist = n.expand(len(self.ray_distances), -1, -1) * self.ray_distances.view(-1, 1, 1).repeat(1, len(p), 3)
    
        p_reg = p.expand(len(self.ray_distances), -1, -1) + p_dist
        p_reg = p_reg.reshape(-1, 3)
        p_reg = p_reg[torch.logical_and(torch.all(p_reg > self.domain_min, dim=1), torch.all(p_reg < self.domain_max, dim=1)), :]

        idx = torch.randperm(len(p_reg))[:6 * p.shape[0]]
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
        K = torch.linalg.solve(
                torch.eye(Psi.shape[0], device=self.device) + self.sigma**(-2) * Psi @ self.iB @ Psi.T,
                1/self.sigma**2 * self.iB @ Psi.T,
                left = False
            )
        self.w += K @ (y - Psi @ self.w)
        self.iB -= K @ Psi @ self.iB

    def generate_bf(self, p, order=0):
        N = p.shape[0]
        p = ((p - self.domain_min)/(self.domain_max-self.domain_min)).reshape(-1)

        p_split = torch.split(p, 100, dim=0)
        T_split = []
        dT_split = []
        ddT_split = []
        idx_expand_split = []
        for p_s in p_split:
            T,dT,ddT,idx_expand = self.compute_T_1D(p_s, order)
            T_split.append(T)
            dT_split.append(dT)
            ddT_split.append(ddT)
            idx_expand_split.append(idx_expand)
        T = torch.cat(T_split)
        if order > 0: 
            dT = torch.cat(dT_split)
        if order == 2: 
            ddT = torch.cat(ddT_split)
        idx_expand = torch.cat(idx_expand_split)

        phi = torch.matmul(T,self.B).matmul(self.C)
        phi = phi.reshape(N,3,self.nw)
        phi_x,phi_y,phi_z = phi[:,0,:],phi[:,1,:],phi[:,2,:]
        phi_xy = torch.einsum("ij,ik->ijk",phi_x,phi_y).view(-1,self.nw**2)
        phi_xyz = torch.einsum("ij,ik->ijk",phi_xy,phi_z).view(-1,self.nw**3)

        d_phi_xyz,dd_phi_full = None,None
        if order > 0:
            d_phi = torch.matmul(dT,self.B).matmul(self.C)
            d_phi = d_phi.reshape(N,3,self.nw)  
            d_phi_x_1D,d_phi_y_1D,d_phi_z_1D= d_phi[:,0,:],d_phi[:,1,:],d_phi[:,2,:]
            d_phi_x = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",d_phi_x_1D,phi_y).view(-1,self.nw**2),phi_z).view(-1,self.nw**3)
            d_phi_y = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",phi_x,d_phi_y_1D).view(-1,self.nw**2),phi_z).view(-1,self.nw**3)
            d_phi_z = torch.einsum("ij,ik->ijk",phi_xy,d_phi_z_1D).view(-1,self.nw**3)
            d_phi_xyz = torch.cat((d_phi_x.unsqueeze(-1),d_phi_y.unsqueeze(-1),d_phi_z.unsqueeze(-1)),dim=-1)
        if order == 2:
            dd_phi = torch.matmul(ddT,self.B).matmul(self.C)
            dd_phi = dd_phi.reshape(N,3,self.nw)
            dd_phi_x_1D,dd_phi_y_1D,dd_phi_z_1D= dd_phi[:,0,:],dd_phi[:,1,:],dd_phi[:,2,:]
            dd_phi_x = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",dd_phi_x_1D,phi_y).view(-1,self.nw**2),phi_z).view(-1,self.nw**3)
            dd_phi_y = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",phi_x,dd_phi_y_1D).view(-1,self.nw**2),phi_z).view(-1,self.nw**3)
            dd_phi_z = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",phi_x,phi_y).view(-1,self.nw**2),dd_phi_z_1D).view(-1,self.nw**3)
            dd_phi_xy = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",d_phi_x_1D,d_phi_y_1D).view(-1,self.nw**2),phi_z).view(-1,self.nw**3)
            dd_phi_xz = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",d_phi_x_1D, phi_y).view(-1,self.nw**2),d_phi_z_1D).view(-1,self.nw**3)
            dd_phi_yz = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",phi_x,d_phi_y_1D).view(-1,self.nw**2),d_phi_z_1D).view(-1,self.nw**3)
            dd_phi_full = torch.cat((dd_phi_x.unsqueeze(-1),dd_phi_y.unsqueeze(-1),dd_phi_z.unsqueeze(-1),
                                     dd_phi_xy.unsqueeze(-1), dd_phi_xz.unsqueeze(-1),dd_phi_yz.unsqueeze(-1)),dim=-1)

        return phi_xyz, d_phi_xyz, dd_phi_full

    def compute_T_1D(self, t, order=0):
        N = len(t)
        t = torch.clamp(t, min=1e-4, max=1-1e-4)
        T = torch.zeros((N,self.n_func * self.n_seg),device=self.device)
        dT = torch.zeros_like(T)
        ddT = torch.zeros_like(T)

        idx = torch.floor(t/self.segment_size).long()
        t = (t - idx*self.segment_size)/self.segment_size
        
        T_seg = t.unsqueeze(-1)**self.i
        dT_seg = torch.zeros_like(T_seg)
        ddT_seg = torch.zeros_like(T_seg)

        idx_expand = idx.unsqueeze(-1).expand(-1,self.n_func)*self.n_func + self.i.unsqueeze(0).expand(N,-1)
        T.scatter_(1,idx_expand,T_seg)
        
        if order > 0:
            dT_seg[:,1:] = self.i[1:]*T_seg[:,:self.n_func-1] * self.n_seg
            dT.scatter_(1,idx_expand,dT_seg)
        
        if order == 2:
            ddT_seg[:,2:] = self.i[2:] * self.i[1:-1] * T_seg[:,:-2] * self.n_seg**2
            ddT.scatter_(1,idx_expand,ddT_seg)
        
        return T, dT, ddT, idx_expand
    
    def compute_C(self):
        if self.n_func > 4:
            C0 = torch.zeros([4+self.n_func-4, 2+self.n_func-4], dtype=torch.float32)
            C0[:self.n_func-4, :self.n_func-4] = torch.eye(self.n_func-4, dtype=torch.float32)
            C0[self.n_func-4:, self.n_func-4:] = torch.tensor([[1, 0, 0, -1], [0, 1, 1, 2]], dtype=torch.float32).T
        else:
            C0 = torch.tensor([[1, 0, 0, -1], [0, 1, 1, 2]], dtype=torch.float32).T

        C = torch.zeros([2 + (self.n_seg-1)*C0.shape[0] + self.n_func - 2, 2 + (self.n_seg-1)*C0.shape[1] + self.n_func - 2], dtype=torch.float32).to(self.device)
        C[:2, :2] = torch.eye(2, dtype=torch.float32)
        r = 2
        c = 2
        for n in range(self.n_seg-1):
            C[r:r+C0.shape[0], c:c+C0.shape[1]] = C0
            r += C0.shape[0]
            c += C0.shape[1]
        C[r:, c:] = torch.eye(self.n_func - 2, dtype=torch.float32)
        return C

    def compute_B(self):
        B0 = torch.zeros([self.n_func, self.n_func], dtype=torch.float32)
        for n in range(1, self.n_func+1):
            for i in range(1, self.n_func+1):
                B0[self.n_func-i, n-1] = (-1) ** (self.n_func-i-n) * -self.binomial(self.n_func-1, i-1) \
                      * self.binomial((self.n_func-1)-(i-1), (self.n_func-1)-(n-1)-(i-1))
        B = torch.kron(torch.eye(self.n_seg, dtype=torch.float32), B0).to(self.device)
        return B
    
    def binomial(self,n, i):
        n = torch.tensor(n, dtype=torch.float64,device=self.device)
        i = torch.tensor(i, dtype=torch.float64,device=self.device)
        mask = (n >= 0) & (i >= 0)
        b = torch.where(mask, torch.exp(torch.lgamma(n + 1) - torch.lgamma(i + 1) - torch.lgamma(n - i + 1)), \
                        torch.tensor(0.0, dtype=torch.float64,device=self.device))
        return b

    def forward(self, p, order=0):
        p_split = torch.split(p, 100, dim=0)
        d =[]
        dd = []
        for p_s in p_split:
            phi_p, dphi_p, _ = self.generate_bf(p_s,order=order)
            d_s = torch.matmul(phi_p, self.w)
            d.append(d_s)
            if order > 0:
                dd_s = torch.einsum("ijk,jl->ik",dphi_p,self.w)
                dd.append(dd_s)
        d = torch.cat(d,dim=0)
        if order > 0:
            dd = torch.cat(dd,dim=0)
        return d, dd


    def get_sdf(self, p, order=0):
        d, dd = self.forward(p, order=order)
        d = d.reshape(self.nbDim, self.nbDim, self.nbDim)
        dd = dd.reshape(self.nbDim, self.nbDim, self.nbDim, 3)
        return d, dd

