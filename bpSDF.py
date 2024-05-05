import torch


class bpSDF():
    """
    Implicit function model for Signed Distance Function (SDF) representation.
    args:
        n_func: number of basis functions per segment
        n_seg: number of segments
        device: 'cpu' or 'cuda'
    """
    def __init__(self, n_func=4, n_seg=5, device='cpu'):
        self.n_func = n_func
        self.n_seg = n_seg
        self.device = device

        # INPUT SPACE
        self.grid_res = 128
        self.domain_min = -1.0
        self.domain_max = 1.0

        # BASIS FUNCTION MATRICES
        self.B = self.compute_B()
        self.C = self.compute_C()

        # HELPER VARIABLES
        self.i = torch.arange(self.n_func, device=self.device)
        self.segment_size = 1.0/self.n_seg

        # WEIGHT MATRIX INITIALIZATION
        self.w = torch.zeros([self.C.size(-1) ** 3, 1]).to(self.device)

    def forward(self, p, order=0):
        """
        Compute SDF(and derivatives) at input points.
        args:
            p: input points
            order: order of derivatives
        returns:
            d: SDF
            dd: derivatives
        """
        p_split = torch.split(p, 100, dim=0) # split inputs to avoid memory issues
        d, dd = [], []
        for p_s in p_split:
            phi_p, dphi_p, _ = self.generate_bf(p_s,order=order)
            d.append(torch.matmul(phi_p, self.w))
            if order > 0:
                dd.append(torch.einsum("ijk,jl->ik",dphi_p,self.w))
        d = torch.cat(d,dim=0)
        if order > 0:
            dd = torch.cat(dd,dim=0)
        return d, dd

    def generate_bf(self, p, order=0):
        """
        Generate basis functions and their derivatives.
        args:
            p: input points
            order: order of derivatives
        returns:
            phi_xyz: basis functions
            d_phi_xyz: derivatives of basis functions
            dd_phi_full: second derivatives of basis functions
        """
        N = p.shape[0]
        nw = self.C.size(-1)
        p = ((p - self.domain_min)/(self.domain_max-self.domain_min)).reshape(-1)

        p_split = torch.split(p, 100, dim=0)  # split inputs to avoid memory issues
        T_split, dT_split, ddT_split, idx_expand_split = [], [], [], []
        for p_s in p_split:
            T,dT,ddT,idx_expand = self.compute_T_1D(p_s, order)
            T_split.append(T)
            dT_split.append(dT)
            ddT_split.append(ddT)
            idx_expand_split.append(idx_expand)
        T = torch.cat(T_split)
        if order > 0:
            dT = torch.cat(dT_split)
        if order > 1:
            ddT = torch.cat(ddT_split)
        idx_expand = torch.cat(idx_expand_split)

        phi = torch.matmul(T,self.B).matmul(self.C)
        phi = phi.reshape(N,3,nw)
        phi_x,phi_y,phi_z = phi[:,0,:],phi[:,1,:],phi[:,2,:]
        phi_xy = torch.einsum("ij,ik->ijk",phi_x,phi_y).view(-1,nw**2)
        phi_xyz = torch.einsum("ij,ik->ijk",phi_xy,phi_z).view(-1,nw**3)

        d_phi_xyz,dd_phi_full = None,None
        if order > 0:
            d_phi = torch.matmul(dT,self.B).matmul(self.C)
            d_phi = d_phi.reshape(N,3,nw)
            d_phi_x_1D,d_phi_y_1D,d_phi_z_1D= d_phi[:,0,:],d_phi[:,1,:],d_phi[:,2,:]
            d_phi_x = torch.einsum(
                "ij,ik->ijk",torch.einsum(
                    "ij,ik->ijk",d_phi_x_1D,phi_y).view(-1,nw**2),phi_z).view(-1,nw**3)
            d_phi_y = torch.einsum(
                "ij,ik->ijk",torch.einsum(
                    "ij,ik->ijk",phi_x,d_phi_y_1D).view(-1,nw**2),phi_z).view(-1,nw**3)
            d_phi_z = torch.einsum(
                "ij,ik->ijk",phi_xy,d_phi_z_1D).view(-1,nw**3)
            d_phi_xyz = torch.cat(
                (d_phi_x.unsqueeze(-1),d_phi_y.unsqueeze(-1),d_phi_z.unsqueeze(-1)),
                dim=-1)
        if order > 1:
            dd_phi = torch.matmul(ddT,self.B).matmul(self.C)
            dd_phi = dd_phi.reshape(N,3,nw)
            dd_phi_x_1D,dd_phi_y_1D,dd_phi_z_1D= dd_phi[:,0,:],dd_phi[:,1,:],dd_phi[:,2,:]
            dd_phi_x = torch.einsum(
                "ij,ik->ijk",torch.einsum(
                    "ij,ik->ijk",dd_phi_x_1D,phi_y).view(-1,nw**2),phi_z).view(-1,nw**3)
            dd_phi_y = torch.einsum(
                "ij,ik->ijk",torch.einsum(
                    "ij,ik->ijk",phi_x,dd_phi_y_1D).view(-1,nw**2),phi_z).view(-1,nw**3)
            dd_phi_z = torch.einsum(
                "ij,ik->ijk",torch.einsum(
                    "ij,ik->ijk",phi_x,phi_y).view(-1,nw**2),dd_phi_z_1D).view(-1,nw**3)
            dd_phi_xy = torch.einsum(
                "ij,ik->ijk",torch.einsum(
                    "ij,ik->ijk",d_phi_x_1D,d_phi_y_1D).view(-1,nw**2),phi_z).view(-1,nw**3)
            dd_phi_xz = torch.einsum(
                "ij,ik->ijk",torch.einsum(
                    "ij,ik->ijk",d_phi_x_1D, phi_y).view(-1,nw**2),d_phi_z_1D).view(-1,nw**3)
            dd_phi_yz = torch.einsum(
                "ij,ik->ijk",torch.einsum(
                    "ij,ik->ijk",phi_x,d_phi_y_1D).view(-1,nw**2),d_phi_z_1D).view(-1,nw**3)
            dd_phi_full = torch.cat(
                (dd_phi_x.unsqueeze(-1),dd_phi_y.unsqueeze(-1),dd_phi_z.unsqueeze(-1),
                 dd_phi_xy.unsqueeze(-1), dd_phi_xz.unsqueeze(-1),dd_phi_yz.unsqueeze(-1)),
                 dim=-1)

        return phi_xyz, d_phi_xyz, dd_phi_full

    def compute_T_1D(self, t, order=0):
        """
        Compute basis functions for 1D input.
        args:
            t: input points
            order: order of derivatives
        returns:
            T: basis functions
            dT: derivatives of basis functions
            ddT: second derivatives of basis functions
            idx_expand: expanded indices
        """
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

        idx_expand = idx.unsqueeze(-1).expand(-1,self.n_func)*self.n_func \
                        + self.i.unsqueeze(0).expand(N,-1)
        T.scatter_(1,idx_expand,T_seg)

        if order > 0:
            dT_seg[:,1:] = self.i[1:]*T_seg[:,:self.n_func-1] * self.n_seg
            dT.scatter_(1,idx_expand,dT_seg)

        if order == 2:
            ddT_seg[:,2:] = self.i[2:] * self.i[1:-1] * T_seg[:,:-2] * self.n_seg**2
            ddT.scatter_(1,idx_expand,ddT_seg)

        return T, dT, ddT, idx_expand

    def compute_C(self):
        """
        Compute constraint matrix.
        returns:
            C: constraint matrix
        """
        if self.n_func > 4:
            C0 = torch.zeros([4+self.n_func-4, 2+self.n_func-4], dtype=torch.float32)
            C0[:self.n_func-4, :self.n_func-4] = torch.eye(self.n_func-4, dtype=torch.float32)
            C0[self.n_func-4:, self.n_func-4:] = torch.tensor([[1, 0, 0, -1],
                                                               [0, 1, 1, 2]], dtype=torch.float32).T
        else:
            C0 = torch.tensor([[1, 0, 0, -1], [0, 1, 1, 2]], dtype=torch.float32).T

        C = torch.zeros([2 + (self.n_seg-1)*C0.shape[0] + self.n_func - 2,
                         2 + (self.n_seg-1)*C0.shape[1] + self.n_func - 2],
                         dtype=torch.float32).to(self.device)
        C[:2, :2] = torch.eye(2, dtype=torch.float32)
        r = 2
        c = 2
        for _ in range(self.n_seg-1):
            C[r:r+C0.shape[0], c:c+C0.shape[1]] = C0
            r += C0.shape[0]
            c += C0.shape[1]
        C[r:, c:] = torch.eye(self.n_func - 2, dtype=torch.float32)
        return C

    def compute_B(self):
        """
        Computes Bernstein coefficient matrix.
        returns:
            B: Bernstein coefficient matrix
        """
        B0 = torch.zeros([self.n_func, self.n_func], dtype=torch.float32)
        for n in range(1, self.n_func+1):
            for i in range(1, self.n_func+1):
                B0[self.n_func-i, n-1] = (-1) ** (self.n_func-i-n) \
                      * -self.binomial(self.n_func-1, i-1) \
                      * self.binomial((self.n_func-1)-(i-1), (self.n_func-1)-(n-1)-(i-1))
        B = torch.kron(torch.eye(self.n_seg, dtype=torch.float32), B0).to(self.device)
        return B

    def binomial(self,n, i):
        """
        Compute binomial coefficient.
        """
        n = torch.tensor(n, dtype=torch.float64,device=self.device)
        i = torch.tensor(i, dtype=torch.float64,device=self.device)
        mask = (n >= 0) & (i >= 0)
        b = torch.where(mask,
                        torch.exp(
                            torch.lgamma(n + 1) - torch.lgamma(i + 1) - torch.lgamma(n - i + 1)),
                        torch.tensor(0.0, dtype=torch.float64,device=self.device))
        return b
