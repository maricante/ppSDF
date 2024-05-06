import os
import numpy as np
import torch
import skimage
import open3d as o3d
from ippSDF import ippSDF
import vis
import argparse
import utils
import scipy.io


def ground_truth(mesh, test_points):
    """
    Compute ground truth SDF and numerical gradients.
    args:
        mesh: input mesh
        test_points: test points
    """
    scene = o3d.t.geometry.RaycastingScene()
    mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(mesh_)
    d = scene.compute_signed_distance(test_points).numpy()
    d = d.reshape(128,128,128)
    d_x = np.gradient(d, axis=0)
    d_y = np.gradient(d, axis=1)
    d_z = np.gradient(d, axis=2)
    d_x = d_x / np.linalg.norm(d_x, axis=0)
    d_y = d_y / np.linalg.norm(d_y, axis=0)
    d_z = d_z / np.linalg.norm(d_z, axis=0)
    dd = np.stack([d_x, d_y, d_z], axis=-1)
    return d, dd

def sample_pcd(mesh, n_data):
    """
    Sample points and normals from mesh.
    args:
        mesh: input mesh
        n_data: number of samples
    """
    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    pcd = pcd.select_by_index(np.random.choice(np.asarray(pcd.points).shape[0],
                                               n_data, replace=False))
    return pcd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training options
    parser.add_argument('--n_seg', type=int, default=5)
    parser.add_argument('--qd', type=float, default=1.0)
    parser.add_argument('--qn', type=float, default=1.0)
    parser.add_argument('--qt', type=float, default=2e-2)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=bool, default=False)

    # test object and number of samples
    parser.add_argument('--object', type=str, default='035_power_drill')
    parser.add_argument('--n_data', type=int, default=1000)

    # visualization options
    parser.add_argument('--cut_x', type=bool, default=False)
    parser.add_argument('--cut_y', type=bool, default=False)
    parser.add_argument('--cut_z', type=bool, default=True)

    args = parser.parse_args()
    device = args.device
    print(device)

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))

    # loading mesh
    print(f"Loading {args.object}")
    mesh_path = os.path.join(CUR_DIR, f"ycb/{args.object}/google_512k/nontextured.stl")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.scale(1.0 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
               center=mesh.get_center())
    mesh.translate((-mesh.get_center()[0], -mesh.get_center()[1], -mesh.get_center()[2]))

    # generating test points
    domain = torch.linspace(-1, 1, 128).to(device)
    grid_x, grid_y, grid_z= torch.meshgrid(domain,domain,domain, indexing='ij')
    grid_x, grid_y, grid_z = grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)
    test_points = torch.cat([grid_x, grid_y, grid_z],dim=1).float().to(device)

    # ground truth SDF and numerical gradients
    d, dd = ground_truth(mesh, test_points.cpu().numpy())

    # sampling points and estimated normals for training
    pcd = sample_pcd(mesh, args.n_data)
    pcd_points = torch.tensor(np.asarray(pcd.points)).to(device).float()
    pcd_normals = torch.tensor(np.asarray(pcd.normals)).to(device).float()

    ## initializing cubic piecewise polynomial SDF
    print("Initializing piecewise polynomial SDF...")
    model = ippSDF(n_seg=args.n_seg,
                   qd = args.qd, qn = args.qn, qt = args.qt,
                   sigma=args.sigma, device=device)
    if os.path.exists(f"priors/sphere_weights_4fun_{args.n_seg}seg.pt"):
        model.w = torch.load(f"priors/sphere_weights_4fun_{args.n_seg}seg.pt").to(device)
    else:
        model.init_w_sphere(radius=0.4, center=torch.tensor(pcd.get_center()).reshape(1, 3))
        torch.save(model.w, f"priors/sphere_weights_4fun_{args.n_seg}seg.pt")

    # visualizing prior and sampled points
    print("Visualizing prior...")
    d_hat, dd_hat = model.forward(test_points, order=1)
    d_hat = d_hat.reshape(model.grid_res, model.grid_res, model.grid_res)
    dd_hat = dd_hat.reshape(model.grid_res, model.grid_res, model.grid_res, 3)
    verts, faces, _, _ = skimage.measure.marching_cubes(
        d_hat.cpu().numpy(),
        level=0.0,
        spacing=np.array([(model.domain_max-model.domain_min)/model.grid_res] * 3)
    )
    verts = verts - [1,1,1]
    prior_mesh = o3d.geometry.TriangleMesh()
    prior_mesh.vertices = o3d.utility.Vector3dVector(verts)
    prior_mesh.triangles = o3d.utility.Vector3iVector(faces)
    prior_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([pcd,
                                       prior_mesh,
                                       mesh], point_show_normal=True)

    # training SDF
    pcd_normals_batches = pcd_normals.split(args.batch_size, dim=0)
    pcd_points_batches = pcd_points.split(args.batch_size, dim=0)
    print("Fitting SDF...")
    i = 1
    for p, n in zip(pcd_points_batches, pcd_normals_batches):
        print("batch: ", i, "/", len(pcd_points_batches))
        model.update_pos(p.to(model.device).float())
        model.update_grad(p.to(model.device).float(),
                          n.to(model.device).float())
        model.regularize_ray(p.to(model.device).float(),
                             n.to(model.device).float())
        i += 1

    # reconstructing SDF
    print("Reconstructing SDF...")
    with torch.no_grad():
        d_hat, dd_hat = model.forward(test_points, order=1)
        d_hat = d_hat.reshape(model.grid_res, model.grid_res, model.grid_res)
        dd_hat = dd_hat.reshape(model.grid_res, model.grid_res, model.grid_res, 3)
    d_hat = d_hat.cpu().numpy()

    # visualization
    print("Visualizing SDF...")
    vis.mesh_vis(d_hat, cut_x=args.cut_x, cut_y=args.cut_y, cut_z=args.cut_z)
    vis.grad_vis(d_hat, dd_hat)
    vis.slice_vis(d_hat, d,
                  title=f'ppSDF ({args.n_data} samples, {args.n_seg} segments, cubic polynomials)')

    # save results
    if args.save:
        if not os.path.exists("results"):
            os.makedirs("results")
        scipy.io.savemat(f"results/{args.object}_bfSDF_{args.n_data}.mat", {'d': d_hat})

    print("Evaluating SDF...")
    utils.print_eval(torch.tensor(d_hat).to(device), torch.tensor(d).to(device),
        dd_hat, torch.tensor(dd).to(device),
        string=f'ppSDF ({args.n_data} samples, {args.n_seg} segments, cubic polynomials)')
