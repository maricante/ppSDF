import os
import numpy as np
import torch
import time
import skimage
import open3d as o3d
import matplotlib.pyplot as plt
import time
import bfSDF
import vis
import argparse
import copy
import utils
import yaml
import scipy.io


CUR_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, default='035_power_drill')
    parser.add_argument('--n_data', type=int, default=1200)
    parser.add_argument('--n_seg', type=int, default=4)
    parser.add_argument('--prior_sphere_radius', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--qd', type=float, default=1.0)
    parser.add_argument('--qn', type=float, default=1.0)
    parser.add_argument('--qt', type=float, default=2e-2)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--cut_x', type=bool, default=False)
    parser.add_argument('--cut_y', type=bool, default=False)
    parser.add_argument('--cut_z', type=bool, default=True)

    args = parser.parse_args()

    device = args.device
    print(device)

    print(f"Loading {args.object}")
    mesh_path = os.path.join(CUR_DIR, f"ycb/{args.object}/google_512k/nontextured.stl")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    mesh.scale(1.0 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    mesh.translate((-mesh.get_center()[0], -mesh.get_center()[1], -mesh.get_center()[2]))

    # generating test points
    domain = torch.linspace(-1, 1, 128).to(device)
    grid_x, grid_y, grid_z= torch.meshgrid(domain,domain,domain, indexing='ij')
    grid_x, grid_y, grid_z = grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)
    test_points = torch.cat([grid_x, grid_y, grid_z],dim=1).float().to(device)

    # ground truth SDF and numerical gradients
    scene = o3d.t.geometry.RaycastingScene()
    mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(mesh_)
    d = scene.compute_signed_distance(test_points.cpu().numpy()).numpy()
    d = d.reshape(128,128,128)
    d_x = np.gradient(d, axis=0)
    d_y = np.gradient(d, axis=1)
    d_z = np.gradient(d, axis=2)
    d_x = d_x / np.linalg.norm(d_x, axis=0)
    d_y = d_y / np.linalg.norm(d_y, axis=0)
    d_z = d_z / np.linalg.norm(d_z, axis=0)
    dd = np.stack([d_x, d_y, d_z], axis=-1)

    # sampling points and estimated normals for training
    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    pcd = pcd.select_by_index(np.random.choice(np.asarray(pcd.points).shape[0], args.n_data, replace=False))
    downpcd_points_tensor = torch.tensor(np.asarray(pcd.points)).to(device).float()
    downpcd_normals_tensor = torch.tensor(np.asarray(pcd.normals)).to(device).float()

    ## training
    print("Fitting piecewise polynomial SDF...")
    bp_sdf = bfSDF.SDFmodel(n_func=4, 
                            n_seg=args.n_seg, 
                            qd = args.qd,
                            qn = args.qn,
                            qt = args.qt,
                            sigma=args.sigma,
                            device=device)
    if os.path.exists(f"priors/sphere_weights_4fun_{args.n_seg}seg.pt"):
        bp_sdf.w = torch.load(f"priors/sphere_weights_4fun_{args.n_seg}seg.pt").to(device)
    else:
        bp_sdf.init_w_sphere(radius=args.prior_sphere_radius, center=torch.tensor(pcd.get_center()).reshape(1, 3))

    # visualizing SDF and mesh prior
    d_hat, _ = bp_sdf.get_sdf(test_points, order=1)
    verts, faces, _, _ = skimage.measure.marching_cubes(
        d_hat.cpu().numpy(), level=0.0, spacing=np.array([(bp_sdf.domain_max-bp_sdf.domain_min)/bp_sdf.nbDim] * 3)
    )
    verts = verts - [1,1,1]
    prior_mesh = o3d.geometry.TriangleMesh()
    prior_mesh.vertices = o3d.utility.Vector3dVector(verts)
    prior_mesh.triangles = o3d.utility.Vector3iVector(faces)
    prior_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([pcd,
                                       prior_mesh,
                                       mesh], point_show_normal=True)

    i = 1
    downpcd_normals_tensor_ = downpcd_normals_tensor.split(args.batch_size, dim=0)
    downpcd_points_tensor_ = downpcd_points_tensor.split(args.batch_size, dim=0)
    print("Batch size: ", args.batch_size)
    for p, n in zip(downpcd_points_tensor_, downpcd_normals_tensor_):
        print("batch: ", i, "/", len(downpcd_points_tensor_))
        bp_sdf.update_pos(p.to(bp_sdf.device).float())
        bp_sdf.update_grad(p.to(bp_sdf.device).float(),
                           n.to(bp_sdf.device).float())
        bp_sdf.regularize_ray(p.to(bp_sdf.device).float(),
                              n.to(bp_sdf.device).float())
        i += 1

    # reconstructing SDF
    print("Reconstructing SDF...")
    with torch.no_grad():
        d_hat, dd_hat = bp_sdf.get_sdf(test_points, order=1)
    d_hat = d_hat.cpu().numpy()

    # visualization
    print("Visualizing SDF...")
    # 3D visualization of reconstructed SDF and mesh
    vis.mesh_vis(d_hat, cut_x=args.cut_x, cut_y=args.cut_y, cut_z=args.cut_z)
    # vizualization of a slice of the reconstructed SDF and projected gradients
    vis.slice_vis(d_hat, dd_hat.cpu().numpy(), d)

    # save results
    if args.save:
        if not os.path.exists("results"):
            os.makedirs("results")
        scipy.io.savemat(f"results/{args.object}_bfSDF_{args.n_data}.mat", {'d': d_hat})

    print("Evaluating SDF...")
    utils.print_eval(torch.tensor(d_hat).to(device), torch.tensor(d).to(device), dd_hat, torch.tensor(dd).to(device), string='bfSDF')
