import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import skimage.measure as measure


def mesh_vis(d_hat,
             cut_x=False, cut_y=False, cut_z=False,
             flip=False):
    """
    Visualize the mesh and level sets of the SDF.
    args:
        d_hat: signed distance field
        cut_x: whether to visualize the x-axis cut
        cut_y: whether to visualize the y-axis cut
        cut_z: whether to visualize the z-axis cut
        flip: whether to flip the SDF for visualization
    """
    if flip and cut_x:
        d_hat = np.flip(d_hat, axis=0)
    elif flip and cut_y:
        d_hat = np.flip(d_hat, axis=1)
    elif flip:
        d_hat = np.flip(d_hat, axis=2)
    verts, faces, _, _ = measure.marching_cubes(d_hat, 0.0)
    rMesh = o3d.geometry.TriangleMesh()
    rMesh.vertices = o3d.utility.Vector3dVector(verts)
    rMesh.triangles = o3d.utility.Vector3iVector(faces)
    rMesh.compute_vertex_normals()
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = np.array([0.3, 0.3, 0.3, 1.0])
    draw_dicts = [{'name': 'zero', 'geometry': rMesh, 'material': mat}]
    if cut_x or cut_y or cut_z:
        if cut_x:
            d_hat_slice = d_hat[:64, :, :]
        elif cut_y:
            d_hat_slice = d_hat[:, :62, :]
        else:
            d_hat_slice = d_hat[:, :, :64]
        levels = [0.04 * i for i in range(1, 6)]
        for l in levels:
            verts, faces, _, _ = measure.marching_cubes(d_hat_slice, l)
            verts = verts - [1,1,1]
            rMesh = o3d.geometry.TriangleMesh()
            rMesh.vertices = o3d.utility.Vector3dVector(verts)
            rMesh.triangles = o3d.utility.Vector3iVector(faces)
            rMesh.compute_vertex_normals()
            if l > 0:
                rMesh.paint_uniform_color([l, 0.0, 0.0])
            elif l == 0:
                rMesh.paint_uniform_color([0.3, 0.3, 0.3])
            else:
                rMesh.paint_uniform_color([0.0, 0.0, -l])
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency"
            mat.base_color = np.array([1, 1, 1, .8])
            if l == 0:
                mat.base_color = np.array([0.3, 0.3, 0.3, 1.0])
            draw_dicts.append({'name': str(l), 'geometry': rMesh, 'material': mat})
    o3d.visualization.draw(draw_dicts, show_skybox=False)

def grad_vis(d_hat, dd_hat):
    """
    Visualize the reconstructed mesh and gradients.
    args:
        d_hat:  signed distance field
        dd_hat: gradients of the signed distance field
    """
    verts, faces, _, _ = measure.marching_cubes(d_hat, 0.0)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    x_i = np.random.randint(0, 128, 50)
    y_i = np.random.randint(0, 128, 50)
    z_i = np.random.randint(0, 128, 50)

    pcd_points = np.stack([x_i, y_i, z_i], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd.paint_uniform_color([0.8, 0.0, 0.0])

    pcd_normals = dd_hat[x_i, y_i, z_i].cpu().numpy()
    pcd_normals = pcd_normals / np.linalg.norm(pcd_normals, axis=1, keepdims=True)
    pcd_normals = o3d.utility.Vector3dVector(pcd_normals)
    pcd.normals = pcd_normals

    o3d.visualization.draw_geometries([pcd,
                                       mesh], point_show_normal=True)


def slice_vis(d_hat, d_true, title='SDF slices'):
    """
    Visualize the slices of the SDF.
    args:
        d_hat:  signed distance field
        d_true: true signed distance field
    """
    fig, ax = plt.subplots(3, 2)

    d_hat = np.flip(d_hat, axis=2)

    d_hat_list = [d_hat[64, :, :], d_hat[:, 64, :], d_hat[:, :, 64]]
    signed_distance_list = [d_true[64, :, :], d_true[:, 64, :], d_true[:, :, 64]]

    for i in range(3):
        ax[i,0].set_aspect('equal')
        im = ax[i,0].imshow(d_hat_list[i].T, cmap='jet',
                            origin='lower', vmin=-0.1, vmax=1.0)
        ax[i,0].contour(d_hat_list[i].T, levels=[0], colors='r', linewidths=3)

        ax[i,1].set_aspect('equal')
        im = ax[i,1].imshow(signed_distance_list[i].T, cmap='jet',
                            origin='lower', vmin=-0.1, vmax=1.0)
        plt.colorbar(im, ax=ax[i,1])
        ax[i,1].contour(signed_distance_list[i].T, levels=[0], colors='r', linewidths=3)

    plt.suptitle(title)
    plt.show()

    return fig
