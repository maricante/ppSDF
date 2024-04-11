import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import skimage.measure as measure

def mesh_vis(d_hat, cut_x=False, cut_y=False, cut_z=False):
    if cut_x:
        d_hat = np.flip(d_hat, axis=0)
    elif cut_y:
        d_hat = np.flip(d_hat, axis=1)
    else:
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

def slice_vis(d_hat, dd_hat, d_true):
    fig, ax = plt.subplots(3, 2)
    d_hat_list = [d_hat[64, :, :], d_hat[:, 64, :], d_hat[:, :, 64]]
    signed_distance_list = [d_true[64, :, :], d_true[:, 64, :], d_true[:, :, 64]]

    x_i = np.random.randint(0, 128, 50)
    y_i = np.random.randint(0, 128, 50)
    z_i = np.random.randint(0, 128, 50)
    grads_hat = dd_hat[x_i, y_i, z_i, :]

    for i in range(3):
        ax[i,0].set_aspect('equal')
        cs = ax[i,0].contour(d_hat_list[i].T, levels=np.arange(-5, 20, 0.2))
        ax[i,0].contour(d_hat_list[i].T, levels=[0], colors='r', linewidths=3)
        ax[i,0].clabel(cs, inline=True, fontsize=10)
        # plot gradient for this slice
        ax[i,0].quiver(y_i, z_i, grads_hat[:,1], grads_hat[:,2], scale=10, color='r')

        ax[i,1].set_aspect('equal')
        cs = ax[i,1].contour(signed_distance_list[i].T, levels=np.arange(-5, 20, 0.2))
        ax[i,1].contour(signed_distance_list[i].T, levels=[0], colors='r', linewidths=3)
        ax[i,1].clabel(cs, inline=True, fontsize=10)

    plt.show()

    return fig