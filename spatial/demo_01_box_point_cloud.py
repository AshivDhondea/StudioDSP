import argparse
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# Update matplotlib parameters to use LateX.
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
plt.rc('font',family='serif')
plt.rc('font',family='DejaVu Sans')


if __name__ == '__main__':
    # Parse input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "-length [m]", type=float,required=True, help="Length of cuboid in metres")
    parser.add_argument("-w", "--width [m]", type=float, required=True, help="Width of cuboid in metres")
    parser.add_argument("-h", "--height [m]", type=float, required=True, help="Height of cuboid in metres")
    parser.add_argument("-r", "--resolution [m]", type=float, required=False, default=0.05, help="Spatial resolution in metres")
    parser.add_argument("-p", "--PLY file", type=str, required=False, help="PLY file name if you want to save the point cloud.")

    args = parser.parse_args()
    length = args.l
    width = args.w
    height = args.h
    resolution = args.r
    ply_file = args.p

    if ply_file:
        if not ply_file.endswith('.ply'):
            ftype_err = f"File type error in {ply_file}. Expected .ply"
            raise ValueError(ftype_err)

    # Create arrays for each dimension with the specified dimensions sampled at the specified resolution.
    x = np.arange(start=0., stop=length, step=resolution, dtype=np.float64) # array of length d_x
    print(f"d_x {np.shape(x)=}")
    y = np.arange(start=0., stop=width, step=resolution, dtype=np.float64) # array of length d_y
    print(f"d_y {np.shape(y)=}")
    z = np.arange(start=0., stop=height, step=resolution, dtype=np.float64) # array of length d_z
    print(f"d_z {np.shape(z)=}")

    #TODO: Clarify how faces are labelled.

    # Faces 1 and 3
    # We create a rectangular grid of x and y coordinates for the given length and width.
    xy_x, xy_y = np.meshgrid(x, y) # Both xy_x and xy_y have shape d_x by d_y
    print(f"{np.shape(xy_x)=}")
    print(f"{np.shape(xy_y)=}")
    #TODO: Explain how the dimensions work.
    # We create the corresponding grid of z coordinates for face 1, which is aligned with the z=0 plane.
    xy_z_1 = np.zeros_like(xy_x)
    print(f"{np.shape(xy_z_1)=}")

    # Visualize face 1 as a point cloud.
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    plt.title(f'3D Box - Scatter plot - Face 1 (blue) with length ${length}$m \& width ${width}$m')
    ax.scatter(xy_x, xy_y, xy_z_1, marker=r'o', color='blue')
    ax.set_xlabel(r'$x\mathrm{[m]}$')
    ax.set_ylabel(r'$y\mathrm{[m]}$')
    ax.set_zlabel(r'$z\mathrm{[m]}$')
    ax.view_init(32, 22.5)  # azimuth and elevation angles for viewing.
    plt.tight_layout()
    plt.show()

    # Visualize face 1 as a surface.
    fig = plt.figure(2)
    ax = fig.add_subplot(projection='3d')
    plt.title(f'3D Box - Surface plot - Face 1 (blue) with length ${length}$m \& width ${width}$m')
    ax.plot_surface(xy_x, xy_y, xy_z_1, alpha=0.5, color='blue')
    ax.set_xlabel(r'$x\mathrm{[m]}$')
    ax.set_ylabel(r'$y\mathrm{[m]}$')
    ax.set_zlabel(r'$z\mathrm{[m]}$')
    ax.view_init(32, 22.5) # azimuth and elevation angles for viewing.
    plt.tight_layout()
    plt.show()

    # Get the corresponding z-coordinates for the points in face 3.
    xy_z_3 = height*np.ones_like(xy_x)

    # Visualize the point clouds for face 1 and face 3.
    fig = plt.figure(3)
    ax = fig.add_subplot(projection='3d')
    plt.title(f'3D Box - Point Cloud - Face 1 (blue) \& face 3 (red) with length ${length}$m \& width ${width}$m')
    ax.scatter(xy_x, xy_y, xy_z_1, marker=r'o', color='blue')
    ax.scatter(xy_x, xy_y, xy_z_3, marker=r'^', color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(32, 22.5)  # azimuth and elevation angles for viewing.
    plt.tight_layout()
    plt.show()

    # Collect the x, y and z coordinates arrays into an array of shape d_x * d_y * d_z x 3
    face_1_arr = np.vstack((np.reshape(xy_x, -1),
                            np.reshape(xy_y, -1),
                            np.reshape(xy_z_1, -1),
                            ))
    print(f"{np.shape(face_1_arr)=}")

    # Create a 3D point cloud using Open3D
    face_1_pcd = o3d.geometry.PointCloud()
    #TODO: explain why we need to transpose here.
    face_1_pcd.points = o3d.utility.Vector3dVector(face_1_arr.T)
    # The colour is set in RGB space in the range [0, 1]
    face_1_pcd.paint_uniform_color([1, 0.706, 0])

    # Calculate normals to improve illumination in data visualization
    face_1_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05,
            max_nn=16
        ),
        fast_normal_computation=True,
    )
    o3d.visualization.draw_geometries(
        geometry_list=[face_1_pcd],
        window_name='Face 1 Point Cloud',
        point_show_normal=False,)

    xz_x, xz_z = np.meshgrid(x, z)
    xy_z_1 = np.zeros_like(xy_x)

    # Explain that there are duplicate points in the point cloud which arising from the edge where
    # two faces meet.



