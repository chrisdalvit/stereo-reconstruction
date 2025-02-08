import open3d as o3d
import numpy as np
import imageio
import os

def render_and_capture_point_cloud(ply_file, output_gif, num_frames=120):
    """Render a point cloud from a PLY file and capture it as a GIF"""

    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    if not pcd.has_points():
        raise ValueError("The PLY file contains no point data.")

    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Offscreen", visible=False)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.array([255, 255, 255])

    # Generate frames
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    images = []
    for i in range(num_frames):
        angle = 360.0 / num_frames
        R = pcd.get_rotation_matrix_from_xyz((0, np.deg2rad(angle), 0))

        pcd.rotate(R, center=pcd.get_center())
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Capture image
        image_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        vis.capture_screen_image(image_path)
        images.append(imageio.imread(image_path))
    
    vis.destroy_window()
    
    # Create GIF
    imageio.mimsave(output_gif, images, duration=50)
    
    # Cleanup temporary files
    for image_path in images:
        os.remove(image_path)
    os.rmdir(temp_dir)

    print(f"GIF saved as {output_gif}")

# Example usage
render_and_capture_point_cloud("output/opencv_python.ply", "output.gif")
