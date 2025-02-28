import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math
from sklearn.metrics.pairwise import cosine_similarity

# Set up figure and 3D axes
def setup_3d_plot(title="3D Visualization"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    return fig, ax

# Sphere and torus generation functions
def generate_sphere(radius=1.0, resolution=30):
    """Generate points on a sphere"""
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(0, np.pi, resolution)
    
    # Create meshgrid
    theta, phi = np.meshgrid(theta, phi)
    
    # Convert to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    return x, y, z

def generate_torus(R=2.0, r=0.5, resolution=30):
    """Generate points on a torus"""
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    
    # Create meshgrid
    theta, phi = np.meshgrid(theta, phi)
    
    # Convert to Cartesian coordinates
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    
    return x, y, z

def plot_sphere(ax, radius=1.0, resolution=30, alpha=0.2, color='blue'):
    """Plot a wireframe sphere"""
    x, y, z = generate_sphere(radius, resolution)
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, rstride=2, cstride=2)
    
    # Set axis limits
    ax.set_xlim([-radius*1.2, radius*1.2])
    ax.set_ylim([-radius*1.2, radius*1.2])
    ax.set_zlim([-radius*1.2, radius*1.2])
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_torus(ax, R=2.0, r=0.5, resolution=30, alpha=0.2, color='green'):
    """Plot a wireframe torus"""
    x, y, z = generate_torus(R, r, resolution)
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, rstride=2, cstride=2)
    
    # Set axis limits
    max_dim = R + r
    ax.set_xlim([-max_dim*1.2, max_dim*1.2])
    ax.set_ylim([-max_dim*1.2, max_dim*1.2])
    ax.set_zlim([-r*1.2, r*1.2])
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# Generate random points
def random_sphere_points(n_points=100, radius=1.0):
    """Generate random points on a sphere using rejection sampling"""
    # Generate points in a cube and reject those outside the sphere
    points = []
    while len(points) < n_points:
        # Generate a random point in a cube
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        z = np.random.uniform(-radius, radius)
        
        # Check if it's in the sphere
        if x**2 + y**2 + z**2 <= radius**2:
            # Project to the sphere surface
            norm = np.sqrt(x**2 + y**2 + z**2)
            points.append([x * radius / norm, y * radius / norm, z * radius / norm])
    
    return np.array(points)

def random_torus_points(n_points=100, R=2.0, r=0.5):
    """Generate random points on a torus"""
    points = []
    for _ in range(n_points):
        # Generate random angles
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        
        # Convert to Cartesian coordinates
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        
        points.append([x, y, z])
    
    return np.array(points)

# Visualization comparing sphere and torus distances
def compare_distance_metrics():
    """Visualize how distances work on sphere vs torus"""
    # Setup plots
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Sphere distance visualization
    ax1 = fig.add_subplot(131, projection='3d')
    plot_sphere(ax1, radius=1.0)
    
    # Generate a reference point on the sphere
    ref_point = np.array([1.0, 0.0, 0.0])  # Point on the sphere
    
    # Generate points with varying distances from reference
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    
    distances = []
    points = []
    
    for t in theta:
        for p in phi:
            x = np.sin(p) * np.cos(t)
            y = np.sin(p) * np.sin(t)
            z = np.cos(p)
            
            point = np.array([x, y, z])
            # Angular distance (angle between vectors)
            distance = np.arccos(np.clip(np.dot(ref_point, point), -1.0, 1.0))
            
            distances.append(distance)
            points.append(point)
    
    points = np.array(points)
    distances = np.array(distances)
    
    # Normalize distances to [0, 1] for coloring
    norm_distances = distances / np.max(distances)
    
    # Plot points colored by distance
    sc1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                     c=norm_distances, cmap='viridis', alpha=0.5)
    
    # Plot reference point
    ax1.scatter([ref_point[0]], [ref_point[1]], [ref_point[2]], 
               color='red', s=100, label='Reference Point')
    
    ax1.set_title('Sphere Distance from Reference')
    plt.colorbar(sc1, ax=ax1, label='Normalized Distance')
    
    # 2. Torus distance visualization
    ax2 = fig.add_subplot(132, projection='3d')
    plot_torus(ax2, R=2.0, r=0.5)
    
    # Generate a reference point on the torus
    ref_theta, ref_phi = 0.0, 0.0
    R, r = 2.0, 0.5
    ref_point_torus = np.array([
        (R + r * np.cos(ref_phi)) * np.cos(ref_theta),
        (R + r * np.cos(ref_phi)) * np.sin(ref_theta),
        r * np.sin(ref_phi)
    ])
    
    # Generate points on the torus
    theta = np.linspace(0, 2*np.pi, 30)
    phi = np.linspace(0, 2*np.pi, 30)
    
    torus_distances = []
    torus_points = []
    
    for t in theta:
        for p in phi:
            x = (R + r * np.cos(p)) * np.cos(t)
            y = (R + r * np.cos(p)) * np.sin(t)
            z = r * np.sin(p)
            
            point = np.array([x, y, z])
            
            # Euclidean distance in 3D space
            distance = np.linalg.norm(ref_point_torus - point)
            
            # Alternatively, "intrinsic" distance on the torus
            # dt = min(abs(ref_theta - t), 2*np.pi - abs(ref_theta - t))
            # dp = min(abs(ref_phi - p), 2*np.pi - abs(ref_phi - p))
            # distance = np.sqrt((R * dt)**2 + (r * dp)**2)
            
            torus_distances.append(distance)
            torus_points.append(point)
    
    torus_points = np.array(torus_points)
    torus_distances = np.array(torus_distances)
    
    # Normalize distances for coloring
    norm_torus_distances = torus_distances / np.max(torus_distances)
    
    # Plot points colored by distance
    sc2 = ax2.scatter(torus_points[:, 0], torus_points[:, 1], torus_points[:, 2], 
                     c=norm_torus_distances, cmap='viridis', alpha=0.5)
    
    # Plot reference point
    ax2.scatter([ref_point_torus[0]], [ref_point_torus[1]], [ref_point_torus[2]], 
               color='red', s=100, label='Reference Point')
    
    ax2.set_title('Torus Euclidean Distance from Reference')
    plt.colorbar(sc2, ax=ax2, label='Normalized Distance')
    
    # 3. Torus intrinsic distance visualization
    ax3 = fig.add_subplot(133, projection='3d')
    plot_torus(ax3, R=2.0, r=0.5)
    
    # Calculate intrinsic distances on the torus
    intrinsic_distances = []
    
    for i, (t, p) in enumerate(zip(np.arctan2(torus_points[:,1], torus_points[:,0]), 
                                  np.arcsin(torus_points[:,2]/r))):
        # Convert back to angles for intrinsic distance
        dt = min(abs(ref_theta - t), 2*np.pi - abs(ref_theta - t))
        dp = min(abs(ref_phi - p), 2*np.pi - abs(ref_phi - p))
        
        # Intrinsic distance on the torus surface
        intrinsic_dist = np.sqrt((R * dt)**2 + (r * dp)**2)
        intrinsic_distances.append(intrinsic_dist)
    
    intrinsic_distances = np.array(intrinsic_distances)
    norm_intrinsic_distances = intrinsic_distances / np.max(intrinsic_distances)
    
    # Plot points colored by intrinsic distance
    sc3 = ax3.scatter(torus_points[:, 0], torus_points[:, 1], torus_points[:, 2], 
                     c=norm_intrinsic_distances, cmap='viridis', alpha=0.5)
    
    # Plot reference point
    ax3.scatter([ref_point_torus[0]], [ref_point_torus[1]], [ref_point_torus[2]], 
               color='red', s=100, label='Reference Point')
    
    ax3.set_title('Torus Intrinsic Distance from Reference')
    plt.colorbar(sc3, ax=ax3, label='Normalized Distance')
    
    plt.tight_layout()
    return fig

# Visualization of how vectors get mapped to the sphere vs. torus
def visualize_vector_mapping():
    """Visualize how vectors in ambient space map to sphere vs torus"""
    # Generate random vectors in ambient space
    ambient_vectors = np.random.randn(50, 3)  # 50 random 3D vectors
    
    # Map to sphere (normalize)
    sphere_vectors = ambient_vectors / np.linalg.norm(ambient_vectors, axis=1)[:, np.newaxis]
    
    # Map to torus (simple mapping for visualization)
    R, r = 2.0, 0.5
    torus_vectors = []
    
    for v in ambient_vectors:
        # Normalize to get point on sphere
        v_norm = v / np.linalg.norm(v)
        
        # Convert to spherical coordinates
        theta = np.arctan2(v_norm[1], v_norm[0])
        phi = np.arccos(np.clip(v_norm[2], -1.0, 1.0))
        
        # Map spherical coordinates to toroidal coordinates
        # (this is just one possible mapping)
        toroidal_theta = theta
        toroidal_phi = phi % (2 * np.pi) # Map phi to [0, 2π)
        
        # Convert back to Cartesian, but on torus
        x = (R + r * np.cos(toroidal_phi)) * np.cos(toroidal_theta)
        y = (R + r * np.cos(toroidal_phi)) * np.sin(toroidal_theta)
        z = r * np.sin(toroidal_phi)
        
        torus_vectors.append([x, y, z])
    
    torus_vectors = np.array(torus_vectors)
    
    # Setup plots
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Original vectors
    ax1 = fig.add_subplot(131, projection='3d')
    # Plot unit sphere for reference
    plot_sphere(ax1, radius=1.0, alpha=0.1)
    
    # Plot ambient vectors
    for i, v in enumerate(ambient_vectors):
        # Plot normalized vectors for better visualization
        v_norm = v / np.linalg.norm(v) * 0.8  # Scale to 80% of unit sphere
        ax1.quiver(0, 0, 0, v_norm[0], v_norm[1], v_norm[2], 
                  color=plt.cm.viridis(i/len(ambient_vectors)), alpha=0.7)
    
    ax1.set_title('Original Vectors (Normalized)')
    
    # 2. Sphere mapping
    ax2 = fig.add_subplot(132, projection='3d')
    plot_sphere(ax2)
    
    # Plot points on sphere
    ax2.scatter(sphere_vectors[:, 0], sphere_vectors[:, 1], sphere_vectors[:, 2], 
               c=range(len(sphere_vectors)), cmap='viridis', alpha=0.7)
    
    ax2.set_title('Vectors Mapped to Sphere')
    
    # 3. Torus mapping
    ax3 = fig.add_subplot(133, projection='3d')
    plot_torus(ax3)
    
    # Plot points on torus
    ax3.scatter(torus_vectors[:, 0], torus_vectors[:, 1], torus_vectors[:, 2], 
               c=range(len(torus_vectors)), cmap='viridis', alpha=0.7)
    
    ax3.set_title('Vectors Mapped to Torus')
    
    plt.tight_layout()
    return fig

# Visualize dot product behavior
def visualize_dot_products():
    """Visualize how dot products behave differently on sphere vs torus"""
    # Generate pairs of random points
    n_pairs = 1000
    
    # Points on the sphere
    sphere_points1 = random_sphere_points(n_pairs)
    sphere_points2 = random_sphere_points(n_pairs)
    
    # Points on the torus
    R, r = 2.0, 0.5
    torus_points1 = random_torus_points(n_pairs, R, r)
    torus_points2 = random_torus_points(n_pairs, R, r)
    
    # Calculate dot products for sphere points
    # For spherical points, dot product directly relates to cosine of angle
    sphere_dots = np.sum(sphere_points1 * sphere_points2, axis=1)
    
    # For torus, we calculate the angular distances in the two directions
    torus_dots = []
    for p1, p2 in zip(torus_points1, torus_points2):
        # Convert to angular coordinates
        theta1 = np.arctan2(p1[1], p1[0])
        phi1 = np.arcsin(p1[2]/r)
        
        theta2 = np.arctan2(p2[1], p2[0])
        phi2 = np.arcsin(p2[2]/r)
        
        # Calculate minimum angular distances
        dtheta = min(abs(theta1 - theta2), 2*np.pi - abs(theta1 - theta2))
        dphi = min(abs(phi1 - phi2), 2*np.pi - abs(phi1 - phi2))
        
        # Simple product of cosines as a demonstration
        # (not necessarily the best metric for torus)
        torus_dot = np.cos(dtheta) * np.cos(dphi)
        torus_dots.append(torus_dot)
    
    torus_dots = np.array(torus_dots)
    
    # Plot histograms of dot products
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(sphere_dots, bins=30, alpha=0.7)
    ax1.set_title('Dot Products on Sphere')
    ax1.set_xlabel('Dot Product Value')
    ax1.set_ylabel('Frequency')
    
    ax2.hist(torus_dots, bins=30, alpha=0.7)
    ax2.set_title('Dot Products on Torus (Angular)')
    ax2.set_xlabel('Dot Product Value')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig

# Animate points on sphere vs torus
def animate_movement():
    """Create animation of points moving on sphere vs torus"""
    # Initialize points
    n_points = 20
    sphere_points = random_sphere_points(n_points)
    torus_points = random_torus_points(n_points)
    
    # Setup plot
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot sphere and torus wireframes
    plot_sphere(ax1)
    plot_torus(ax2)
    
    # Initialize scatter plots
    scatter_sphere = ax1.scatter([], [], [], c=range(n_points), cmap='viridis')
    scatter_torus = ax2.scatter([], [], [], c=range(n_points), cmap='viridis')
    
    ax1.set_title('Movement on Sphere')
    ax2.set_title('Movement on Torus')
    
    # Animation update function
    def update(frame):
        # Move points on sphere (along great circles)
        new_sphere_points = []
        for p in sphere_points:
            # Create a random rotation axis
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            
            # Small rotation matrix around axis
            angle = 0.05  # Small angle for smooth movement
            c = np.cos(angle)
            s = np.sin(angle)
            x, y, z = axis
            rotation_matrix = np.array([
                [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
                [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
                [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
            ])
            
            # Apply rotation
            p_new = rotation_matrix @ p
            new_sphere_points.append(p_new)
        
        # Move points on torus (along toroidal and poloidal directions)
        R, r = 2.0, 0.5
        new_torus_points = []
        for p in torus_points:
            # Convert to angular coordinates
            theta = np.arctan2(p[1], p[0])
            phi = np.arcsin(p[2]/r)
            
            # Small random adjustments to angles
            theta += np.random.uniform(-0.1, 0.1)
            phi += np.random.uniform(-0.1, 0.1)
            
            # Convert back to Cartesian on torus
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)
            
            new_torus_points.append([x, y, z])
        
        # Update points
        sphere_points[:] = new_sphere_points
        torus_points[:] = new_torus_points
        
        # Update scatter plots
        scatter_sphere._offsets3d = (
            [p[0] for p in sphere_points],
            [p[1] for p in sphere_points],
            [p[2] for p in sphere_points]
        )
        
        scatter_torus._offsets3d = (
            [p[0] for p in torus_points],
            [p[1] for p in torus_points],
            [p[2] for p in torus_points]
        )
        
        return scatter_sphere, scatter_torus
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=100, blit=True, interval=50)
    
    return ani, fig

# Run all visualizations
def run_visualizations(save_path='.'):
    """Run all visualizations and save figures"""
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Distance metrics
    print("Visualizing distance metrics...")
    fig = compare_distance_metrics()
    fig.savefig(os.path.join(save_path, 'distance_metrics.png'), dpi=300, bbox_inches='tight')
    
    # 2. Vector mapping
    print("Visualizing vector mapping...")
    fig = visualize_vector_mapping()
    fig.savefig(os.path.join(save_path, 'vector_mapping.png'), dpi=300, bbox_inches='tight')
    
    # 3. Dot products
    print("Visualizing dot products...")
    fig = visualize_dot_products()
    fig.savefig(os.path.join(save_path, 'dot_products.png'), dpi=300, bbox_inches='tight')
    
    # 4. Animation
    print("Creating animation...")
    ani, fig = animate_movement()
    ani.save(os.path.join(save_path, 'movement_animation.mp4'), dpi=200)
    
    print(f"All visualizations saved to {save_path}")

# Run the script
if __name__ == '__main__':
    run_visualizations()