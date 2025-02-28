import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
import os
import time
from nGPT_pytorch import nGPT, l2norm, NormLinear, Attention, FeedForward

# Check CUDA availability and handle gracefully
if torch.cuda.is_available():
    # Set environment variable to handle CUDA blocking
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    try:
        # Try to get a device - with a timeout to avoid hanging
        device = torch.device('cuda')
        # Test the device with a small tensor
        test_tensor = torch.zeros(1, device=device)
        print(f"CUDA is available and working. Using device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"CUDA error: {e}")
        print("Falling back to CPU...")
        device = torch.device('cpu')
else:
    print("CUDA is not available. Using CPU...")
    device = torch.device('cpu')

# Create a small model for debugging
print("Initializing mini model...")
mini_model = nGPT(
    num_tokens=256,
    dim=64,
    depth=2,
    heads=2,
    dim_head=32,
    tied_embedding=True,
    add_value_residual=True,
    attn_norm_qk=True,
    manual_norm_weights=False
).to(device)
print("Model initialized successfully!")

# Helper functions for visualization and analysis
def visualize_embeddings(embeddings, title="Embeddings"):
    """
    Project and visualize high-dimensional embeddings in 2D or 3D
    """
    from sklearn.decomposition import PCA
    
    # Get embeddings as numpy array
    if isinstance(embeddings, torch.Tensor):
        emb_np = embeddings.detach().cpu().numpy()
    else:
        emb_np = np.array(embeddings)
    
    # Determine dimensionality for visualization
    if emb_np.shape[1] > 3:
        # Use PCA to reduce to 3D
        pca = PCA(n_components=3)
        emb_reduced = pca.fit_transform(emb_np)
    else:
        emb_reduced = emb_np
    
    # Create figure
    if emb_reduced.shape[1] == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(emb_reduced[:, 0], emb_reduced[:, 1], emb_reduced[:, 2], alpha=0.6)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.scatter(emb_reduced[:, 0], emb_reduced[:, 1], alpha=0.6)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
    
    ax.set_title(title)
    plt.tight_layout()
    
    return fig

def analyze_norm_distribution(tensors, names):
    """
    Analyze the distribution of norms for a list of tensors
    """
    fig, axs = plt.subplots(1, len(tensors), figsize=(15, 5))
    
    for i, (tensor, name) in enumerate(zip(tensors, names)):
        if isinstance(tensor, torch.Tensor):
            norms = torch.norm(tensor, dim=-1).detach().cpu().numpy()
        else:
            norms = np.linalg.norm(tensor, axis=-1)
        
        # Ensure we have a valid axis even with a single tensor
        if len(tensors) > 1:
            ax = axs[i]
        else:
            ax = axs
        
        # Safe histogram creation - handle uniform/narrow distributions
        min_val = np.min(norms)
        max_val = np.max(norms)
        range_val = max_val - min_val
        
        if range_val < 1e-10:
            # All values are essentially the same - use a single bar
            ax.bar([min_val], [len(norms)], width=0.01, alpha=0.7)
            ax.set_xlim(min_val - 0.1, min_val + 0.1)
        else:
            # Determine an appropriate number of bins
            # For normalized values (close to 1), we need fewer bins
            # We add a small buffer to ensure we don't hit the exact edge case
            num_bins = max(3, min(10, int(range_val * 100) + 1))
            
            # Create custom bin edges that span the data range plus a small buffer
            bin_edges = np.linspace(min_val - 0.001*range_val, 
                                   max_val + 0.001*range_val, 
                                   num_bins + 1)
            
            # Create the histogram
            ax.hist(norms, bins=bin_edges, alpha=0.7)
        
        ax.set_title(f'{name} Norm Distribution')
        ax.set_xlabel('Norm Value')
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig

def trace_forward_pass(model, input_ids):
    """
    Trace through a forward pass, capturing intermediate states for analysis
    """
    # Store intermediate activations
    activations = {}
    
    # Register hooks to capture activations
    hooks = []
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0]
            else:
                activations[name] = output
        return hook
    
    # Attach hooks to various components of the model
    for i, (attn, ff) in enumerate(model.layers):
        hooks.append(attn.fn.register_forward_hook(get_activation(f'attn_{i}')))
        hooks.append(ff.fn.register_forward_hook(get_activation(f'ff_{i}')))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return outputs, activations

def debug_l2norm_function():
    """
    Visualize how the l2norm function works
    """
    # Create random tensors with different properties
    tensors = [
        torch.randn(10, 64),                 # Random with mean 0, std 1
        torch.ones(10, 64) * 0.5,            # Uniform 0.5
        torch.randn(10, 64) * 5              # Random with larger scale
    ]
    
    # Apply l2norm with different parameters
    results = []
    for t in tensors:
        # Apply different l2norm configurations
        l2norm_basic = l2norm(t)
        l2norm_eps_01 = l2norm(t, norm_eps=0.1)
        l2norm_groups_2 = l2norm(t, groups=2)
        
        # Store results
        results.append({
            'original': t,
            'l2norm_basic': l2norm_basic,
            'l2norm_eps_01': l2norm_eps_01,
            'l2norm_groups_2': l2norm_groups_2
        })
    
    # Analyze and visualize the norms
    figs = []
    for i, res in enumerate(results):
        # Extract tensors and names for visualization
        tensors_list = list(res.values())
        names = list(res.keys())
        
        # Create visualization
        try:
            fig = analyze_norm_distribution(tensors_list, names)
            figs.append(fig)
            
            # Print some summary statistics for verification
            for name, tensor in zip(names, tensors_list):
                norms = torch.norm(tensor, dim=-1).detach().cpu().numpy()
                print(f"Tensor {i}, {name}: Mean norm = {norms.mean():.4f}, "
                      f"Min = {norms.min():.4f}, Max = {norms.max():.4f}, "
                      f"Std = {norms.std():.4f}")
        except Exception as e:
            print(f"Error visualizing tensor {i}: {e}")
    
    return figs

def debug_residual_interpolation(model, input_ids):
    """
    Visualize how the residual interpolation works
    """
    # Trace a forward pass to get intermediate activations
    _, activations = trace_forward_pass(model, input_ids)
    
    # Extract token embeddings
    token_embed = model.token_embed.weight[input_ids]
    
    # Track how vectors evolve through the network
    vector_evolution = {'token_embed': token_embed}
    
    # Compute similarities between layers
    similarities = {}
    
    keys = sorted([k for k in activations.keys()])
    for i, key in enumerate(keys):
        if i > 0:
            # Compute cosine similarity between consecutive layers
            prev_act = vector_evolution[keys[i-1]]
            curr_act = activations[key]
            
            # Reshape to 2D
            prev_flat = rearrange(prev_act, 'b ... d -> (b ...) d')
            curr_flat = rearrange(curr_act, 'b ... d -> (b ...) d')
            
            # Normalize
            prev_norm = l2norm(prev_flat)
            curr_norm = l2norm(curr_flat)
            
            # Compute cosine similarity
            sim = (prev_norm * curr_norm).sum(dim=-1)
            similarities[f"{keys[i-1]}_to_{key}"] = sim.mean().item()
        
        vector_evolution[key] = activations[key]
    
    return similarities

def debug_attention_mechanism(model, input_ids):
    """
    Debug the attention mechanism
    """
    # Get the first attention layer 
    attn_layer = model.layers[0][0].fn
    
    # Forward pass to get Q, K, V matrices
    with torch.no_grad():
        tokens = model.token_embed.weight[input_ids]
        q, k, v = attn_layer.to_q(tokens), attn_layer.to_k(tokens), attn_layer.to_v(tokens)
        
        # Split heads
        q, k, v = map(attn_layer.split_heads, (q, k, v))
        
        try:
            # Apply rotary embeddings if needed
            q = attn_layer.rotary_embed.rotate_queries_or_keys(q) 
            k = attn_layer.rotary_embed.rotate_queries_or_keys(k)
        except Exception as e:
            # If rotary_emb is not available or not a method
            print(f"Note: rotary embeddings not applied - {e}")
        
        # Normalize if needed
        if hasattr(attn_layer, 'norm_qk') and attn_layer.norm_qk:
            q_normalized = l2norm(q)
            k_normalized = l2norm(k)
        else:
            q_normalized = q
            k_normalized = k
        
        # Scale q
        try:
            q_scaled = q_normalized * rearrange(attn_layer.qk_scale(), '(h d) -> h 1 d', h=attn_layer.heads)
        except Exception as e:
            print(f"Error scaling q: {e}")
            q_scaled = q_normalized
        
        # Compute attention scores
        attn_scores = torch.einsum('bhid,bhjd->bhij', q_scaled, k_normalized) / attn_layer.attn_scale
        
        # Apply causal mask if needed
        if attn_layer.causal:
            mask = torch.triu(torch.ones(attn_scores.shape[-2:], device=attn_scores.device), diagonal=1) * -1e9
            attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)
        
        # Apply softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhij,bhjd->bhid', attn_probs, v)
        out = attn_layer.merge_heads(out)
        out = attn_layer.to_out(out)
    
    return {
        'q': q,
        'k': k,
        'v': v,
        'q_normalized': q_normalized,
        'k_normalized': k_normalized,
        'attn_scores': attn_scores,
        'attn_probs': attn_probs,
        'out': out
    }

# Additional functions for toroidal adaptation
def project_to_torus(embeddings, R=2, r=1):
    """
    Project embeddings to torus (for theoretical exploration)
    Torus is parameterized by R (major radius) and r (minor radius)
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # First, normalize to get points on a sphere
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    sphere_points = embeddings / np.maximum(norms, 1e-10)  # Avoid division by zero
    
    # Get two angles from sphere points
    theta = np.arccos(np.clip(sphere_points[:, 2], -1.0, 1.0))  # Polar angle
    phi = np.arctan2(sphere_points[:, 1], sphere_points[:, 0])  # Azimuthal angle
    
    # Map to torus
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    
    return np.column_stack([x, y, z])

def visualize_torus_mapping(embeddings, R=2, r=1):
    """
    Visualize how embeddings map to a torus
    """
    torus_points = project_to_torus(embeddings, R, r)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(torus_points[:, 0], torus_points[:, 1], torus_points[:, 2], alpha=0.6)
    ax.set_title('Embeddings Mapped to Torus')
    plt.tight_layout()
    
    return fig

def visualize_toroidal_distance(embeddings, R=2, r=1):
    """
    Visualize distances in toroidal space vs. spherical space
    """
    # Normalize embeddings to unit sphere
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings
        
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    sphere_points = embeddings_np / np.maximum(norms, 1e-10)  # Avoid division by zero
    
    # Project to torus
    torus_points = project_to_torus(embeddings_np, R, r)
    
    # Compute distances
    n = min(len(embeddings_np), 100)  # Limit to 100 points for visualization
    sphere_dist = np.zeros((n, n))
    torus_dist = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Spherical distance (angular distance)
            dot_product = np.clip(np.dot(sphere_points[i], sphere_points[j]), -1.0, 1.0)
            sphere_dist[i, j] = np.arccos(dot_product)
            
            # Toroidal distance (Euclidean in 3D)
            torus_dist[i, j] = np.linalg.norm(torus_points[i] - torus_points[j])
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    im1 = axs[0].imshow(sphere_dist)
    axs[0].set_title('Spherical Distances')
    plt.colorbar(im1, ax=axs[0])
    
    im2 = axs[1].imshow(torus_dist)
    axs[1].set_title('Toroidal Distances')
    plt.colorbar(im2, ax=axs[1])
    
    plt.tight_layout()
    return fig

# Main debug function
def debug_ngpt(model, input_text=None):
    """
    Main function to debug an nGPT model
    """
    # Set up input
    if input_text is None:
        # Generate random input
        input_ids = torch.randint(0, 256, (2, 16)).to(device)
    else:
        # Convert text to input_ids
        input_ids = torch.tensor([[ord(c) for c in input_text]]).to(device)
    
    print("1. Basic Model Information:")
    print(f"Model Dimension: {model.dim}")
    print(f"Number of Layers: {len(model.layers)}")
    print(f"Token Embedding Shape: {model.token_embed.weight.shape}")
    
    print("\n2. Testing l2norm Function")
    try:
        l2norm_figs = debug_l2norm_function()
        print(f"Created {len(l2norm_figs)} visualizations for l2norm")
    except Exception as e:
        print(f"Error in l2norm testing: {e}")
    
    print("\n3. Analyzing Token Embeddings")
    try:
        # Get a subset of token embeddings for visualization
        token_embeddings = model.token_embed.weight.detach().cpu().numpy()
        print(f"Token Embeddings Shape: {token_embeddings.shape}")
        norms = np.linalg.norm(token_embeddings, axis=1)
        print(f"Token Embeddings Norms Min: {norms.min():.4f}")
        print(f"Token Embeddings Norms Max: {norms.max():.4f}")
        print(f"Token Embeddings Norms Mean: {norms.mean():.4f}")
        print(f"Token Embeddings Norms Std: {norms.std():.4f}")
    except Exception as e:
        print(f"Error analyzing token embeddings: {e}")
    
    print("\n4. Tracing a Forward Pass")
    try:
        outputs, activations = trace_forward_pass(model, input_ids)
        print(f"Number of Captured Activations: {len(activations)}")
        
        # Print summary of activations
        for name, activation in activations.items():
            print(f"  {name}: Shape {activation.shape}, "
                  f"Mean {activation.abs().mean().item():.4f}, "
                  f"Max {activation.abs().max().item():.4f}")
    except Exception as e:
        print(f"Error tracing forward pass: {e}")
        activations = {}
    
    print("\n5. Analyzing Residual Interpolation")
    try:
        similarities = debug_residual_interpolation(model, input_ids)
        for name, sim in similarities.items():
            print(f"  {name}: {sim:.4f}")
    except Exception as e:
        print(f"Error analyzing residual interpolation: {e}")
        similarities = {}
    
    print("\n6. Debugging Attention Mechanism")
    try:
        attn_debug = debug_attention_mechanism(model, input_ids)
        print(f"  Q Shape: {attn_debug['q'].shape}")
        print(f"  K Shape: {attn_debug['k'].shape}")
        print(f"  V Shape: {attn_debug['v'].shape}")
        print(f"  Attention Scores Shape: {attn_debug['attn_scores'].shape}")
        
        # Analyze attention pattern
        attn_probs = attn_debug['attn_probs']
        print(f"  Attention Probs Min: {attn_probs.min().item():.4f}")
        print(f"  Attention Probs Max: {attn_probs.max().item():.4f}")
        print(f"  Attention Entropy: {-(attn_probs * torch.log(attn_probs + 1e-10)).sum(-1).mean().item():.4f}")
    except Exception as e:
        print(f"Error debugging attention mechanism: {e}")
        attn_debug = {}
    
    return {
        'outputs': outputs if 'outputs' in locals() else None,
        'activations': activations,
        'similarities': similarities,
        'attn_debug': attn_debug
    }

# Run debug analysis if this script is executed directly
if __name__ == "__main__":
    try:
        # Run full debugging
        print("Running nGPT debugging...")
        debug_results = debug_ngpt(mini_model, "Hello world!")
        
        print("\n7. Exploring toroidal mapping of embeddings...")
        try:
            token_embeddings = mini_model.token_embed.weight[:100].detach().cpu().numpy()
            torus_fig = visualize_torus_mapping(token_embeddings)
            
            print("\n8. Analyzing distance metrics...")
            distance_fig = visualize_toroidal_distance(token_embeddings)
            
            print("\nDebugging complete!")
            
            # Save some visualizations if needed
            plt.figure(figsize=(10, 8))
            plt.title("Token Embeddings PCA")
            pca_fig = visualize_embeddings(token_embeddings)
            pca_fig.savefig("token_embeddings_pca.png")
            
            torus_fig.savefig("token_embeddings_torus.png")
            distance_fig.savefig("token_embeddings_distances.png")
        except Exception as e:
            print(f"Error in toroidal analysis: {e}")
        
        # Show plots if interactive
        plt.show()
        
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()