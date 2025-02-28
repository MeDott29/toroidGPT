import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
from einops import rearrange

# Import the toroidal model
from toroidal_ngpt import ToroidalGPT, toroidal_norm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
SEQUENCE_LENGTH = 64
VALIDATION_EVERY = 100

# Simple synthetic dataset for testing
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=64, vocab_size=256):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Create synthetic sequences with some patterns
        self.data = []
        for _ in range(num_samples):
            # Start with a pattern
            if np.random.random() < 0.5:
                # Repeated pattern
                pattern_len = np.random.randint(1, 5)
                pattern = np.random.randint(0, vocab_size, pattern_len)
                repeats = seq_len // pattern_len + 1
                seq = np.tile(pattern, repeats)[:seq_len]
            else:
                # Random with some structure
                seq = np.random.randint(0, vocab_size, seq_len)
                
                # Add some dependency
                for i in range(1, seq_len):
                    if np.random.random() < 0.3:
                        # Make this token depend on previous token
                        seq[i] = (seq[i-1] + np.random.randint(1, 5)) % vocab_size
            
            self.data.append(torch.tensor(seq, dtype=torch.long))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

# Helper functions for visualization
def visualize_token_embeddings(model, projection_type='pca'):
    """Visualize token embeddings in 2D or 3D space"""
    # Get token embeddings
    token_embeddings = model.token_embed.weight.detach().cpu().numpy()
    
    if projection_type == 'pca':
        from sklearn.decomposition import PCA
        # Use PCA for dimensionality reduction
        if token_embeddings.shape[1] > 3:
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(token_embeddings)
        else:
            embeddings_3d = token_embeddings
    elif projection_type == 'torus':
        # Project to torus surface (simplified visualization)
        R, r = 2.0, 1.0  # Major and minor radii
        
        # Normalize to unit sphere
        norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True)
        sphere_points = token_embeddings / norms
        
        # Use first two dims for first angle and next two for second angle
        dim = sphere_points.shape[1]
        half_dim = dim // 2
        
        # Get angles (simplified)
        theta = np.arctan2(sphere_points[:, 0], sphere_points[:, 1])
        phi = np.arctan2(sphere_points[:, half_dim], sphere_points[:, half_dim+1 if half_dim+1 < dim else 0])
        
        # Map to 3D torus coordinates
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        
        embeddings_3d = np.column_stack([x, y, z])
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot only a subset if there are too many tokens
    max_tokens = 100
    if len(embeddings_3d) > max_tokens:
        indices = np.random.choice(len(embeddings_3d), max_tokens, replace=False)
        plot_embeddings = embeddings_3d[indices]
        tokens = [chr(i) if i >= 32 and i < 127 else f"{i}" for i in indices]
    else:
        plot_embeddings = embeddings_3d
        tokens = [chr(i) if i >= 32 and i < 127 else f"{i}" for i in range(len(embeddings_3d))]
    
    # Color by position in embedding space
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_embeddings)))
    
    # Plot points
    ax.scatter(
        plot_embeddings[:, 0],
        plot_embeddings[:, 1],
        plot_embeddings[:, 2],
        c=colors,
        alpha=0.8
    )
    
    # Add labels for printable ASCII characters
    for i, token in enumerate(tokens):
        if ord(token[0]) >= 32 and ord(token[0]) < 127:
            ax.text(
                plot_embeddings[i, 0],
                plot_embeddings[i, 1],
                plot_embeddings[i, 2],
                token,
                fontsize=8
            )
    
    ax.set_title(f"Token Embeddings - {projection_type.upper()} Projection")
    
    # If torus, add wireframe to show torus structure
    if projection_type == 'torus':
        # Add a wireframe torus
        theta = np.linspace(0, 2*np.pi, 30)
        phi = np.linspace(0, 2*np.pi, 30)
        theta, phi = np.meshgrid(theta, phi)
        
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)
    
    plt.tight_layout()
    return fig

def visualize_attention_patterns(model, input_ids):
    """Visualize attention patterns for a given input"""
    import matplotlib.pyplot as plt
    
    # Forward pass to get attention weights
    attention_weights = []
    
    def get_attention_hook(i):
        def hook(module, input, output):
            # This assumes output is the attention output
            # We need the attention weights, which might require adaptation
            # depending on how the model calculates attention
            attention_weights.append(output)
        return hook
    
    # Register hooks to capture attention
    hooks = []
    for i, (attn, _) in enumerate(model.layers):
        # This might need adaptation depending on your model structure
        hooks.append(attn.fn.register_forward_hook(get_attention_hook(i)))
    
    # Forward pass
    with torch.no_grad():
        model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Create visualizations
    num_layers = len(model.layers)
    fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 4, 4))
    
    if num_layers == 1:
        axes = [axes]
    
    for i, attn_weights in enumerate(attention_weights):
        ax = axes[i]
        im = ax.imshow(attn_weights[0].cpu().numpy(), cmap='viridis')
        ax.set_title(f"Layer {i+1}")
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig

def compare_spherical_vs_toroidal(spherical_model, toroidal_model):
    """Compare embeddings between spherical and toroidal models"""
    # Get embeddings
    spherical_embeds = spherical_model.token_embed.weight.detach().cpu().numpy()
    toroidal_embeds = toroidal_model.token_embed.weight.detach().cpu().numpy()
    
    # Compute distance matrices
    from sklearn.metrics.pairwise import cosine_similarity
    
    spherical_sim = cosine_similarity(spherical_embeds)
    toroidal_sim = cosine_similarity(toroidal_embeds)
    
    # Plot comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Show heatmaps
    im1 = ax1.imshow(spherical_sim, cmap='viridis')
    ax1.set_title("Spherical Similarity")
    fig.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(toroidal_sim, cmap='viridis')
    ax2.set_title("Toroidal Similarity")
    fig.colorbar(im2, ax=ax2)
    
    # Difference
    diff = toroidal_sim - spherical_sim
    im3 = ax3.imshow(diff, cmap='coolwarm')
    ax3.set_title("Difference (Toroidal - Spherical)")
    fig.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    return fig

# Training loop
def train_toroidal_model(
    model,
    train_dataset,
    val_dataset=None,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    validation_every=VALIDATION_EVERY
):
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Register weight normalization hook if not using parametrization
    model.register_step_post_hook(optimizer)
    
    # Training metrics
    step = 0
    train_losses = []
    val_losses = []
    
    # Initialize progress bar
    total_steps = num_epochs * len(train_loader)
    pbar = tqdm(total=total_steps, desc="Training")
    
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            # Get data
            batch = batch.to(device)
            
            # Forward pass
            loss = model(batch, return_loss=True)
            
            # Backward pass (with gradient accumulation)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Track metrics
            train_losses.append(loss.item() * gradient_accumulation_steps)
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
            
            # Validation
            if step % validation_every == 0 and val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = val_batch.to(device)
                        val_batch_loss = model(val_batch, return_loss=True)
                        val_loss += val_batch_loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                pbar.write(f"Epoch {epoch+1}, Step {step}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_loss:.4f}")
                
                # Visualize embeddings periodically
                if step % (validation_every * 5) == 0:
                    visualize_token_embeddings(model, projection_type='torus')
                    plt.savefig(f"embeddings_step_{step}.png")
                    plt.close()
                
                model.train()
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({"train_loss": f"{train_losses[-1]:.4f}"})
    
    pbar.close()
    
    # Final visualization
    print("Training complete. Generating final visualizations...")
    fig = visualize_token_embeddings(model, projection_type='torus')
    plt.savefig("final_embeddings_torus.png")
    plt.close()
    
    fig = visualize_token_embeddings(model, projection_type='pca')
    plt.savefig("final_embeddings_pca.png")
    plt.close()
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        # Plot validation loss at the correct x-positions
        val_x = np.arange(0, len(train_losses), validation_every)[:len(val_losses)]
        plt.plot(val_x, val_losses, label='Val Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curve.png")
    plt.close()
    
    return train_losses, val_losses

# Main function to run the training and debugging
def main():
    print("Creating synthetic dataset...")
    train_data = SyntheticDataset(num_samples=5000, seq_len=SEQUENCE_LENGTH)
    val_data = SyntheticDataset(num_samples=500, seq_len=SEQUENCE_LENGTH)
    
    print("Initializing toroidal model...")
    model = ToroidalGPT(
        num_tokens=256,
        dim=64,
        depth=4,
        heads=4,
        dim_head=16,
        tied_embedding=True,
        R=2.0,  # Major radius
        r=1.0,  # Minor radius
        groups=1,
        causal=True
    ).to(device)
    
    print("Model summary:")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Visualize initial embeddings
    print("Visualizing initial embeddings...")
    fig = visualize_token_embeddings(model, projection_type='torus')
    plt.savefig("initial_embeddings_torus.png")
    plt.close()
    
    fig = visualize_token_embeddings(model, projection_type='pca')
    plt.savefig("initial_embeddings_pca.png")
    plt.close()
    
    # Train the model
    print("Starting training...")
    train_losses, val_losses = train_toroidal_model(
        model,
        train_data,
        val_data,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS
    )
    
    # Save the model
    torch.save(model.state_dict(), "toroidal_model.pt")
    
    print("Debug training complete!")

if __name__ == "__main__":
    main()
