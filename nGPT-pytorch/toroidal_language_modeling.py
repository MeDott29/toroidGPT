import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
import random
import string
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from nGPT_pytorch import nGPT

# Import our toroidal model implementation
# This import will need to be adjusted based on your file structure
from toroidal_ngpt import ToroidalGPT, toroidal_norm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper function to create dummy text data with special patterns
def generate_synthetic_language_data(n_samples=1000, sequence_length=128, vocab_size=256):
    """
    Generate synthetic language data with circular/periodic patterns
    that might benefit from toroidal representation
    """
    data = []
    
    # Create patterns that might benefit from toroidal representation
    pattern_types = [
        'circular',        # Words/tokens that naturally form a ring (e.g., days, months)
        'grid',            # 2D periodic data (e.g., coordinates)
        'hierarchical',    # Multi-level categories
        'random'           # Random text for control
    ]
    
    for _ in range(n_samples):
        pattern_type = random.choice(pattern_types)
        
        if pattern_type == 'circular':
            # Create circular patterns like days, months, or hours
            cycle_type = random.choice(['days', 'months', 'hours'])
            
            if cycle_type == 'days':
                # Use ASCII codes for days (e.g., Sunday=83, Monday=77, etc.)
                days = "SMTWRFS"  # Codes for Sun,Mon,Tue,Wed,Thu,Fri,Sat
                seq = []
                start_day = random.randint(0, 6)
                for i in range(sequence_length):
                    day_idx = (start_day + i) % 7
                    seq.append(ord(days[day_idx]))
            
            elif cycle_type == 'months':
                # Use ASCII codes for months (Jan=74, Feb=70, etc.)
                months = "JFMAMJJASOND"  # Codes for Jan-Dec
                seq = []
                start_month = random.randint(0, 11)
                for i in range(sequence_length):
                    month_idx = (start_month + i) % 12
                    seq.append(ord(months[month_idx]))
            
            else:  # hours
                # Use ASCII numbers for hours (0-23)
                seq = []
                start_hour = random.randint(0, 23)
                for i in range(sequence_length):
                    hour = (start_hour + i) % 24
                    # Convert to digits (ASCII 48-57)
                    if hour < 10:
                        seq.append(ord('0') + hour)
                    else:
                        seq.append(ord('1'))
                        seq.append(ord('0') + hour - 10)
                        # Handle the extra digit
                        if len(seq) >= sequence_length:
                            seq = seq[:sequence_length]
                            break
        
        elif pattern_type == 'grid':
            # Create 2D grid patterns with periodic boundary
            # Like coordinates (x,y) where each wraps around
            seq = []
            grid_size = random.choice([5, 7, 10])
            start_x = random.randint(0, grid_size-1)
            start_y = random.randint(0, grid_size-1)
            
            for i in range(0, sequence_length, 2):
                if i + 1 >= sequence_length:
                    break
                    
                x = (start_x + i//2) % grid_size
                y = (start_y + i//2) % grid_size
                
                # Encode x and y as ASCII digits
                seq.append(ord('0') + x)
                seq.append(ord('0') + y)
        
        elif pattern_type == 'hierarchical':
            # Create nested hierarchy with cycles
            # Like (section, subsection) where both wrap around
            seq = []
            n_sections = random.choice([3, 4, 5])
            n_subsections = random.choice([3, 4, 5])
            
            for i in range(0, sequence_length, 2):
                if i + 1 >= sequence_length:
                    break
                    
                section = (i // (2 * n_subsections)) % n_sections
                subsection = (i // 2) % n_subsections
                
                # Encode as uppercase (sections) and lowercase (subsections) letters
                seq.append(ord('A') + section)
                seq.append(ord('a') + subsection)
        
        else:  # random text
            seq = [random.randint(32, min(126, vocab_size-1)) for _ in range(sequence_length)]
        
        # Ensure we hit exactly sequence_length
        if len(seq) < sequence_length:
            # Pad with spaces if needed
            seq.extend([32] * (sequence_length - len(seq)))
        elif len(seq) > sequence_length:
            seq = seq[:sequence_length]
            
        data.append(torch.tensor(seq, dtype=torch.long))
    
    return data

# Dataset for our synthetic data
class SyntheticLanguageDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Function to train a model
def train_model(model, train_loader, val_loader=None, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # If using manual normalization, register hook
    model.register_step_post_hook(optimizer)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in train_loader:
                batch = batch.to(device)
                
                optimizer.zero_grad()
                loss = model(batch, return_loss=True)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    loss = model(batch, return_loss=True)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

# Function to analyze token embeddings
def analyze_embeddings(model, title="Token Embeddings Analysis"):
    # Get token embeddings
    token_embeddings = model.token_embed.weight.detach().cpu().numpy()
    
    # For interpretability, focus on ASCII printable characters
    printable_indices = [i for i in range(32, 127)]  # ASCII 32-126 (space to ~)
    printable_chars = [chr(i) for i in printable_indices]
    printable_embeddings = token_embeddings[printable_indices]
    
    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(printable_embeddings)
    
    # Plot embeddings
    plt.figure(figsize=(12, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=range(len(printable_chars)), cmap='viridis', alpha=0.7)
    
    # Add labels for some interesting characters
    interesting_chars = {
        # Days of week shorthand
        'S': ord('S') - 32, 'M': ord('M') - 32, 'T': ord('T') - 32, 
        'W': ord('W') - 32, 'R': ord('R') - 32, 'F': ord('F') - 32,
        # Months shorthand
        'J': ord('J') - 32, 'F': ord('F') - 32, 'A': ord('A') - 32,
        'O': ord('O') - 32, 'N': ord('N') - 32, 'D': ord('D') - 32,
        # Digits
        '0': ord('0') - 32, '1': ord('1') - 32, '9': ord('9') - 32,
        # Other interesting characters
        'a': ord('a') - 32, 'z': ord('z') - 32,
    }
    
    for char, idx in interesting_chars.items():
        plt.annotate(char, (embeddings_2d[idx, 0], embeddings_2d[idx, 1]), fontsize=12)
    
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()

# Function to visualize how days of the week, months, digits map to torus
def visualize_circular_concepts(model, concept_type='days'):
    """Visualize how inherently circular concepts map to the torus"""
    # Get token embeddings
    token_embeddings = model.token_embed.weight.detach().cpu().numpy()
    
    if concept_type == 'days':
        # Days of the week
        chars = "SMTWRFS"  # Sunday, Monday, Tuesday, etc.
        title = "Days of Week Embeddings"
    elif concept_type == 'months':
        # Months
        chars = "JFMAMJJASOND"  # January, February, etc.
        title = "Months Embeddings"
    elif concept_type == 'digits':
        # Digits 0-9
        chars = "0123456789"
        title = "Digit Embeddings"
    else:
        # Default to letters
        chars = string.ascii_uppercase
        title = "Letter Embeddings"
    
    # Get embeddings for the characters
    indices = [ord(c) for c in chars]
    concept_embeddings = token_embeddings[indices]
    
    # Project to 3D using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(concept_embeddings)
    
    # Visualize in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the embeddings
    ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=range(len(chars)), cmap='viridis', s=100)
    
    # Add labels
    for i, char in enumerate(chars):
        ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], char, fontsize=12)
    
    # Connect points in sequence to show circularity
    for i in range(len(chars)):
        next_i = (i + 1) % len(chars)
        ax.plot([embeddings_3d[i, 0], embeddings_3d[next_i, 0]],
                [embeddings_3d[i, 1], embeddings_3d[next_i, 1]],
                [embeddings_3d[i, 2], embeddings_3d[next_i, 2]], 'k-', alpha=0.5)
    
    ax.set_title(title)
    
    return fig

# Function to compare spherical vs toroidal models
def compare_models(spherical_model, toroidal_model, test_data):
    """Compare the performance of spherical vs toroidal models"""
    spherical_model.eval()
    toroidal_model.eval()
    
    spherical_loss = 0.0
    toroidal_loss = 0.0
    
    with torch.no_grad():
        for batch in test_data:
            batch = batch.to(device)
            
            # Evaluate spherical model
            s_loss = spherical_model(batch, return_loss=True)
            spherical_loss += s_loss.item()
            
            # Evaluate toroidal model
            t_loss = toroidal_model(batch, return_loss=True)
            toroidal_loss += t_loss.item()
    
    avg_spherical_loss = spherical_loss / len(test_data)
    avg_toroidal_loss = toroidal_loss / len(test_data)
    
    print(f"Spherical Model Loss: {avg_spherical_loss:.4f}")
    print(f"Toroidal Model Loss: {avg_toroidal_loss:.4f}")
    
    return avg_spherical_loss, avg_toroidal_loss

# Main function
def main():
    # Parameters
    VOCAB_SIZE = 128  # ASCII-based vocabulary
    DIM = 64
    DEPTH = 4
    HEADS = 4
    DIM_HEAD = 16
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    
    # Generate synthetic data
    print("Generating synthetic language data...")
    data = generate_synthetic_language_data(
        n_samples=5000, 
        sequence_length=64,
        vocab_size=VOCAB_SIZE
    )
    
    # Split data
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    test_size = len(data) - train_size - val_size
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    # Create datasets and dataloaders
    train_dataset = SyntheticLanguageDataset(train_data)
    val_dataset = SyntheticLanguageDataset(val_data)
    test_dataset = SyntheticLanguageDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize models
    print("Initializing models...")
    
    # Standard nGPT (spherical)
    try:
        from nGPT_pytorch import nGPT
        spherical_model = nGPT(
            num_tokens=VOCAB_SIZE,
            dim=DIM,
            depth=DEPTH,
            heads=HEADS,
            dim_head=DIM_HEAD,
            tied_embedding=True,
            add_value_residual=True,
            attn_norm_qk=True,
            causal=True
        ).to(device)
    except ImportError:
        print("nGPT not found. Using ToroidalGPT with spherical settings (R=1, r=0).")
        spherical_model = ToroidalGPT(
            num_tokens=VOCAB_SIZE,
            dim=DIM,
            depth=DEPTH,
            heads=HEADS,
            dim_head=DIM_HEAD,
            tied_embedding=True,
            R=1.0,  # Effectively spherical (no torus)
            r=0.0,  # No minor radius
            causal=True
        ).to(device)
    
    # Toroidal model
    toroidal_model = ToroidalGPT(
        num_tokens=VOCAB_SIZE,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        tied_embedding=True,
        R=2.0,  # Major radius
        r=0.5,  # Minor radius
        causal=True
    ).to(device)
    
    # Train models
    print("Training spherical model...")
    spherical_losses = train_model(
        spherical_model, 
        train_loader, 
        val_loader, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE
    )
    
    print("Training toroidal model...")
    toroidal_losses = train_model(
        toroidal_model, 
        train_loader, 
        val_loader, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE
    )
    
    # Compare models
    print("Comparing models on test data...")
    spherical_test_loss, toroidal_test_loss = compare_models(
        spherical_model, 
        toroidal_model, 
        test_loader
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(spherical_losses[0], label='Spherical Train')
    plt.plot(toroidal_losses[0], label='Toroidal Train')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if spherical_losses[1] and toroidal_losses[1]:
        plt.plot(spherical_losses[1], label='Spherical Val')
        plt.plot(toroidal_losses[1], label='Toroidal Val')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    
    # Analyze embeddings
    print("Analyzing embeddings...")
    spherical_fig = analyze_embeddings(spherical_model, "Spherical Model Embeddings")
    spherical_fig.savefig('spherical_embeddings.png')
    
    toroidal_fig = analyze_embeddings(toroidal_model, "Toroidal Model Embeddings")
    toroidal_fig.savefig('toroidal_embeddings.png')
    
    # Visualize circular concept embeddings
    for concept in ['days', 'months', 'digits']:
        print(f"Visualizing {concept} embeddings...")
        s_fig = visualize_circular_concepts(spherical_model, concept)
        s_fig.savefig(f'spherical_{concept}_embeddings.png')
        
        t_fig = visualize_circular_concepts(toroidal_model, concept)
        t_fig.savefig(f'toroidal_{concept}_embeddings.png')
    
    print("Analysis complete!")

# Function to perform next-token predictions on specific patterns
def evaluate_next_token_prediction():
    """
    Evaluate how well the models predict the next token in circular patterns
    like days of the week, months, etc.
    """
    # Initialize small models
    DIM = 64
    DEPTH = 2
    VOCAB_SIZE = 128
    
    spherical_model = nGPT(
        num_tokens=VOCAB_SIZE,
        dim=DIM,
        depth=DEPTH,
        tied_embedding=True,
        causal=True
    ).to(device)
    
    toroidal_model = ToroidalGPT(
        num_tokens=VOCAB_SIZE,
        dim=DIM,
        depth=DEPTH,
        tied_embedding=True,
        R=2.0,
        r=0.5,
        causal=True
    ).to(device)
    
    # Generate patterns for testing
    days = "SMTWRFS"  # Days of the week
    months = "JFMAMJJASOND"  # Months
    hours = "0123456789101112131415161718192021222324"  # Hours (00-23)
    
    # Create test sequences
    day_seqs = []
    for i in range(len(days)):
        # Starting from each day, predict next few days
        start_idx = i
        seq = [ord(days[(start_idx + j) % len(days)]) for j in range(10)]
        day_seqs.append(torch.tensor(seq, dtype=torch.long))
    
    month_seqs = []
    for i in range(len(months)):
        # Starting from each month, predict next few months
        start_idx = i
        seq = [ord(months[(start_idx + j) % len(months)]) for j in range(10)]
        month_seqs.append(torch.tensor(seq, dtype=torch.long))
    
    # Function to evaluate prediction accuracy
    def evaluate_sequence_prediction(model, sequences, pattern_type):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for seq in sequences:
                seq = seq.to(device)
                
                # Use all but last token as input
                input_seq = seq[:-1].unsqueeze(0)  # Add batch dimension
                target = seq[-1].item()
                
                # Predict next token
                logits = model(input_seq)
                predicted = torch.argmax(logits[0, -1]).item()
                
                if predicted == target:
                    correct_predictions += 1
                total_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        print(f"{pattern_type} prediction accuracy: {accuracy:.4f}")
        return accuracy
    
    # Train models on each pattern type
    def train_on_pattern(model, sequences, epochs=50):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for seq in sequences:
                seq = seq.to(device)
                
                optimizer.zero_grad()
                loss = model(seq.unsqueeze(0), return_loss=True)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss / len(sequences):.4f}")
    
    # Train and evaluate on days pattern
    print("Training and evaluating on days pattern...")
    train_on_pattern(spherical_model, day_seqs)
    train_on_pattern(toroidal_model, day_seqs)
    
    day_acc_spherical = evaluate_sequence_prediction(spherical_model, day_seqs, "Days (Spherical)")
    day_acc_toroidal = evaluate_sequence_prediction(toroidal_model, day_seqs, "Days (Toroidal)")
    
    # Train and evaluate on months pattern
    print("Training and evaluating on months pattern...")
    train_on_pattern(spherical_model, month_seqs)
    train_on_pattern(toroidal_model, month_seqs)
    
    month_acc_spherical = evaluate_sequence_prediction(spherical_model, month_seqs, "Months (Spherical)")
    month_acc_toroidal = evaluate_sequence_prediction(toroidal_model, month_seqs, "Months (Toroidal)")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    patterns = ['Days', 'Months']
    spherical_acc = [day_acc_spherical, month_acc_spherical]
    toroidal_acc = [day_acc_toroidal, month_acc_toroidal]
    
    x = np.arange(len(patterns))
    width = 0.35
    
    plt.bar(x - width/2, spherical_acc, width, label='Spherical Model')
    plt.bar(x + width/2, toroidal_acc, width, label='Toroidal Model')
    
    plt.xlabel('Pattern Type')
    plt.ylabel('Prediction Accuracy')
    plt.title('Next Token Prediction Accuracy')
    plt.xticks(x, patterns)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('prediction_accuracy.png')
    
    return {
        'day_acc_spherical': day_acc_spherical,
        'day_acc_toroidal': day_acc_toroidal,
        'month_acc_spherical': month_acc_spherical,
        'month_acc_toroidal': month_acc_toroidal
    }

# Function to explore the effect of torus parameters
def explore_torus_parameters():
    """
    Explore how different torus parameters (R and r) affect model performance
    """
    # Generate data
    data = generate_synthetic_language_data(n_samples=2000, sequence_length=64)
    
    # Split data
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    train_dataset = SyntheticLanguageDataset(train_data)
    val_dataset = SyntheticLanguageDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Define torus parameter configurations to test
    configs = [
        {'R': 1.0, 'r': 0.0},  # Effectively spherical
        {'R': 1.5, 'r': 0.3},
        {'R': 2.0, 'r': 0.5},
        {'R': 2.5, 'r': 0.7},
        {'R': 3.0, 'r': 1.0}
    ]
    
    # Train models with different parameters
    results = []
    
    for config in configs:
        print(f"Training model with R={config['R']}, r={config['r']}...")
        model = ToroidalGPT(
            num_tokens=128,
            dim=64,
            depth=2,
            tied_embedding=True,
            R=config['R'],
            r=config['r'],
            causal=True
        ).to(device)
        
        # Train for a few epochs
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            epochs=3
        )
        
        # Get final validation loss
        final_val_loss = val_losses[-1] if val_losses else None
        
        results.append({
            'R': config['R'],
            'r': config['r'],
            'val_loss': final_val_loss
        })
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    Rs = [config['R'] for config in configs]
    rs = [config['r'] for config in configs]
    val_losses = [result['val_loss'] for result in results]
    
    plt.subplot(1, 2, 1)
    plt.plot(Rs, val_losses, 'o-')
    plt.xlabel('Major Radius (R)')
    plt.ylabel('Validation Loss')
    plt.title('Effect of Major Radius on Model Performance')
    
    plt.subplot(1, 2, 2)
    plt.plot(rs, val_losses, 'o-')
    plt.xlabel('Minor Radius (r)')
    plt.ylabel('Validation Loss')
    plt.title('Effect of Minor Radius on Model Performance')
    
    plt.tight_layout()
    plt.savefig('torus_parameters.png')
    
    return results

# Run the program
if __name__ == '__main__':
    main()
    evaluate_next_token_prediction()
    explore_torus_parameters()