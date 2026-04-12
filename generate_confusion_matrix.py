import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_cm():
    # Style setup to match the provided image
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Data
    cm = np.array([[3049, 33], [60, 1077]])
    labels = ["Normal", "TB"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=labels, 
        yticklabels=labels,
        ax=ax,
        cbar=True,
        annot_kws={"size": 11}
    )
    
    # Configure axes and title
    ax.set_title("Confusion Matrix - TB-Guard-XAI", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    
    # Tweak axes limits slightly to prevent clipping on top/bottom
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    # Remove grid lines from heatmap if style enabled them
    ax.grid(False)
    
    # Save the figure
    plt.tight_layout()
    output_path = "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved successfully to {output_path}")

if __name__ == "__main__":
    generate_cm()
