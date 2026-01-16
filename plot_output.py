import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import glob
from pathlib import Path
from PIL import Image
import tempfile

def load_ising_data(filename):
    """
    Load Ising model data from CSV file.
    Expected format: NxN with values 1 or -1
    """
    try:
        data = pd.read_csv(filename, header=None)
        return data.values
    except:
        data = np.loadtxt(filename, delimiter=',')
        return data

def calculate_magnetization(lattice):
    """
    Calculate average magnetization per spin.
    M = <sum of all spins> / N
    """
    N = lattice.size
    M = np.sum(lattice) / N
    return M

def calculate_absolute_magnetization(lattice):
    """
    Calculate absolute magnetization per spin.
    |M| = |sum of all spins| / N
    """
    N = lattice.size
    M_abs = np.abs(np.sum(lattice)) / N
    return M_abs

def calculate_energy(lattice):
    """
    Calculate total energy of the system using periodic boundary conditions.
    E = -J * sum_{<i,j>} s_i * s_j
    Assuming J=1 for simplicity
    """
    N, M = lattice.shape
    energy = 0
    
    for i in range(N):
        for j in range(M):
            # Interaction with right neighbor (periodic boundary)
            energy -= lattice[i, j] * lattice[i, (j + 1) % M]
            # Interaction with bottom neighbor (periodic boundary)
            energy -= lattice[i, j] * lattice[(i + 1) % N, j]
    
    return energy

def calculate_energy_per_spin(lattice):
    """
    Calculate energy per spin.
    """
    total_energy = calculate_energy(lattice)
    return total_energy / lattice.size

def analyze_ising_state(lattice):
    """
    Calculate and display all relevant statistics for the Ising model.
    """
    M = calculate_magnetization(lattice)
    M_abs = calculate_absolute_magnetization(lattice)
    E = calculate_energy(lattice)
    E_per_spin = calculate_energy_per_spin(lattice)
    n_up = np.sum(lattice == 1)
    n_down = np.sum(lattice == -1)
    
    return {
        'M': M,
        'M_abs': M_abs,
        'E': E,
        'E_per_spin': E_per_spin,
        'n_up': n_up,
        'n_down': n_down
    }

def plot_all_states(directory):
    """
    Create a GIF animation from all CSV files in the directory.
    """
    # Find all CSV files
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return
    
    # Sort numerically by sweep number extracted from filename
    def extract_sweep_num(filepath):
        filename = os.path.basename(filepath)
        sweep_str = filename.replace('output_', '').replace('.csv', '')
        try:
            return int(sweep_str)
        except ValueError:
            return 0
    
    csv_files = sorted(csv_files, key=extract_sweep_num)
    
    print(f"Found {len(csv_files)} CSV files in {directory}")
    print(f"Creating GIF animation...")
    
    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        frames = []
        
        # Generate each frame
        for idx, csv_file in enumerate(csv_files):
            filename = os.path.basename(csv_file)
            print(f"Processing {filename}... ({idx+1}/{len(csv_files)})")
            
            lattice = load_ising_data(csv_file)
            stats = analyze_ising_state(lattice)
            
            # Extract sweep number from filename
            sweep_num = filename.replace('output_', '').replace('.csv', '')
            
            # Create individual plot
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(lattice, cmap='binary', interpolation='nearest', vmin=-1, vmax=1)
            ax.set_title(f'Sweep {sweep_num} - M={stats["M"]:.4f}, E/N={stats["E_per_spin"]:.4f}', 
                         fontsize=14)
            ax.axis('off')
            
            # Save frame to temporary file
            temp_frame = os.path.join(temp_dir, f"frame_{idx:04d}.png")
            plt.savefig(temp_frame, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Load frame as PIL Image
            frames.append(Image.open(temp_frame))
            
            print(f"  Size: {lattice.shape}, M={stats['M']:.4f}, |M|={stats['M_abs']:.4f}, E/N={stats['E_per_spin']:.4f}")
        
        # Create GIF
        gif_filename = "animation.gif"
        gif_path = os.path.join(directory, gif_filename)
        
        # Save as GIF with 500ms per frame (2 fps)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=500,
            loop=0
        )
        
        print(f"\nGIF animation saved to: {gif_path}")
        print(f"Total frames: {len(frames)}, Duration per frame: 500ms")

def main():
    """
    Main function to process all CSV files in a directory.
    """
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        # Try to find output_* directories
        output_dirs = sorted(glob.glob('output_*'))
        if output_dirs:
            directory = output_dirs[-1]  # Use most recent
            print(f"Using directory: {directory}")
        else:
            print("Usage: python plot_output.py <directory>")
            print("No output_* directories found in current directory.")
            sys.exit(1)
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)
    
    print("=" * 60)
    print(f"CREATING GIF ANIMATION FROM: {directory}")
    print("=" * 60)
    
    plot_all_states(directory)

if __name__ == "__main__":
    main()
