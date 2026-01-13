import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_ising_data(filename):
    """
    Load Ising model data from CSV file.
    Expected format: 1024x1024 with values 1 or -1
    """
    try:
        # Try loading as a standard CSV
        data = pd.read_csv(filename, header=None)
        return data.values
    except:
        # If that fails, try loading with numpy
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

def calculate_susceptibility(lattice):
    """
    Calculate magnetic susceptibility (proportional to variance of magnetization).
    χ = (N/T) * (<M²> - <M>²)
    For a single configuration, we use variance approximation
    """
    M = calculate_magnetization(lattice)
    M_squared = M ** 2
    return M_squared * lattice.size

def plot_ising_state(lattice, filename='ising_state.png'):
    """
    Plot the Ising model lattice in black and white.
    -1 -> black, +1 -> white
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(lattice, cmap='binary', interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar(label='Spin value', ticks=[-1, 1])
    plt.title(f'Ising Model State ({lattice.shape[0]}x{lattice.shape[1]})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    # plt.savefig(filename, dpi=150)
    # print(f"Figure saved as {filename}")
    plt.show()

def analyze_ising_state(lattice):
    """
    Calculate and display all relevant statistics for the Ising model.
    """
    print("=" * 60)
    print("ISING MODEL ANALYSIS")
    print("=" * 60)
    print(f"Lattice size: {lattice.shape[0]} x {lattice.shape[1]}")
    print(f"Total spins: {lattice.size}")
    print()
    
    # Magnetization
    M = calculate_magnetization(lattice)
    M_abs = calculate_absolute_magnetization(lattice)
    print(f"Average magnetization per spin (M): {M:.6f}")
    print(f"Absolute magnetization per spin (|M|): {M_abs:.6f}")
    print()
    
    # Energy
    E = calculate_energy(lattice)
    E_per_spin = calculate_energy_per_spin(lattice)
    print(f"Total energy (E): {E:.2f}")
    print(f"Energy per spin (E/N): {E_per_spin:.6f}")
    print()
    
    # Additional statistics
    n_up = np.sum(lattice == 1)
    n_down = np.sum(lattice == -1)
    print(f"Number of up spins (+1): {n_up}")
    print(f"Number of down spins (-1): {n_down}")
    print(f"Fraction of up spins: {n_up / lattice.size:.4f}")
    print(f"Fraction of down spins: {n_down / lattice.size:.4f}")
    print()
    
    # Order parameter
    print(f"Order parameter (|M|): {M_abs:.6f}")
    if M_abs > 0.8:
        print("  -> System is highly ordered (ferromagnetic)")
    elif M_abs < 0.2:
        print("  -> System is disordered (paramagnetic)")
    else:
        print("  -> System is in intermediate state")
    print("=" * 60)
    
    return {
        'M': M,
        'M_abs': M_abs,
        'E': E,
        'E_per_spin': E_per_spin,
        'n_up': n_up,
        'n_down': n_down
    }

def main():
    """
    Main function to load data, analyze, and visualize Ising model state.
    """
    filename = 'output.csv'
    
    print(f"Loading data from {filename}...")
    lattice = load_ising_data(filename)
    
    print(f"Data loaded successfully!")
    print(f"Shape: {lattice.shape}")
    print()
    
    analyze_ising_state(lattice)
    
    plot_ising_state(lattice, filename=filename.replace('.csv', '_visualization.png'))

if __name__ == "__main__":
    main()
