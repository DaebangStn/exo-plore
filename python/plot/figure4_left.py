import numpy as np
import matplotlib.pyplot as plt
from python.plot.util import *

# Polynomial coefficients for quadratic curves: ax^2 + bx + c
# Format: [a, b, c] for each mEE condition
POLYNOMIAL_DATA = {
    r'$\mathbf{mEE} = ma^{1.0}$': [1.8951, -6.2224, 7.8584],
    r'$\mathbf{mEE} = ma^{3.0}$': [0.5612, -1.5823, 4.4093],
    r'$\mathbf{mEE} = ma^{1.5}$': [3.7358, -11.0409, 10.6849]
}

# Plot configuration
FIGURE_SIZE = (6, 9)
X_RANGE = (1.8, 6.48)  # Range for x-axis (can be adjusted)
N_POINTS = 1000   # Number of points for smooth curves

def quadratic_function(x, coeffs):
    """
    Evaluate quadratic polynomial: ax^2 + bx + c

    Args:
        x: Input values (array or scalar)
        coeffs: List of coefficients [a, b, c]

    Returns:
        y: Polynomial values
    """
    a, b, c = coeffs
    return a * x**2 + b * x + c

def plot_quadratic_curves():
    """
    Plot quadratic curves for different mEE conditions
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(FIGURE_SIZE))
    fig.subplots_adjust(
        left=0.15,
        right=0.95,
        top=0.95,
        bottom=0.1,
    )

    # Generate x values for smooth curves
    x = np.linspace(X_RANGE[0] / 3.6, X_RANGE[1] / 3.6, N_POINTS)

    # Color palette for different curves
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green

    # Plot each quadratic curve
    for idx, (label, coeffs) in enumerate(POLYNOMIAL_DATA.items()):
        # Calculate y values
        y = quadratic_function(x, coeffs)

        # Plot the curve
        ax.plot(x * 3.6, y,
               color=colors[idx % len(colors)],
               linewidth=2.5,
               label=label,
               alpha=0.8)

        # Print polynomial equation for verification
        a, b, c = coeffs
        print(f"{label}: y = {a:.4f}x^2 + {b:.4f}x + {c:.4f}")

    # Configure plot appearance
    ax.set_xlabel('Input Variable', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Output Value', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Quadratic Polynomial Curves for Different mEE Conditions',
                fontsize=FONT_SIZE_LABEL + 2)

    # Set axis limits
    ax.set_xlim(X_RANGE)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add legend
    ax.legend(fontsize=FONT_SIZE_LEGEND,
             loc='best',
             frameon=True,
             fancybox=True,
             shadow=True)

    # Improve tick formatting
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_LABEL)

    # Tight layout for better spacing
    plt.tight_layout()

    return fig, ax

def main():
    """
    Main function to create and display the quadratic curve plot
    """
    print("Plotting quadratic curves with given polynomial coefficients...")
    print("="*60)

    # Create the plot
    fig, ax = plot_quadratic_curves()

    save_path = PLOT_SAVE_DIR / "figure4_left.png"
    plt.savefig(save_path, dpi=150, transparent=True)
    print(f"Figure saved as: {save_path}")


    print("="*60)
    print(" Quadratic curve plotting completed successfully!")

if __name__ == "__main__":
    main()