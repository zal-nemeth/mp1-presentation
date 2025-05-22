import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy import interpolate

# Create a DataFrame from the provided data with only the non-null values
data = {
    'x': [1, 4, 9, 12, 15, 19, 20],
    'y': [1, 1, 0.5, 0.5, 0.5, 0.1, 0.1]
}

df = pd.DataFrame(data)

# Get x and y values for only the known data points
x = df['x'].values
y = df['y'].values

# Define a finer x scale for smoother curve
x_smooth = np.linspace(x.min(), x.max(), 300)

# Use PCHIP interpolation which preserves monotonicity and is flatter near data points
pchip = PchipInterpolator(x, y)

# Evaluate the interpolation on the finer x scale
y_smooth = pchip(x_smooth)

plt.figure(figsize=(10, 6))

# Plot only the original data points that were provided
plt.plot(x, y, 'o', markersize=8, label='Original data points', color='red')
plt.plot(x_smooth, y_smooth, '-', linewidth=3, label='Sweeping curve', color='blue')

# Add labels and title
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Smooth Curve with Sweeping Arcs Between Data Points', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Save the plot
plt.savefig('sweeping_curve_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a DataFrame for the smoothed data
smoothed_df = pd.DataFrame({
    'x': x_smooth,
    'y': y_smooth
})

# Save the smoothed coordinates to a CSV file
smoothed_df.to_csv('flatter_curve_coordinates.csv', index=False)

plt.figure(figsize=(10, 6))

# Plot only the original data points that were provided
plt.plot(x, y, 'o', markersize=8, label='Original data points', color='red')
plt.plot(x_smooth, y_smooth, '-', linewidth=3, label='PCHIP interpolation curve', color='blue')

# Add labels and title
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Smooth Curve with Flatter Regions Near Data Points', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Save the plot
plt.savefig('flatter_curve_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'flatter_curve_plot.png'")
print("Flatter curve coordinates saved as 'flatter_curve_coordinates.csv'")

# Let's try another approach for comparison using Akima interpolation
# which also tends to be flatter near data points
from scipy.interpolate import Akima1DInterpolator

akima = Akima1DInterpolator(x, y)
y_akima = akima(x_smooth)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', markersize=8, label='Original data points', color='red')
plt.plot(x_smooth, y_akima, '-', linewidth=3, label='Akima interpolation curve', color='green')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Alternative Curve with Akima Interpolation', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.savefig('akima_curve_plot.png', dpi=300, bbox_inches='tight')

# Save the alternative coordinates
akima_df = pd.DataFrame({
    'x': x_smooth,
    'y': y_akima
})
akima_df.to_csv('akima_curve_coordinates.csv', index=False)
print("Akima plot saved as 'akima_curve_plot.png'")
print("Akima curve coordinates saved as 'akima_curve_coordinates.csv'")