import numpy as np
import matplotlib.pyplot as plt

# Constants
GrayCardReflectance = 0.18
WaterSurfaceReflectanceFactor = 0.028

def load_data(file_path):
    """Load data from a .npy file and return wavelengths and separate channels."""
    data = np.load(file_path)
    wavelengths = data[0, :]
    r_channel = data[1, :]
    g_channel = data[2, :]
    b_channel = data[3, :]
    return wavelengths, r_channel, g_channel, b_channel

def compute_reflectance(grey_light_level, sky_light_level, water_light_level):
    """Compute the Remote Sensing Reflectance for each wavelength."""
    # Water-leaving radiance
    Lw = water_light_level - (WaterSurfaceReflectanceFactor * sky_light_level)
    Lw = np.maximum(Lw, 0)  # Ensure non-negative values

    # Downwelling irradiance
    Ed = (np.pi / GrayCardReflectance) * grey_light_level

    # Compute Rrs
    valid = Ed > 1e-6
    Rrs = np.where(valid, Lw / Ed, 0)

    return Rrs

def plot_spectrum(wavelengths, r, g, b, title, filename, ylabel='Intensity'):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, r, 'r', label='Red Channel')
    plt.plot(wavelengths, g, 'g', label='Green Channel')
    plt.plot(wavelengths, b, 'b', label='Blue Channel')
    plt.title(title)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Main processing
# Load data for each measurement type
wavelengths, water_r, water_g, water_b = load_data('water.npy')
_, sky_r, sky_g, sky_b = load_data('sky.npy')
_, grey_r, grey_g, grey_b = load_data('grey.npy')

# Compute reflectance
rrs_r = compute_reflectance(grey_r, sky_r, water_r)
rrs_g = compute_reflectance(grey_g, sky_g, water_g)
rrs_b = compute_reflectance(grey_b, sky_b, water_b)

# Print results
print("Light Levels (mean values):")
print(f"Grey - Red: {np.mean(grey_r):.6f}, Green: {np.mean(grey_g):.6f}, Blue: {np.mean(grey_b):.6f}")
print(f"Sky  - Red: {np.mean(sky_r):.6f}, Green: {np.mean(sky_g):.6f}, Blue: {np.mean(sky_b):.6f}")
print(f"Water- Red: {np.mean(water_r):.6f}, Green: {np.mean(water_g):.6f}, Blue: {np.mean(water_b):.6f}")

print("\nRemote Sensing Reflectance (mean values):")
print(f"Red:   {np.mean(rrs_r):.6f}")
print(f"Green: {np.mean(rrs_g):.6f}")
print(f"Blue:  {np.mean(rrs_b):.6f}")





# Plot results
plot_spectrum(wavelengths, water_r, water_g, water_b, 'Water Spectrum', 'water_spectrum.png')
plot_spectrum(wavelengths, rrs_r, rrs_g, rrs_b, 'Remote Sensing Reflectance (Rrs)', 'rrs_spectrum.png', ylabel='Rrs (sr^-1)')

print("Analysis complete. Check the generated PNG files for results.")