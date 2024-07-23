import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from a .npy file and return wavelengths and separate channels."""
    data = np.load(file_path)
    wavelengths = data[0, :]
    r_channel = data[1, :]
    g_channel = data[2, :]
    b_channel = data[3, :]
    return wavelengths, r_channel, g_channel, b_channel

def load_dark(file_path):
    """Load dark data from a .npy file."""
    data = np.load(file_path)
    wavelengths = data[0, :]
    dark_current = np.mean(data[1:, :], axis=0)  # Average across all channels
    return wavelengths, dark_current

def plot_dark_data(wavelengths, dark_current, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, dark_current, label='Dark Current')
    plt.title(title)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def print_dark_statistics(dark_current):
    print("Dark current statistics:")
    print(f"  Min: {np.min(dark_current):.6f}")
    print(f"  Max: {np.max(dark_current):.6f}")
    print(f"  Mean: {np.mean(dark_current):.6f}")
    print(f"  Std Dev: {np.std(dark_current):.6f}")
    print(f"  Negative values: {np.sum(dark_current < 0)}")
    print(f"  Zero values: {np.sum(dark_current == 0)}")
    print()

def apply_dark_correction(data, dark_current):
    """Apply dark correction and handle negative values."""
    corrected = data - dark_current
    negative_count = np.sum(corrected < 0)
    if negative_count > 0:
        print(f"Warning: {negative_count} negative values found after dark correction. Correcting to 0.")
        corrected = np.maximum(corrected, 0)
    return corrected

def validate_input_data(grey, sky, water):
    """Validate input data and print statistics."""
    for name, data in [("Grey", grey), ("Sky", sky), ("Water", water)]:
        print(f"{name} data statistics:")
        print(f"  Min: {np.min(data):.6f}")
        print(f"  Max: {np.max(data):.6f}")
        print(f"  Mean: {np.mean(data):.6f}")
        print(f"  Negative values: {np.sum(data < 0)}")
        print(f"  Zero values: {np.sum(data == 0)}")
        print()

def compute_reflectance(grey_light_level, sky_light_level, water_light_level):
    """Compute the Remote Sensing Reflectance for a single channel."""
    GrayCardReflectance = 0.18  # 18% reflectance for gray card
    WaterSurfaceReflectanceFactor = 0.028  # Constant from Mobley 1999
    
    # Ensure non-negative input data
    grey_light_level = np.maximum(grey_light_level, 0)
    sky_light_level = np.maximum(sky_light_level, 0)
    water_light_level = np.maximum(water_light_level, 0)
    
    # Calculate water-leaving radiance
    WaterLeavingRadiance = np.maximum(water_light_level - (WaterSurfaceReflectanceFactor * sky_light_level), 0)
    
    # Calculate downwelling irradiance using the grey card
    DownwellingIrradiance = (np.pi / GrayCardReflectance) * grey_light_level
    
    # Avoid division by zero and ensure non-negative reflectance
    valid = DownwellingIrradiance > 0
    reflectance = np.where(valid, WaterLeavingRadiance / DownwellingIrradiance, 0)
    reflectance = np.maximum(reflectance, 0)
    
    # Round the results
    reflectance = np.round(reflectance, 6)
    
    # Debugging information
    print(f"Max values - Grey: {np.max(grey_light_level)}, Sky: {np.max(sky_light_level)}, Water: {np.max(water_light_level)}")
    print(f"Min values - Grey: {np.min(grey_light_level)}, Sky: {np.min(sky_light_level)}, Water: {np.min(water_light_level)}")
    print(f"Negative values - Grey: {np.sum(grey_light_level < 0)}, Sky: {np.sum(sky_light_level < 0)}, Water: {np.sum(water_light_level < 0)}")
    print(f"Max WaterLeavingRadiance: {np.max(WaterLeavingRadiance)}")
    print(f"Min WaterLeavingRadiance: {np.min(WaterLeavingRadiance)}")
    print(f"Max DownwellingIrradiance: {np.max(DownwellingIrradiance)}")
    print(f"Min DownwellingIrradiance: {np.min(DownwellingIrradiance)}")
    print(f"Max Reflectance: {np.max(reflectance)}")
    print(f"Min Reflectance: {np.min(reflectance)}")
    print(f"Negative Reflectance values: {np.sum(reflectance < 0)}")
    
    return reflectance

def plot_single_channel(wavelengths, data, title, filename, color, label, ylabel='Intensity'):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, data, color=color, label=label)
    plt.title(title)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_npy_data(wavelengths, r, g, b, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, r, 'r', label='Red Channel')
    plt.plot(wavelengths, g, 'g', label='Green Channel')
    plt.plot(wavelengths, b, 'b', label='Blue Channel')
    plt.title(title)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_combined_rrs(wavelengths, rrs_r, rrs_g, rrs_b, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, rrs_r, 'r', label='Red Channel')
    plt.plot(wavelengths, rrs_g, 'g', label='Green Channel')
    plt.plot(wavelengths, rrs_b, 'b', label='Blue Channel')
    plt.title(title)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Rrs [sr^-1]')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Load and plot dark data
wavelengths, dark_current = load_dark('dark.npy')
plot_dark_data(wavelengths, dark_current, 'Dark Current Spectrum', 'dark_spectrum.png')
print_dark_statistics(dark_current)

# Load data for each measurement type and apply dark correction
wavelengths, water_r, water_g, water_b = load_data('water.npy')
water_r = apply_dark_correction(water_r, dark_current)
water_g = apply_dark_correction(water_g, dark_current)
water_b = apply_dark_correction(water_b, dark_current)

_, sky_r, sky_g, sky_b = load_data('sky.npy')
sky_r = apply_dark_correction(sky_r, dark_current)
sky_g = apply_dark_correction(sky_g, dark_current)
sky_b = apply_dark_correction(sky_b, dark_current)

_, grey_r, grey_g, grey_b = load_data('grey.npy')
grey_r = apply_dark_correction(grey_r, dark_current)
grey_g = apply_dark_correction(grey_g, dark_current)
grey_b = apply_dark_correction(grey_b, dark_current)

# Validate input data
print("Validating Red Channel Data:")
validate_input_data(grey_r, sky_r, water_r)
print("Validating Green Channel Data:")
validate_input_data(grey_g, sky_g, water_g)
print("Validating Blue Channel Data:")
validate_input_data(grey_b, sky_b, water_b)

# Plot input NPY data for reference
plot_npy_data(wavelengths, water_r, water_g, water_b, 'Water NPY Data (Dark Corrected)', 'water_npy_reference_corrected.png')
plot_npy_data(wavelengths, sky_r, sky_g, sky_b, 'Sky NPY Data (Dark Corrected)', 'sky_npy_reference_corrected.png')
plot_npy_data(wavelengths, grey_r, grey_g, grey_b, 'Grey Card NPY Data (Dark Corrected)', 'grey_npy_reference_corrected.png')

# Plot original spectra for each channel separately
for data, color, channel, measurement in zip(
    [water_r, water_g, water_b, sky_r, sky_g, sky_b, grey_r, grey_g, grey_b],
    ['r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b'],
    ['Red', 'Green', 'Blue', 'Red', 'Green', 'Blue', 'Red', 'Green', 'Blue'],
    ['Water', 'Water', 'Water', 'Sky', 'Sky', 'Sky', 'Grey', 'Grey', 'Grey']
):
    plot_single_channel(wavelengths, data, f'Original {measurement} Spectrum - {channel} Channel (Dark Corrected)',
                        f'original_{measurement.lower()}_{channel.lower()}_corrected.png', color, f'{channel} Channel')

# Calculate RRS for each channel separately
print("Calculating RRS for Red Channel:")
rrs_r = compute_reflectance(grey_r, sky_r, water_r)
print("\nCalculating RRS for Green Channel:")
rrs_g = compute_reflectance(grey_g, sky_g, water_g)
print("\nCalculating RRS for Blue Channel:")
rrs_b = compute_reflectance(grey_b, sky_b, water_b)

# Print overall max RRS value
print(f"\nMax RRS - Red: {np.max(rrs_r)}, Green: {np.max(rrs_g)}, Blue: {np.max(rrs_b)}")

# Plot RRS for each channel separately
plot_single_channel(wavelengths, rrs_r, 'Remote Sensing Reflectance - Red Channel', 'RRS_r_corrected.png', 'red', 'Red Channel', ylabel='Rrs [sr^-1]')
plot_single_channel(wavelengths, rrs_g, 'Remote Sensing Reflectance - Green Channel', 'RRS_g_corrected.png', 'green', 'Green Channel', ylabel='Rrs [sr^-1]')
plot_single_channel(wavelengths, rrs_b, 'Remote Sensing Reflectance - Blue Channel', 'RRS_b_corrected.png', 'blue', 'Blue Channel', ylabel='Rrs [sr^-1]')

# Plot combined RRS
plot_combined_rrs(wavelengths, rrs_r, rrs_g, rrs_b, 'Combined Remote Sensing Reflectance (Dark Corrected)', 'RRS_combined_corrected.png')

print("Analysis complete. Check the generated PNG files for results.")