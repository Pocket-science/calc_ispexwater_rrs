"""
Calibrate the wavelength response of an iSPEX unit using a spectrum of a
fluorescent light.

Command line arguments:
    * `file`: location of a RAW photograph of a fluorescent light spectrum,
    taken with iSPEX.

This should either be made generic (for any spectrometric data) or be forked
into the iSPEX repository.

NOTE: May not function correctly due to changes to flat-fielding methods. This
will be fixed with the general overhaul for iSPEX 2.
"""
import os
import numpy as np
from sys import argv
from lib.spectacle import general, io, plot, wavelength, raw as ispex_raw, raw2
from lib.ispex import general as ispex_general, wavelength as wvl, plot as ispex_plot
from pathlib import Path
from matplotlib import pyplot as plt
# to read raw image to find spectrum
import rawpy as rawpy_lib
from typing import List

def find_spectrum_in_raw_image(file_path: str) -> List[int]:
    """
    Analyze a .dng raw image file to find the spectrum.

    Parameters:
    file_path (str): The file path to the .dng image.

    Returns:
    List[int]: A list of x-coordinates where the spectrum appears.
    """

      # Convert Path object to string if necessary
    if isinstance(file_path, Path):
        file_path = file_path.as_posix()

    # Read the raw image file
    with rawpy_lib.imread(file_path) as raw:
        # Postprocess the raw image to get an RGB numpy array
        rgb_image = raw.postprocess()

    # Rotate the image if the vertical dimension is longer than the horizontal
    if rgb_image.shape[0] > rgb_image.shape[1]:
    # Rotate the image by 90 degrees
        rgb_image = np.rot90(rgb_image)
    
    # Convert the RGB image to grayscale
    gray_image = np.dot(rgb_image[..., :3], [0.33, 0.33, 0.33])
    
    # Apply a horizontal projection by summing along the vertical axis
    horizontal_projection = np.sum(gray_image, axis=0)
    
    # Define the midpoint to analyze the right half of the image
    midpoint = len(horizontal_projection) // 2
    right_side_data = horizontal_projection[midpoint:]

    # Set a threshold to find the significant peaks
    threshold = 0.3 * np.max(right_side_data)
    
    # Find peaks with the specified threshold
    peaks = find_peaks(right_side_data, threshold=threshold)

    # Adjust the peaks to account for the midpoint offset
    adjusted_peaks = [peak + midpoint for peak in peaks]
    spectrum_start_pixel = min(adjusted_peaks)
    spectrum_end_pixel = max(adjusted_peaks)
    print(f"Found spectrum between pixels {spectrum_start_pixel} and {spectrum_end_pixel}")

    # Plot the full horizontal projection and the peaks on the right side
    plt.figure(figsize=(10, 4))
    plt.plot(horizontal_projection, label='Full Horizontal Projection')
    plt.scatter(adjusted_peaks, horizontal_projection[adjusted_peaks], color='red', label='Peaks on the Right')
    plt.title('Full Horizontal Projection with Identified Peaks on the Right')
    plt.legend()
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    # plt.show()
    plt.savefig("FHPWIPR.pdf", bbox_inches="tight")
    plt.close()    

    # Return the start pixel value
    return spectrum_start_pixel

# Helper function to find peaks in the data
def find_peaks(data: np.ndarray, threshold: float) -> List[int]:
    """
    Find peaks in a 1D array of data points above a certain threshold.

    Parameters:
    data (np.ndarray): The 1D array of data points.
    threshold (float): The threshold to identify significant peaks.

    Returns:
    List[int]: A list of indices where peaks are found.
    """
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > threshold:
            peaks.append(i)
    return peaks

# Get the data folder from the command line
file = io.path_from_input(argv)


#print(find_spectrum_in_raw_image(file))
x_offset = find_spectrum_in_raw_image(file)

save_to_Qp = "wavelength_calibration_Qp.npy"
save_to_Qm = "wavelength_calibration_Qm.npy"
file_str = str(file)

with rawpy_lib.imread(file_str) as raw:
    # Get image size
    width, height = raw.sizes.raw_width, raw.sizes.raw_height

    # Print horizontal and vertical pixel lengths
    print('Raw Image Width:', width)
    print('Raw Image Height:', height)

# Check if the file exists
if not os.path.exists(file):
    raise FileNotFoundError(f"The file {file} was not found.")

# Load the data
img = io.load_raw_file(file)
print("Loaded RAW image")

# Check if raw_type is RawType.Flat
raw_type = img.raw_type
# Check if raw_type is RawType.Flat
if str(raw_type) != "RawType.Flat":
    raise ValueError(f"Invalid raw type: {raw_type}. Expected RawType.Flat")

print(img.raw_type)
data = img.raw_image.astype(np.float64)
bayer_map = img.raw_colors

# Post-processed image = transform into numpy array
img_post = img.postprocess()

# Transpose the image from portrait to landscape
img_post = np.swapaxes(img_post, 0, 1)

# Bias calibration without SPECTACLE data
data = data - float(img.black_level_per_channel[0])

#define start and end position for the slices containing slit + spectrum for Qm/Qp in the image (based on hardcoded iPhoneSE data)
slice_Qp, slice_Qm = ispex_general.find_spectrum(data)

# Show slices on top of the original image (probably to check if it's aligned correctly)
ispex_plot.plot_bounding_boxes(img_post, label_file=file, saveto="bounding_boxes.pdf")

# cut out 2 slices of 750 pixels high x 4032 pixels wide on the data and bayer map images
data_Qp, data_Qm = data[slice_Qp], data[slice_Qm]
bayer_Qp, bayer_Qm = bayer_map[slice_Qp], bayer_map[slice_Qm]

#Debayer the 4 channels RGBG RAW image into RGB data
RGB_Qp = raw2.pull_apart2(data_Qp, bayer_Qp)
RGB_Qm = raw2.pull_apart2(data_Qm, bayer_Qm)

#Define variables for size 
x = np.arange(data_Qp.shape[1])
yp = np.arange(data_Qp.shape[0])
ym = np.arange(data_Qm.shape[0])

#add extra xp and xm for raw_demosaic at bottom of script
xp = np.repeat(x[:,np.newaxis], bayer_Qp.shape[0], axis=1).T
xm = np.repeat(x[:,np.newaxis], bayer_Qm.shape[0], axis=1).T

# Convolve the data with a Gaussian kernel on the wavelength axis to remove noise
gauss_Qp = general.gauss_filter_multidimensional(RGB_Qp, sigma=(0,0,6))
gauss_Qm = general.gauss_filter_multidimensional(RGB_Qm, sigma=(0,0,6))

# Find the range of pixel values for the R,G,B peaks in the image
# x_offset = 2100
lines_Qp = wavelength.find_fluorescent_lines(gauss_Qp[...,x_offset:]) + x_offset
lines_Qm = wavelength.find_fluorescent_lines(gauss_Qm[...,x_offset:]) + x_offset

lines_fit_Qp = wavelength.fit_fluorescent_lines(lines_Qp, yp)
lines_fit_Qm = wavelength.fit_fluorescent_lines(lines_Qm, ym)

ispex_plot.plot_fluorescent_lines(yp, lines_Qp, lines_fit_Qp,saveto="fl_linesQp.pdf")
ispex_plot.plot_fluorescent_lines(ym, lines_Qm, lines_fit_Qm,saveto="fl_linesQm.pdf")

ispex_plot.plot_fluorescent_lines_double([yp, ym], [lines_Qp, lines_Qm], [lines_fit_Qp, lines_fit_Qm], saveto="TL_calibration.pdf")

# Calculate the dispersion (nm/pixel) for each row (R line - B line) / (R pixel - B pixel)
dispersion_Qp = wvl.dispersion_fluorescent(lines_fit_Qp)
dispersion_Qm = wvl.dispersion_fluorescent(lines_fit_Qm)

#plot the lines, the fit and the dispersion and save to file
ispex_plot.plot_fluorescent_lines_dispersion([yp, ym], [lines_Qp, lines_Qm], [lines_fit_Qp, lines_fit_Qm], [dispersion_Qp, dispersion_Qm], saveto="TL_calibration_dispersion.pdf")

# Calculate the spectral resolution for all rows
resolution_Qp = wvl.resolution(gauss_Qp, dispersion_Qp)
resolution_Qm = wvl.resolution(gauss_Qm, dispersion_Qm)
# print(f"Resolution Qp: {resolution_Qp}")
# print(f"Resolution Qm: {resolution_Qm}")

# Fit a wavelength relation for each row, meaning: try to fit a polynomial to the 3 lines (R, G, B) with 3 coefficients
# Using an ax^2 + bx + c function with the coefficients to match the wavelength, where x = the pixel value that corresponds with R,G,B
wavelength_fits_Qp = wavelength.fit_many_wavelength_relations(yp, lines_fit_Qp)
wavelength_fits_Qm = wavelength.fit_many_wavelength_relations(ym, lines_fit_Qm)

# Fit a polynomial to the coefficients of the previous fit
#These values are the most important of the whole script, as this array of 15 values can be used
# to calculate any value of wavelength for any pixel in the image!
coefficients_Qp, coefficients_fit_Qp = wavelength.fit_wavelength_coefficients(yp, wavelength_fits_Qp)
coefficients_Qm, coefficients_fit_Qm = wavelength.fit_wavelength_coefficients(ym, wavelength_fits_Qm)
# print(coefficients_Qp)

# Save the coefficients to file for use with other scripts like spectrum.py
wavelength.save_coefficients(coefficients_Qp, saveto=save_to_Qp)
wavelength.save_coefficients(coefficients_Qm, saveto=save_to_Qm)
print(f"Saved wavelength coefficients to '{save_to_Qp}' and '{save_to_Qm}'")

# Convert the input image pixel values to wavelengths values using the coefficients
wavelengths_Qp = wavelength.calculate_wavelengths(coefficients_Qp, x, yp)
wavelengths_Qm = wavelength.calculate_wavelengths(coefficients_Qm, x, ym)

# Demoisaic the image by splitting the image into 4 channels (R, G, B, G) and interpolating the pixel intensities for the bayer pattern 
#this halves the width and height of the image, so the image is now 750 pixels high x 2016 pixels wide
wavelengths_split_Qp, RGBG_Qp, xp_split = ispex_raw.demosaick(bayer_Qp, [wavelengths_Qp, data_Qp, xp])
wavelengths_split_Qm, RGBG_Qm, xm_split = ispex_raw.demosaick(bayer_Qm,[wavelengths_Qm, data_Qm, xm])

#Extra smoothing on the curve (OPTIONAL)
RGBG_Qp = general._gauss_nan(RGBG_Qp, sigma=(0,0,3))
RGBG_Qm = general._gauss_nan(RGBG_Qm, sigma=(0,0,3))

#Interpolate all float values pixel values that contain the wavelength of that pixel to the lambdarange (390-700 nm) with a step of 1 nm
lambdarange, all_interpolated_Qp = wavelength.interpolate_multi(wavelengths_split_Qp, RGBG_Qp)
lambdarange, all_interpolated_Qm = wavelength.interpolate_multi(wavelengths_split_Qm, RGBG_Qm)

#Stack and plot the spectrum
stacked_Qp = wavelength.stack(lambdarange, all_interpolated_Qp)
stacked_Qm = wavelength.stack(lambdarange, all_interpolated_Qm)

plot.plot_fluorescent_spectrum(stacked_Qp[0], stacked_Qp[1:])
plot.plot_fluorescent_spectrum(stacked_Qm[0], stacked_Qm[1:])