import numpy as np
from sys import argv
from lib.spectacle import plot, io, wavelength, raw as ispex_raw, general, calibrate
from lib.ispex import general as ispex_general, plot as ispex_plot
from matplotlib import pyplot as plt
from pathlib import Path
import rawpy as rawpy_lib

def process_files(dng_path, npy_qp_path, npy_qm_path):
    # Use the provided paths instead of argv
    file_path = Path(dng_path)

    # Convert the Path object to a string if needed
    file_path_str = str(file_path)



    # Identify the directory of the input file
    input_file_directory = file_path.parent



    with rawpy_lib.imread(file_path_str) as raw:
        data = raw.raw_image.astype(np.float64)
        bayer_map = raw.raw_colors
        img_post = raw.postprocess()
        
        np.save(input_file_directory / f"{file_path.stem}_post_processed_data.npy", img_post)


    # Update paths for npy files to use npy_qp_path and npy_qm_path
    coefficients_Qp = np.load(Path(npy_qp_path))
    coefficients_Qm = np.load(Path(npy_qm_path))
        
    # Load the data

    data = io.load_raw_image(file_path)
    np.save(input_file_directory / f"{file_path.stem}l0_raw_data.npy", data)

    # Slice the data
    slice_Qp, slice_Qm = ispex_general.find_spectrum(data)

    # Show the bounding boxes for visualisation

    #ispex_plot.plot_bounding_boxes(data, label=file, saveto=input_file_directory/f"{file.stem}_bounding_boxes.pdf") -- changed path to file_path
    ispex_plot.plot_bounding_boxes(img_post, label=file_path, saveto=input_file_directory/f"bounding_boxes.png")
    data_Qp, data_Qm = data[slice_Qp], data[slice_Qm]

    np.save(input_file_directory / f"{file_path.stem}_spectrum_slices_Qp.npy", data_Qp)
    np.save(input_file_directory / f"{file_path.stem}_spectrum_slices_Qm.npy", data_Qm)




    bayer_Qp, bayer_Qm = bayer_map[slice_Qp], bayer_map[slice_Qm]

    noise_level = np.average(np.mean(data_Qp[:, :250], axis=1))
    print("noise_level:", noise_level)

    #subtract the noise level from the image
    data_Qp = data_Qp - noise_level
    data_Qm = data_Qm - noise_level



    x = np.arange(data_Qp.shape[1])
    xp = np.repeat(x[:,np.newaxis], bayer_Qp.shape[0], axis=1).T
    xm = np.repeat(x[:,np.newaxis], bayer_Qm.shape[0], axis=1).T
    yp = np.arange(data_Qp.shape[0])
    ym = np.arange(data_Qm.shape[0])


    wavelengths_Qp = wavelength.calculate_wavelengths(coefficients_Qp, x, yp)
    wavelengths_Qm = wavelength.calculate_wavelengths(coefficients_Qm, x, ym)

    wavelengths_split_Qp, RGBG_Qp, xp_split = ispex_raw.demosaick(bayer_Qp, [wavelengths_Qp, data_Qp, xp])
    wavelengths_split_Qm, RGBG_Qm, xm_split = ispex_raw.demosaick(bayer_Qm,[wavelengths_Qm, data_Qm, xm])


    np.save(input_file_directory / f"{file_path.stem}_demosaicked_Qp.npy", wavelengths_split_Qp)
    np.save(input_file_directory / f"{file_path.stem}_demosaicked_Qm.npy", wavelengths_split_Qm)



    # 1 pixel in RGBG space = 2 pixels in RGB space
    RGBG_Qp = general._gauss_nan(RGBG_Qp, sigma=(0,0,3))
    RGBG_Qm = general._gauss_nan(RGBG_Qm, sigma=(0,0,3))


    lambdarange, all_interpolated_Qp = wavelength.interpolate_multi(wavelengths_split_Qp, RGBG_Qp)
    lambdarange, all_interpolated_Qm = wavelength.interpolate_multi(wavelengths_split_Qm, RGBG_Qm)
    np.save(input_file_directory / f"{file_path.stem}_interpolated_Qp.npy", all_interpolated_Qp)
    np.save(input_file_directory / f"{file_path.stem}_interpolated_Qm.npy", all_interpolated_Qm)



    original_interpolate_Qp = all_interpolated_Qp.copy()
    original_interpolare_Qm = all_interpolated_Qm.copy()


    print('all_interpolated_Qp:', all_interpolated_Qp)
    stacked_Qp = wavelength.stack(lambdarange, all_interpolated_Qp)
    stacked_Qm = wavelength.stack(lambdarange, all_interpolated_Qm)



    np.save(input_file_directory / f"{file_path.stem}_stacked_spectrum_Qp.npy", stacked_Qp)
    np.save(input_file_directory / f"{file_path.stem}_stacked_spectrum_Qm.npy", stacked_Qm)


    print('stacked_Qp:', stacked_Qp)

    plt.figure(figsize=(6,2))
    for j, c in enumerate("rgb", 1):
        plt.plot(stacked_Qp[0], stacked_Qp[j], c=c)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Radiance [a.u.]")
    plt.grid(ls="--")
    plt.ylim(-5, np.nanmax(stacked_Qp[1:])*1.05)
    plt.xlim(390, 700)
    plt.savefig(input_file_directory/"Qp.png", dpi=300, bbox_inches="tight")
    plt.close()



    # Increase font sizes
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14})

    # Change the figure size for better aspect ratio on mobile
    plt.figure(figsize=(10, 4))  # Wider figure

    # Use a loop to plot each spectrum with a thicker line for visibility
    for j, color in zip(range(1, 4), ['red', 'green', 'blue']):  # Explicit color names for clarity
        plt.plot(stacked_Qp[0], stacked_Qp[j], c=color, linewidth=2)  # Thicker lines

    # Labeling with more descriptive terms
    plt.xlabel("Wavelength (nm)", fontsize=14, fontweight='bold')
    plt.ylabel("Brightness", fontsize=14, fontweight='bold')

    # Modify grid lines for a cleaner look
    plt.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

    # Set the y-limit to be a bit more than the max value for better spacing
    plt.ylim(0, np.nanmax(stacked_Qp[1:])*1.1)  # 10% more space above the max value

    # Set the x-limit to encompass the range of wavelengths you're interested in
    plt.xlim(390, 700)

    # Save as high-resolution PNG with adjusted DPI for quality
    plt.savefig(input_file_directory / "spectrum.png", bbox_inches="tight", dpi=300)

    # Clear the figure after saving to prevent overlap with future plots
    plt.close()

    plt.figure(figsize=(6,2))
    for j, c in enumerate("rgb", 1):
        plt.plot(stacked_Qm[0], stacked_Qm[j], c=c)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Radiance [a.u.]")
    plt.grid(ls="--")
    plt.ylim(-5, np.nanmax(stacked_Qm[1:])*1.05)
    plt.xlim(390, 700)
    plt.savefig(input_file_directory/"Qm.png", dpi=300, bbox_inches="tight")
    plt.close()

