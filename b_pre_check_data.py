




# ---------------------------------------------------------------
# Daniel T. K. W. - github.com/danieltkw - danielkopolo95@gmail.com
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# b_pre_check_data.py beggin
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Imports
import cv2
import numpy as np
import os
import platform
from datetime import datetime
from skimage import measure, io
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------------


# ---------------------------------------------------------------
# ---- Clear terminal function ----
def clear_terminal():
    print("\033[H\033[J", end="")
    
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# ---- Function to crop the circular image from the black background and save it with transparency ----
def crop_circular_image(image_path, output_folder, log_file):
    start_time = time.time()
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect the outer circle which contains the black background
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                   param1=100, param2=30, minRadius=0, maxRadius=0)
        
        if circles is not None:
            # Convert circles to integer values
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Create a mask with the same size as the original image
                mask = np.zeros_like(image)
                cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
                
                # Check if there is a significant black part inside the detected circle
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                masked_image = cv2.bitwise_and(gray, mask_gray)
                mean_val = cv2.mean(masked_image, mask=mask_gray)[0]
                
                # Consider a threshold to determine if the black part should be included
                black_threshold = 50
                
                if mean_val > black_threshold:
                    # Create an output image with transparency (4 channels)
                    output = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                    
                    # Copy the image to the output where the mask is white
                    for i in range(3):  # Copy RGB channels
                        output[:, :, i] = image[:, :, i]
                    output[:, :, 3] = mask_gray  # Copy the mask to the alpha channel
                    
                    # Crop the circular part from the image
                    x1, y1, x2, y2 = max(0, x - r), max(0, y - r), min(image.shape[1], x + r), min(image.shape[0], y + r)
                    cropped_image = output[y1:y2, x1:x2]
                    
                    # Generate output path with timing in the name
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    output_png_path = os.path.join(output_folder, f"cropped_image_{timestamp}.png")
                    output_eps_path = os.path.join(output_folder, f"cropped_image_{timestamp}.eps")
                    
                    # Save the cropped image
                    cv2.imwrite(output_png_path, cropped_image)
                    print(f"Cropped image saved as {output_png_path}")
                    
                    # Convert the PNG to EPS (vectorized)
                    convert_to_eps(output_png_path, output_eps_path)
                    
                    # Log the success
                    with open(log_file, 'a') as log:
                        log.write(f"{image_path} processed successfully in {time.time() - start_time:.2f} seconds\n")
                    
                    # Open the cropped image
                    if platform.system() == "Windows":
                        os.system(f'start "" "{output_eps_path}"')
                    elif platform.system() == "Darwin":
                        os.system(f'open "{output_eps_path}"')
                    else:
                        os.system(f'xdg-open "{output_eps_path}"')
                    
                    return

        # Log no circle detected
        with open(log_file, 'a') as log:
            log.write(f"No circular part detected for {image_path}\n")
    except Exception as e:
        # Log any error
        with open(log_file, 'a') as log:
            log.write(f"Error processing {image_path}: {e}\n")

# ---------------------------------------------------------------


# ---------------------------------------------------------------
# ---- Function to convert PNG to EPS ----
def convert_to_eps(png_path, eps_path):
    # Read the PNG image
    image = io.imread(png_path, as_gray=True)
    
    # Edge detection
    edges = measure.find_contours(image, 0.8)
    
    # Create the plot
    fig, ax = plt.subplots()
    for edge in edges:
        ax.plot(edge[:, 1], edge[:, 0], linewidth=2)
    
    ax.set_axis_off()
    plt.gca().invert_yaxis()
    
    # Save as EPS
    plt.savefig(eps_path, format='eps', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Vectorized image saved as {eps_path}")

# ---------------------------------------------------------------


# ---------------------------------------------------------------
# ---- Main execution ----
if __name__ == "__main__":
    clear_terminal()
    start_time = time.time()
    
    # Search for images in the current folder
    folder_path = os.getcwd()
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images_to_process = [file_name for file_name in os.listdir(folder_path) if any(file_name.lower().endswith(ext) for ext in supported_formats)]
    
    # Create results folder and run-specific subfolder
    results_folder = os.path.join(folder_path, 'results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    run_folder = os.path.join(results_folder, f"run_{timestamp}")
    os.makedirs(run_folder)
    
    # Log file
    log_file = os.path.join(run_folder, 'process_log.txt')
    
    # Write initial log
    with open(log_file, 'a') as log:
        log.write(f"Process started at {timestamp}\n")
        log.write(f"Found {len(images_to_process)} images to process\n")
    
    # Process images
    total_images = len(images_to_process)
    for idx, file_name in enumerate(images_to_process):
        print(f"Processing image {idx + 1}/{total_images}...")
        input_image_path = os.path.join(folder_path, file_name)
        crop_circular_image(input_image_path, run_folder, log_file)
    
    # Write final log
    with open(log_file, 'a') as log:
        log.write(f"Process finished in {time.time() - start_time:.2f} seconds\n")
    
    print(f"Process finished. Log file saved at {log_file}")

# ---------------------------------------------------------------

    
# ---------------------------------------------------------------
# b_pre_check_data.py end
# ---------------------------------------------------------------











