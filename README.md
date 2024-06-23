
## Script for Processing Images by Detecting and Cropping Circular Areas

This script processes images by detecting and cropping circular areas, saving the results in PNG and EPS formats. Key features include:

### Image Detection and Cropping

- Detects and crops circular areas, preserving transparency.
- Handles black parts within the circle.

### Results Handling

- Saves results in `results/run_xxxx` folders, where `xxxx` is a timestamp.
- Outputs PNG and vectorized EPS formats.

### Logging and Display

- Displays progress and the number of images found.
- Logs processing times, errors, and completion status.

### File Handling

- Searches the current directory for images (e.g., .jpg, .jpeg, .png, .bmp, .tiff).
- Creates necessary folders for storing results and logs.
 
### Required libraries
```
pip install opencv-python-headless numpy scikit-image matplotlib

