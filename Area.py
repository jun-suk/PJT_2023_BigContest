import cv2
import numpy as np
import pandas as pd
import glob
import os

# Define the mouse callback function
def mouse_callback(event, x, y, flags, param):
    global points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # Draw a red dot on the image

# Directory containing the images
image_directory = './01.sight_img/'

# Get a list of image paths using glob
image_paths = glob.glob(image_directory + '*.png')

# Load the list of processed image names or paths (if available)
processed_images_file = 'processed_images.txt'

if os.path.exists(processed_images_file):
    with open(processed_images_file, 'r') as f:
        processed_images = f.read().splitlines()
else:
    processed_images = []

# Filter out the already processeㅊㅇd images
remaining_image_paths = [path for path in image_paths if os.path.splitext(os.path.basename(path))[0] not in processed_images]

# Create an empty list to store the area data
area_data = []

# Define the epsilon value for contour approximation
epsilon = 0.005  # Adjust this value as needed

# Define the scaling factor to make the area value appear larger
scaling_factor = 1  # Adjust this value as needed

# Define the bright light green color (BGR format)
bright_light_green = (144, 238, 144)  # Adjust the color here

for idx, image_path in enumerate(remaining_image_paths):
    # Extract the original image name from the path without the extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary mask
    _, thresholded = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("Number of contours:", len(contours))  # Print number of contours
    
    # Create a list to store selected points
    points = []
    
    # Set up the mouse callback
    cv2.namedWindow('Original Image')
    cv2.setMouseCallback('Original Image', mouse_callback)
    
    while True:
        if len(points) > 0:
            temp_image = image.copy()
            overlay = temp_image.copy()
            cv2.fillPoly(overlay, [np.array(points)], color=bright_light_green)
            cv2.addWeighted(overlay, 0.3, temp_image, 0.7, 0, temp_image)
            cv2.drawContours(temp_image, contours, 0, (0, 0, 255), 1)
            
            cv2.imshow('Original Image', temp_image)
        else:
            cv2.imshow('Original Image', image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press the "Esc" key to exit
            break
        elif key == ord('c'):
            if len(points) > 2:  # At least 3 points needed to create a polygon
                polygon = np.array(points)
                area = cv2.contourArea(polygon) * scaling_factor
                print("Polygon area:", area)
                area_data.append({'seat_num': image_name, 'Area': area, 'Points': points})
                
                processed_images.append(image_name)
                with open(processed_images_file, 'a') as f:
                    f.write(image_name + '\n')
                
                print(f"Processed image {idx + 1}/{len(remaining_image_paths)} - Area: {area:.2f}")
            else:
                print("At least 3 points needed to calculate area.")
            break
        elif chr(key) == 'r':  # Press 'r' key to reset points
            points = []
            image = cv2.imread(image_path)
            cv2.imshow('Original Image', image)
    
    cv2.destroyAllWindows()

# Create a DataFrame from the area data
area_df = pd.DataFrame(area_data)

# Save the DataFrame to a CSV file
area_df.to_csv('sight_area_data.csv', index=False)
