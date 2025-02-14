from PIL import Image
import numpy as np

def count_number_of_object(binary_image_array):
    # Get dimensions of the binary image
    image_height, image_width = len(binary_image_array), len(binary_image_array[0])
    
    # Matrix to track visited pixels
    visited_pixels = [[False for _ in range(image_width)] for _ in range(image_height)]
    
    # Directions for 4-connectivity (up, down, left, right)
    movement_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def perform_depth_first_search(start_row, start_col):
        pixel_stack = [(start_row, start_col)]
        
        while pixel_stack:
            current_row, current_col = pixel_stack.pop()
            if not visited_pixels[current_row][current_col]:
                visited_pixels[current_row][current_col] = True
                
                # Explore all 4-connected neighboring pixels
                for row_offset, col_offset in movement_directions:
                    neighbor_row, neighbor_col = current_row + row_offset, current_col + col_offset
                    
                    # Check bounds and if the neighbor is part of the object
                    if (0 <= neighbor_row < image_height and 0 <= neighbor_col < image_width and 
                        not visited_pixels[neighbor_row][neighbor_col] and binary_image_array[neighbor_row][neighbor_col] == 1):
                        pixel_stack.append((neighbor_row, neighbor_col))
    
    # Counter for distinct objects
    object_count = 0
    
    # Iterate through each pixel in the binary image
    for row in range(image_height):
        for col in range(image_width):
            # If the pixel is part of an object and has not been visited
            if binary_image_array[row][col] == 1 and not visited_pixels[row][col]:
                # Start a new DFS traversal from this pixel
                perform_depth_first_search(row, col)
                # Increment object count after finishing DFS for one object
                object_count += 1
    
    return object_count

# Load the binary image from file
binary_image_path = 'Project1.png'  # Update this path if necessary

try:
    # Open the image and convert it to binary (black and white)
    binary_image = Image.open(binary_image_path).convert('1')  # Convert to 1-bit pixels (binary)
    binary_image_array = np.array(binary_image, dtype=np.uint8)

    # Count objects in the binary image using the function
    total_objects_detected = count_number_of_object(binary_image_array)
    
    print(f"Number of objects in the binary image is: {total_objects_detected}")

except FileNotFoundError:
    print(f"Error: The file '{binary_image_path}' was not found. Please check the file path.")
except Exception as error:
    print(f"An error occurred: {error}")
