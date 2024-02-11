import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from loguru import logger
import sys
import time
import random


class Settings:
    """Class to keep useful settings in one place"""

    viewer_size = '1000x1000'  # The size of whole picture viewer window (GUI)
    resizable_width = False  # Default viewer window is constant size (False)
    resizable_height = False  # Set True if you want resize window using mouse
    thumbnail_size = (320, 240)  # The thumbnails size of the original images at top row of the viewer
    image_size = (800, 600)  # The size of modified images in the middle of the viewer
    colour_scheme = 'gray'  # Default colour scheme for images, available: 'gray', 'black', 'red', 'green', 'blue'
    slice_count = 100  # Count of the slices is allowed from 20 to 200
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(project_root, 'images')  # Directory for original images
    temp_dir = os.path.join(project_root, 'temp')  # Directory for modified images
    # Logger settings
    log_level = 'INFO'
    log_file_path = os.path.join(project_root, 'logs', 'error.log')


def log_decorator(func):
    """Use for function debugging"""
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering '{func.__name__}' function with args: {args} and kwargs: {kwargs}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Executed '{func.__name__}' successfully in {time.time() - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Exception occurred in '{func.__name__}': {e}")
            raise e
    return wrapper


@log_decorator
def scan_directory_for_images(directory: str) -> list[str]:

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    return [os.path.join(directory, f) for f in os.listdir(directory) if
            f.endswith(supported_extensions) and os.path.isfile(os.path.join(directory, f))]


def slice_array_vertically(image_array: np.ndarray) -> np.ndarray:
    """Slice image ndarray into a number of slices and rearrange them first even, then odd."""

    slice_height = image_array.shape[0] // Settings.slice_count
    slices = [image_array[i*slice_height:(i+1)*slice_height, :] for i in range(Settings.slice_count)]
    rearranged_slices = slices[::2] + slices[1::2]
    return np.vstack(rearranged_slices)


def slice_array_horizontally(image_array: np.ndarray) -> np.ndarray:
    """Slice image ndarray into a number of slices and rearrange them first even, then odd."""

    slice_width = image_array.shape[1] // Settings.slice_count
    slices = [image_array[:, i*slice_width:(i+1)*slice_width] for i in range(Settings.slice_count)]
    rearranged_slices = slices[::2] + slices[1::2]
    return np.hstack(rearranged_slices)


def apply_color_scheme(image_array: np.ndarray) -> np.ndarray:
    """Apply a color scheme to an image ndarray and return a 3D array according to the scheme choice."""

    if Settings.colour_scheme == 'gray':
        grayscale = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
        return np.repeat(grayscale[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

    elif Settings.colour_scheme == 'black':
        grayscale = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
        black_and_white = grayscale > 115
        black_and_white = (black_and_white * 255).astype(np.uint8)
        return np.repeat(black_and_white[:, :, np.newaxis], 3, axis=2)

    elif Settings.colour_scheme == 'red':
        red_only = image_array.copy()
        red_only[:, :, 1:3] = 0
        return red_only

    elif Settings.colour_scheme == 'green':
        green_only = image_array.copy()
        green_only[:, :, 0] = 0
        green_only[:, :, 2] = 0
        return green_only

    elif Settings.colour_scheme == 'blue':
        blue_only = image_array.copy()
        blue_only[:, :, 0:2] = 0
        return blue_only

    else:
        error_message = (f"Unrecognized color scheme '{Settings.colour_scheme}'. Choose from 'gray', 'black', 'red', "
                         f"'green', 'blue'.")
        logger.error(error_message)
        raise ValueError(error_message)


def concatenate_arrays(arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Concatenate 1st with 2nd ndarray and 3rd with 4th array along axis 0, then concatenate these both along axis
    1."""
    # Shape arrays equal
    min_height = min(image.shape[0] for image in arrays)
    min_width = min(image.shape[1] for image in arrays)
    images = [image[:min_height, :min_width] for image in arrays]
    first_pair_concat = np.concatenate((images[0], images[1]), axis=0)
    second_pair_concat = np.concatenate((images[2], images[3]), axis=0)
    final_concat = np.concatenate((first_pair_concat, second_pair_concat), axis=1)
    return final_concat


def rotate_array(image_array: np.ndarray) -> np.ndarray:
    """Rotate an image ndarray by a multiple of 90 degrees, chosen randomly."""

    rotated_image = np.rot90(image_array, k=random.choice([1, 2, 3]))
    return rotated_image


def blur_array(image_array: np.ndarray, iterations: int = 1) -> np.ndarray:
    kernel = np.array([
        [1,  2,  3,  4,  5,  4,  3,  2, 1],
        [2,  4,  6,  8, 10,  8,  6,  4, 2],
        [3,  6,  9, 12, 15, 12,  9,  6, 3],
        [4,  8, 12, 16, 20, 16, 12,  8, 4],
        [5, 10, 15, 20, 25, 20, 15, 10, 5],
        [4,  8, 12, 16, 20, 16, 12,  8, 4],
        [3,  6,  9, 12, 15, 12,  9,  6, 3],
        [2,  4,  6,  8, 10,  8,  6,  4, 2],
        [1,  2,  3,  4,  5,  4,  3,  2, 1]
    ], dtype=np.float32)
    kernel /= kernel.sum()
    for _ in range(iterations):
        array_list = []
        for y in range(kernel.shape[0]):
            temp_array = np.roll(image_array, shift=y - kernel.shape[0] // 2, axis=0)
            for x in range(kernel.shape[1]):
                temp_array_x = np.roll(temp_array, shift=x - kernel.shape[1] // 2, axis=1)
                weighted_temp_array_x = temp_array_x * kernel[y, x]
                array_list.append(weighted_temp_array_x)

        array_list = np.array(array_list)
        image_array = np.sum(array_list, axis=0)
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    return image_array


class ImageProcessor:
    """Class that organizes processing, saving and cleanup of images"""

    def __init__(self, source_dir: str, temp_dir: str):
        self.source_dir = source_dir
        self.temp_dir = temp_dir

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup_temp()
        logger.info("Program exited")

    def process_images(self):
        idx = 1000
        images = scan_directory_for_images(self.source_dir)
        screenshot_present = 'screenshot.png' in [os.path.basename(image) for image in images]
        image_count = len(images) - screenshot_present
        logger.info(f"Found {image_count} images")
        for i, image_path in enumerate(scan_directory_for_images(self.source_dir)):
            image = Image.open(image_path)
            image_array = np.array(image)
            # Skip screenshot for readme file
            if os.path.basename(image_path) == "screenshot.png":
                continue
            # Skip images with resolution less than 640x480
            if image_array.shape[0] < 480 or image_array.shape[1] < 640:
                logger.warning(f"Image {os.path.basename(image_path)} skipped due to resolution less than 640x480.")
                continue
            # Drop alpha channel if image comes with RGBA
            image_array = image_array[:, :, :3]
            original_array = image_array.copy()
            idx += 1
            vertically_array = slice_array_vertically(image_array)
            self.save_image(vertically_array, os.path.basename(image_path), effect='ver_slice', i=idx)
            idx += 1
            horizontally_array = slice_array_horizontally(vertically_array)
            self.save_image(horizontally_array, os.path.basename(image_path), effect='hor_slice', i=idx)
            idx += 1
            color_scheme_array = apply_color_scheme(image_array)
            self.save_image(color_scheme_array, os.path.basename(image_path), effect='color_scheme', i=idx)
            idx += 1
            concatenated_array = concatenate_arrays((original_array, color_scheme_array,
                                                     vertically_array, horizontally_array))
            self.save_image(concatenated_array, os.path.basename(image_path), effect='full_concat', i=idx)
            idx += 1
            blured_array = blur_array(original_array)
            self.save_image(blured_array, os.path.basename(image_path), effect='blurred', i=idx)
            idx += 1
            rotated_array = rotate_array(original_array)
            self.save_image(rotated_array, os.path.basename(image_path), effect='rotated', i=idx)
            logger.success(f"Image No {i + 1} processed")
        logger.success(f"{idx - 1000} images saved")
        viewer = ImageViewer(self.source_dir, self.temp_dir)
        viewer.mainloop()

    def save_image(self, image_array: np.ndarray, filename: str, effect: str, i: int) -> None:
        try:
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            save_path = os.path.join(self.temp_dir, f"{i}_{effect}_{base_filename}.jpeg")
            Image.fromarray(image_array).save(save_path)
            logger.debug(f"Image saved successfully at {save_path}")

        except Exception as e:
            logger.error(f"Failed to save image '{filename}' with effect '{effect}': {e}")

    @log_decorator
    def cleanup_temp(self):
        for f in os.listdir(self.temp_dir):
            if f == '.gitignore':
                continue
            try:
                os.remove(os.path.join(self.temp_dir, f))
            except Exception as e:
                logger.error(f"Error deleting file {f}: {e}")


class ImageViewer(tk.Tk):
    """Class for simple image viewer to show image processing results"""

    def __init__(self, images_dir, temp_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.images_dir = images_dir
        self.temp_dir = temp_dir
        self.temp_images = scan_directory_for_images(self.temp_dir)
        self.current_image_index = 0

        self.title("Image Slicer")
        self.geometry(Settings.viewer_size)
        self.resizable(Settings.resizable_width, Settings.resizable_height)

        self.create_thumbnail_bar()
        self.create_image_display()
        self.create_navigation_buttons()
        logger.success("GUI launched")

    def create_thumbnail_bar(self):
        scroll_frame = tk.Frame(self)
        scroll_frame.pack(side=tk.TOP, fill=tk.X, expand=True)
        canvas = tk.Canvas(scroll_frame, height=Settings.thumbnail_size[1] + 20)
        scrollbar = tk.Scrollbar(scroll_frame, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        thumbnails_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=thumbnails_frame, anchor='nw')
        self.load_thumbnails(thumbnails_frame, canvas)
        thumbnails_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def load_thumbnails(self, thumbnails_frame, canvas):
        x_pos = 5
        for image_file in scan_directory_for_images(self.images_dir):
            # Skip screenshot image
            if os.path.basename(image_file) == "screenshot.png":
                continue
            img = Image.open(image_file)
            img.thumbnail(Settings.thumbnail_size)
            photo_img = ImageTk.PhotoImage(img)
            label = tk.Label(thumbnails_frame, image=photo_img,
                             bg='black')
            label.image = photo_img
            label.place(x=x_pos, y=5)
            x_pos += Settings.thumbnail_size[0] + 10
        thumbnails_frame.config(width=x_pos, height=Settings.thumbnail_size[1] + 20)
        canvas.configure(scrollregion=(0, 0, x_pos, Settings.thumbnail_size[1] + 20))

    def create_image_display(self):
        self.image_panel = tk.Label(self)
        self.image_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.update_image_display()

    def update_image_display(self):
        if self.temp_images:
            img_path = self.temp_images[self.current_image_index]
            img = Image.open(img_path)
            img.thumbnail((Settings.image_size[0], Settings.image_size[1]), Image.Resampling.LANCZOS)
            photo_img = ImageTk.PhotoImage(img)
            self.image_panel.configure(image=photo_img)
            self.image_panel.image = photo_img
            file_name = os.path.basename(img_path)
            if hasattr(self, 'file_name_label'):
                self.file_name_label.configure(text=file_name)
            else:
                self.file_name_label = tk.Label(self, text=file_name)
                self.file_name_label.pack(side=tk.TOP)

    def create_navigation_buttons(self):
        buttons_frame = tk.Frame(self)
        buttons_frame.pack(side=tk.BOTTOM)

        prev_button = tk.Button(buttons_frame, text="Previous", width=25, command=self.prev_image)
        prev_button.pack(side=tk.LEFT, padx=100, pady=20)

        next_button = tk.Button(buttons_frame, text="Next", width=25, command=self.next_image)
        next_button.pack(side=tk.RIGHT, padx=100, pady=20)

    def prev_image(self):
        self.current_image_index = (self.current_image_index - 1) % len(self.temp_images)
        self.update_image_display()

    def next_image(self):
        self.current_image_index = (self.current_image_index + 1) % len(self.temp_images)
        self.update_image_display()


# Set log level and log file
logger.remove()
logger.add(sys.stderr, level=Settings.log_level)
logger.add(Settings.log_file_path, level="ERROR")

# Check if arguments where provided in command line and handle them
num_args = len(sys.argv) - 1
if num_args > 2:
    logger.error("Too many arguments provided. Only the first two will be considered.")
for arg in sys.argv[1:3]:
    if arg.isdigit():
        slice_count = int(arg)
        if 20 <= slice_count <= 200:
            Settings.slice_count = slice_count
            logger.success(f"slice_count set to {slice_count}")
        else:
            logger.error("slice_count argument must be between 20 and 200")
    elif arg in ['gray', 'red', 'green', 'blue']:
        Settings.colour_scheme = arg
        logger.success(f"colour_scheme set to {arg}")
    else:
        logger.error(f"Unrecognized argument: {arg}")

# Set restrictions to slice count
Settings.slice_count = max(20, min(Settings.slice_count, 200))


def main():

    logger.info(f"Program started")
    with ImageProcessor(Settings.source_dir, Settings.temp_dir) as processor:
        processor.process_images()


if __name__ == "__main__":
    main()
