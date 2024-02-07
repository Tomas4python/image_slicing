import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


class Settings:
    """Class to keep useful settings in one place"""

    viewer_size = '1000x1000'  # The size of whole viewer window
    resizable_width = False  # Default viewer window is constant size, set True if you want resize window with mouse
    resizable_height = False
    thumbnail_size = (320, 240)  # The thumbnails size of the original images at top row of the viewer
    image_size = (800, 600)  # The size of modified images in the middle of the viewer
    colour_scheme = 'gray'  # Default colour scheme for third processing, available: 'gray', 'black', 'red', 'green',
    # 'blue'
    slice_count = 200  # Count of the slices is allowed from 20 to 200
    source_dir = 'images'  # Directory for original images
    temp_dir = 'temp'  # Directory for modified images, images are deleted when viewer is closed


def scan_directory_for_images(directory: str) -> list[str]:

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    return [os.path.join(directory, f) for f in os.listdir(directory) if
            f.endswith(supported_extensions) and os.path.isfile(os.path.join(directory, f))]


def slice_array_vertically(image_array: np.ndarray) -> np.ndarray:
    """Slice image ndarray into a number of slices and rearrange them first even, then odd."""

    slice_height = image_array.shape[0] // settings.slice_count
    slices = [image_array[i*slice_height:(i+1)*slice_height, :] for i in range(settings.slice_count)]
    rearranged_slices = slices[::2] + slices[1::2]
    return np.vstack(rearranged_slices)


def slice_array_horizontally(image_array: np.ndarray) -> np.ndarray:
    """Slice image ndarray into a number of slices and rearrange them first even, then odd."""

    slice_width = image_array.shape[1] // settings.slice_count
    slices = [image_array[:, i*slice_width:(i+1)*slice_width] for i in range(settings.slice_count)]
    rearranged_slices = slices[::2] + slices[1::2]
    return np.hstack(rearranged_slices)


def apply_color_scheme(image_array: np.ndarray) -> np.ndarray:
    """Apply a color scheme to an image ndarray and return a 3D array according to the scheme choice."""

    if settings.colour_scheme == 'gray':
        grayscale = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
        return np.repeat(grayscale[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

    elif settings.colour_scheme == 'black':
        grayscale = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
        black_and_white = grayscale > 115
        black_and_white = (black_and_white * 255).astype(np.uint8)
        return np.repeat(black_and_white[:, :, np.newaxis], 3, axis=2)

    elif settings.colour_scheme == 'red':
        red_only = image_array.copy()
        red_only[:, :, 1:3] = 0
        return red_only

    elif settings.colour_scheme == 'green':
        green_only = image_array.copy()
        green_only[:, :, 0] = 0
        green_only[:, :, 2] = 0
        return green_only

    elif settings.colour_scheme == 'blue':
        blue_only = image_array.copy()
        blue_only[:, :, 0:2] = 0
        return blue_only

    else:
        raise ValueError("Unrecognized color scheme. Choose from 'gray', 'black', 'red', 'green', 'blue'.")


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


class ImageProcessor:
    """Class that organizes processing, saving and cleanup of images"""

    def __init__(self, source_dir: str, temp_dir: str):
        self.source_dir = source_dir
        self.temp_dir = temp_dir

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup_temp()

    def process_images(self):
        idx = 1000
        for image_path in scan_directory_for_images(self.source_dir):
            image = Image.open(image_path)
            image_array = np.array(image)
            original_array = image_array.copy()
            idx += 1
            vertically_array = slice_array_vertically(image_array)
            self.save_image(vertically_array, os.path.basename(image_path), effect='ver_slice', i=idx)
            idx += 1
            horizontally_array = slice_array_horizontally(vertically_array)
            self.save_image(horizontally_array, os.path.basename(image_path), effect='hor_slice', i=idx)
            idx += 1
            black_and_white_array = apply_color_scheme(image_array)
            self.save_image(black_and_white_array, os.path.basename(image_path), effect='color_scheme', i=idx)
            idx += 1
            concatenated_array = concatenate_arrays((original_array, black_and_white_array, vertically_array, horizontally_array))
            self.save_image(concatenated_array, os.path.basename(image_path), effect='full_concat', i=idx)
        viewer = ImageViewer(self.source_dir, self.temp_dir)
        viewer.mainloop()

    def save_image(self, image_array: np.ndarray, filename: str, effect: str, i: int) -> None:
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        save_path = os.path.join(self.temp_dir, f"{i}_{effect}_{base_filename}.jpeg")
        Image.fromarray(image_array).save(save_path)

    def cleanup_temp(self):
        for f in os.listdir(self.temp_dir):
            if f == '.gitignore':
                continue
            try:
                os.remove(os.path.join(self.temp_dir, f))
            except Exception as e:
                print(f"Error deleting file {f}: {e}")


class ImageViewer(tk.Tk):
    """Class for simple image viewer to show image processing results"""

    def __init__(self, images_dir, temp_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.images_dir = images_dir
        self.temp_dir = temp_dir
        self.temp_images = scan_directory_for_images(self.temp_dir)
        self.current_image_index = 0

        self.title("Image Slicer")
        self.geometry(settings.viewer_size)
        self.resizable(settings.resizable_width, settings.resizable_height)

        self.create_thumbnail_bar()
        self.create_image_display()
        self.create_navigation_buttons()

    def create_thumbnail_bar(self):
        scroll_frame = tk.Frame(self)
        scroll_frame.pack(side=tk.TOP, fill=tk.X, expand=True)
        canvas = tk.Canvas(scroll_frame, height=settings.thumbnail_size[1] + 20)
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
            img = Image.open(image_file)
            img.thumbnail(settings.thumbnail_size)
            photo_img = ImageTk.PhotoImage(img)
            label = tk.Label(thumbnails_frame, image=photo_img,
                             bg='black')
            label.image = photo_img
            label.place(x=x_pos, y=5)
            x_pos += settings.thumbnail_size[0] + 10
        thumbnails_frame.config(width=x_pos, height=settings.thumbnail_size[1] + 20)
        canvas.configure(scrollregion=(0, 0, x_pos, settings.thumbnail_size[1] + 20))

    def create_image_display(self):
        self.image_panel = tk.Label(self)
        self.image_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.update_image_display()

    def update_image_display(self):
        if self.temp_images:
            img = Image.open(self.temp_images[self.current_image_index])
            img.thumbnail((settings.image_size[0], settings.image_size[1]), Image.Resampling.LANCZOS)
            photo_img = ImageTk.PhotoImage(img)
            self.image_panel.configure(image=photo_img)
            self.image_panel.image = photo_img

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


settings = Settings()


def main():
    source_dir = settings.source_dir
    temp_dir = settings.temp_dir
    with ImageProcessor(source_dir, temp_dir) as processor:
        processor.process_images()


if __name__ == "__main__":
    main()
