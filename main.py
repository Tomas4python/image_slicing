import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


class Settings:
    """Class to keep configurable settings in one place"""

    viewer_size = '1200x1200'  # The size of whole viewer window
    resizable_width = False  # Default viewer window is constant size, set True if you want resize window with mouse
    resizable_height = False
    thumbnail_size = (320, 240)  # The thumbnails size of the original images at top row of the viewer
    image_size = (1200, 900)  # The size of modified images in the middle of the viewer
    slice_count = 20  # Count of the slices is allowed from 20 to 50
    source_dir = 'images'  # Directory for original images
    temp_dir = 'temp'  # Directory for modified images, images are deleted when viewer is closed


def scan_directory_for_images(directory: str) -> list[str]:

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    return [os.path.join(directory, f) for f in os.listdir(directory) if
            f.endswith(supported_extensions) and os.path.isfile(os.path.join(directory, f))]


class ImageProcessor:
    """Class that does main job - slices images"""

    def __init__(self, source_dir: str, temp_dir: str):
        self.source_dir = source_dir
        self.temp_dir = temp_dir

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup_temp()

    def process_images(self):
        for idx, image_path in enumerate(scan_directory_for_images(self.source_dir)):
            image = Image.open(image_path)
            image_array = np.array(image)
            self.save_image(image_array, os.path.basename(image_path), vertical=True, i=idx)
            self.save_image(image_array, os.path.basename(image_path), vertical=False, i=idx)
        viewer = ImageViewer(self.source_dir, self.temp_dir)
        viewer.mainloop()

    def save_image(self, image_array: np.ndarray, filename: str, vertical: bool, i: int) -> None:
        orientation = 'first_cut' if vertical else 'second_cut'
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        save_path = os.path.join(self.temp_dir, f"{i}_{orientation}_{base_filename}.jpeg")
        Image.fromarray(image_array).save(save_path)

    def cleanup_temp(self):
        for f in os.listdir(self.temp_dir):
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
