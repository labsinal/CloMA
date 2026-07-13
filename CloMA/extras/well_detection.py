"""
Module for detecting wells on a whole plate image
"""

######################################
# imports
from numpy import ndarray
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import cv2
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
import os 
from sys import exit
##########################
# define functions

def create_edges(image: ndarray, sigma: float) -> ndarray:
    """
    Function that creates edges from a plate image

    Args:
        image (ndarray): Grayscale image of the plate
        sigma (float): Intensity of gaussian blur
    Return (ndarray): Edges of the original image
    """
    # Detect edges
    return canny(image, sigma=sigma)

def create_hough_detection(edges: ndarray, radius: int) -> ndarray:
    """
    Function that creates hough detection from a plate image edges

    Args:
        edges (ndarray): Edges of the plate image
        radius (int): radius for hough detection
    Return (ndarray): hough detection
    """
    # Detect edges
    return hough_circle(edges, [radius])

def detect_wells(image: ndarray, well_radius: int, sigma:float) -> list[ndarray]:
    """
    Function that detects wells in a plate photograph

    Args:
        image (ndarray): Image of plate
        well_radius (int): Radius of wells in pixels
        sigma (float): Sigma value for gaussian blur in the process
    Return (list[ndarray]): List of detected wells as numpy arrays
    """

    # convert images to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) >= 3 else image

    edges = create_edges(image_gray, sigma)

    # Apply Hough Transform to detect circles
    hough_res = create_hough_detection(edges, well_radius)

    # Extract circles
    accum, cx, cy, radii = hough_circle_peaks(hough_res, 
                                              [well_radius], 
                                              min_xdistance=well_radius, 
                                              min_ydistance=well_radius)

    # Filter circles that do not overlap and are entirelly contained in the image
    max_y, max_x, _= image.shape
    circles_info = [{"cx": x, "cy": y, "radii": r} for x, y, r in zip(cx, cy, radii)]
    circles_filter = lambda circle : circle["cx"] - circle["radii"] > 0 and \
                                      circle["cy"] - circle["radii"] > 0 and \
                                      circle["cx"] + circle["radii"] < max_x and \
                                      circle["cy"] + circle["radii"] < max_y
    
    # sort circles by row (y) then column (x) to preserve grid order
    row_height = max(well_radius * 1.5, 1)
    circles_info.sort(key=lambda circle: (int(circle["cy"] / row_height), circle["cx"]))

    circles_info = list(filter(circles_filter, circles_info))

    # Create list of image crops of detected circles
    circles = [image[circle["cy"] - circle["radii"]:circle["cy"] + circle["radii"],
                      circle["cx"] - circle["radii"]:circle["cx"] + circle["radii"]]
                        for circle in circles_info]

    # If no circles detected warn the user and keep the GUI open
    if circles == []:
        print("No circles detected with given parameters.")
        return []

    # Return circles list
    return circles

###################################
# define GUI functions

class WellDetectorGUI:

    def __init__(self, root, image_path=None, output_folder=None):

        # Create window
        self.root = root
        self.root.title("Well Detection Tester")
        self.root.resizable(True, True)

        self.image = None
        self.crops = []

        self.image_path = image_path
        self.output_folder = output_folder

        ###################################
        # Controls frame
        control = tk.Frame(root)
        control.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        tk.Button(control, text="Load Image", command=self.load_image).pack(pady=5)

        tk.Label(control, text="Sigma").pack()
        self.sigma_entry = tk.Entry(control)
        self.sigma_entry.insert(0, "3")
        self.sigma_entry.pack()

        tk.Label(control, text="Well radius").pack()
        self.radius_entry = tk.Entry(control)
        self.radius_entry.insert(0, "40")
        self.radius_entry.pack()

        tk.Button(control, text="Run Detection", command=self.run_detection).pack(pady=10)

        self.info_label = tk.Label(control, text="Circles found: 0")
        self.info_label.pack(pady=5)

        tk.Button(control, text="Save Cropped Wells", command=self.save_wells).pack(pady=10)

        ###################################
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(2, 2, figsize=(7, 7))

        for a in self.ax.flatten():
            a.axis("off")

        self.ax[0,0].set_title("Original")
        self.ax[0,1].set_title("Edges")
        self.ax[1,0].set_title("Hough")
        self.ax[1,1].set_title("Example Crop")

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        if image_path is not None:
            self.load_image_from_path(image_path)

    ###################################
    # Image loader

    def load_image(self):

        path = filedialog.askopenfilename()

        if path:
            self.load_image_from_path(path)

    def load_image_from_path(self, path):

        self.image = cv2.imread(path)

        if self.image is None:
            messagebox.showerror("Error", f"Could not load image:\n{path}")
            return

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.update_plot(original=self.image)

    ###################################
    # Run detection

    def run_detection(self):

        if self.image is None:
            messagebox.showerror("Error", "Load an image first")
            return

        try:
            sigma = float(self.sigma_entry.get())
            radius = int(self.radius_entry.get())
        except:
            messagebox.showerror("Error", "Invalid parameters")
            return

        edges = create_edges(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) 
                             if len(self.image.shape) >= 3 else self.image, sigma)
        hough = create_hough_detection(edges, radius)
        crops = detect_wells(self.image, radius, sigma)

        self.crops = crops

        n = len(crops)
        self.info_label.config(text=f"Circles found: {n}")

        example_crop = crops[0] if n > 0 else None

        self.update_plot(
            original=self.image,
            edges=edges,
            hough=hough[0],
            crop=example_crop
        )

    ###################################
    # Update matplotlib plots

    def update_plot(self, original=None, edges=None, hough=None, crop=None):

        if original is not None:
            self.ax[0,0].cla()
            self.ax[0,0].imshow(original)

            # Enable axis and show pixel size
            self.ax[0,0].set_title("Original")
            self.ax[0,0].tick_params(axis='both', labelsize=8)

        if edges is not None:
            self.ax[0,1].cla()
            self.ax[0,1].imshow(edges, cmap="gray")
            self.ax[0,1].set_title("Edges")
            self.ax[0,1].axis("off")

        if hough is not None:
            self.ax[1,0].cla()
            self.ax[1,0].imshow(hough, cmap="magma")
            self.ax[1,0].set_title("Hough")
            self.ax[1,0].axis("off")

        if crop is not None:
            self.ax[1,1].cla()
            self.ax[1,1].imshow(crop)
            self.ax[1,1].set_title("Example Crop")
            self.ax[1,1].axis("off")
        else:
            self.ax[1,1].cla()
            self.ax[1,1].set_title("Example Crop")
            self.ax[0,0].tick_params(axis='both', labelsize=8)

        self.canvas.draw()

    ###################################
    # Save wells

    def save_wells(self):

        if len(self.crops) == 0:
            messagebox.showinfo("Info", "No wells to save")
            return

        folder = self.output_folder

        if folder is None:
            folder = filedialog.askdirectory()
            if not folder:
                return

        os.makedirs(folder, exist_ok=True)

        for i, crop in enumerate(self.crops):

            path = os.path.join(folder, f"well_{i:04d}.tiff")

            cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        messagebox.showinfo("Done", f"Saved {len(self.crops)} wells")
        
        if self.output_folder is not None:
            self.root.destroy()

# end of current module

if __name__ == "__main__":
    root = tk.Tk()
    app = WellDetectorGUI(root)
    root.mainloop()