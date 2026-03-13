import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
import os

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class WellDetectorGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Well Detection Tester")

        self.image = None
        self.circles = []
        self.crops = []

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
        self.canvas.get_tk_widget().pack(side=tk.LEFT)

    ###################################
    # Image loader

    def load_image(self):

        path = filedialog.askopenfilename()

        if path:
            self.image = cv2.imread(path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.update_plot(original=self.image)

    ###################################
    # Core detection

    def detect_wells(self, sigma, radius):

        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        edges = canny(gray, sigma=sigma)

        hough_res = hough_circle(edges, [radius])

        accum, cx, cy, radii = hough_circle_peaks(
            hough_res,
            [radius],
            min_xdistance=radius,
            min_ydistance=radius
        )

        max_y, max_x, _ = self.image.shape

        circles_info = [
            {"cx": x, "cy": y, "r": r}
            for x, y, r in zip(cx, cy, radii)
        ]

        circles_info = [
            c for c in circles_info
            if c["cx"]-c["r"]>0 and c["cy"]-c["r"]>0
            and c["cx"]+c["r"]<max_x and c["cy"]+c["r"]<max_y
        ]

        crops = [
            self.image[
                c["cy"]-c["r"]:c["cy"]+c["r"],
                c["cx"]-c["r"]:c["cx"]+c["r"]
            ]
            for c in circles_info
        ]

        return edges, hough_res[0], circles_info, crops

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

        edges, hough, circles, crops = self.detect_wells(sigma, radius)

        self.circles = circles
        self.crops = crops

        n = len(circles)
        self.info_label.config(text=f"Circles found: {n}")

        example_crop = crops[0] if n > 0 else None

        self.update_plot(
            original=self.image,
            edges=edges,
            hough=hough,
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
            self.ax[1,1].axis("off")

        self.canvas.draw()

    ###################################
    # Save wells

    def save_wells(self):

        if len(self.crops) == 0:
            messagebox.showinfo("Info", "No wells to save")
            return

        folder = filedialog.askdirectory()

        if not folder:
            return

        os.makedirs(folder, exist_ok=True)

        for i, crop in enumerate(self.crops):

            path = os.path.join(folder, f"well_{i:04d}.tiff")

            cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        messagebox.showinfo("Done", f"Saved {len(self.crops)} wells")


###################################
# Run GUI

if __name__ == "__main__":

    root = tk.Tk()
    app = WellDetectorGUI(root)
    root.mainloop()