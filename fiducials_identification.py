import os
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # Had issues with other backends on my machine
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
from optical.utils import fit_circle_to_points


def interactive_fiducial_measurement(image_path, file_name, max_num_circles=3):
    """
    Shows an image and lets user click points to fit fiducials manually
    """
    try:
        image_data = np.load(image_path)
    except Exception as e:
        print(f"Couldn't load image: {e}")
        return []

    fig, ax = plt.subplots(figsize=(15, 15))  # large figure for easy clicking
    vmin, vmax = np.percentile(image_data, (5, 95))
    ax.imshow(image_data,origin='lower',cmap=plt.get_cmap('gray'),vmin=vmin, vmax=vmax)
    ax.set_title(f"Working on: {file_name}\nClick points, press Enter to confirm circle, Backspace to undo last click")
    plt.axis("image")

    clicked_points = []
    finished_circles = []
    point_plots = []  # store plotted points for easy removal

    def redraw_preview():
        """Redraw the preview circle if possible."""
        for patch in ax.patches[:]:
            patch.remove()

        if len(clicked_points) >= 3:
            x_values = np.array([pt[0] for pt in clicked_points])
            y_values = np.array([pt[1] for pt in clicked_points])
            try:
                cx, cy, r = fit_circle_to_points(x_values, y_values)
                preview_circle = Circle((cx, cy), r, fill=False, color='cyan', linewidth=2, linestyle='--')
                ax.add_patch(preview_circle)
            except Exception:
                pass

        fig.canvas.draw_idle()

    def handle_mouse_click(event):
        toolbar = plt.get_current_fig_manager().toolbar
        if toolbar is not None and toolbar.mode != '':
            return

        if event.inaxes == ax and event.button == 1:  # left click
            x_pos, y_pos = event.xdata, event.ydata
            clicked_points.append((x_pos, y_pos))
            plot, = ax.plot(x_pos, y_pos, 'bo', markersize=2)
            point_plots.append(plot)
            redraw_preview()

    def handle_keypress(event):
        if event.key == 'enter':
            if len(clicked_points) >= 3:
                x_values = np.array([pt[0] for pt in clicked_points])
                y_values = np.array([pt[1] for pt in clicked_points])

                try:
                    cx, cy, r = fit_circle_to_points(x_values, y_values)
                    circle_number = len(finished_circles) + 1

                    print(f"{file_name} | Circle #{circle_number}: center=({cx:.2f}, {cy:.2f}), radius={r:.2f}")
                    finished_circles.append((cx, cy))

                    # Clear points for next circle
                    clicked_points.clear()
                    for p in point_plots:
                        p.remove()
                    point_plots.clear()

                    final_circle = Circle((cx, cy), r, fill=False, color='lime', linewidth=2)
                    ax.add_patch(final_circle)
                    fig.canvas.draw_idle()

                    if len(finished_circles) >= max_num_circles:
                        print("All circles done. Close the window to continue...")
                except Exception as e:
                    print(f"Error fitting circle: {e}")
            else:
                print("Need at least 3 points to make a circle")

        elif event.key == 'backspace':
            # Undo last click
            if clicked_points:
                clicked_points.pop()
                if point_plots:
                    last_plot = point_plots.pop()
                    last_plot.remove()
                print("Last click undone.")
                redraw_preview()
            else:
                print("No points to remove.")

    fig.canvas.mpl_connect('button_press_event', handle_mouse_click)
    fig.canvas.mpl_connect('key_press_event', handle_keypress)

    plt.show(block=True)
    return finished_circles



# Main script starts here
print("=== Fiducials Fitting Tool ===")

folder_path = input("️  Enter folder path containing NPY files: ").strip().strip("'\"")

if not os.path.isdir(folder_path):
    print("Folder not found")
    exit()

circles_per_image = 3
excel_output_path = os.path.join(folder_path, "fiducials_measurements.xlsx")

# --- Charger les anciens résultats si l'Excel existe ---
if os.path.isfile(excel_output_path):
    try:
        previous_results = pd.read_excel(excel_output_path)
        processed_files = set(previous_results["filename"].tolist())
        measurement_results = previous_results.to_dict(orient="records")
        print(f" Loaded existing Excel with {len(processed_files)} measured files.")
    except Exception as e:
        print(f" Could not read Excel file ({e}). Starting fresh.")
        processed_files = set()
        measurement_results = []
else:
    processed_files = set()
    measurement_results = []
    print(" No existing Excel found, starting a new one.")

# --- Trouver tous les fichiers .npy ---
npy_file_list = [f for f in os.listdir(folder_path) if f.lower().endswith("qpsix_max.npy")]
if not npy_file_list:
    print("No NPY files found in the specified folder")
    exit()

print(f"Found {len(npy_file_list)} NPY files to process ({len(processed_files)} already measured).")

print("\nHow to use:")
print(f" • Click on at least 3 points to define each fiducial")
print(f" • Press Enter to confirm and save a circle (need {circles_per_image} total)")
print(" • Press Backspace to undo last click")
print(" • Close the window when you're done with an image\n")

# --- Boucle principale ---
for current_file in npy_file_list:
    if current_file in processed_files:
        print(f"  Skipping already processed file: {current_file}")
        continue

    print(f"\nProcessing: {current_file}")
    full_image_path = os.path.join(folder_path, current_file)

    measured_fiducials = interactive_fiducial_measurement(full_image_path, current_file, circles_per_image)

    data_row = {"filename": current_file.replace("_qpsix_max.npy", ".datx").replace("_", "/", 1)}
    for fiducial_idx, (center_x, center_y) in enumerate(measured_fiducials, start=1):
        data_row[f"xc_{fiducial_idx}"] = center_x
        data_row[f"yc_{fiducial_idx}"] = center_y

    measurement_results.append(data_row)
    processed_files.add(current_file)

    # Sauvegarde après chaque image
    results_df = pd.DataFrame(measurement_results)
    results_df.to_excel(excel_output_path, index=False)
    print(f" Saved current results to Excel: {excel_output_path}")

print("\n All files processed!")


