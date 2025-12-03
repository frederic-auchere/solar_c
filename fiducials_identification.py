import os
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # Had issues with other backends on my machine
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import time
from utils import fit_circle_to_points

import getpass
USER = getpass.getuser()



def interactive_fiducial_measurement(image_path, file_name, max_num_circles=3):
    """
    Shows an image and lets user click points to fit fiducials manually
    """
    try:
        image_data = np.load(image_path)
    except Exception as e:
        print(f"Couldn't load image: {e}")
        return []
    shift_pressed = False
    zoom_was_active = False
    fig, ax = plt.subplots(figsize=(15, 15))  # large figure for easy clicking
    # activer le zoom automatiquement
    try:
        fig.canvas.manager.toolbar.zoom()  # version la plus fréquente
    except:
        fig.canvas.toolbar.zoom()  # fallback si manager n'existe pas
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
        if event.inaxes != ax or event.button != 1:
            return

        # SHIFT → zoom temporairement désactivé → on prend un point
        if shift_pressed:
            x, y = event.xdata, event.ydata
            clicked_points.append((x, y))
            plot, = ax.plot(x, y, 'bo', markersize=2)
            point_plots.append(plot)
            redraw_preview()
            return

    def handle_keyrelease(event):
        nonlocal shift_pressed
        if event.key == 'shift':
            shift_pressed = False
    def handle_keypress(event):
        nonlocal shift_pressed, zoom_was_active
        toolbar = plt.get_current_fig_manager().toolbar
        if event.key == 'shift':
            shift_pressed = True
            # Si zoom actif → désactivation temporaire
            if toolbar.mode == 'zoom rect':
                zoom_was_active = True
                toolbar.zoom()  # this toggles zoom *off*
            else:
                zoom_was_active = False
            return

    def handle_keyrelease(event):
        nonlocal shift_pressed, zoom_was_active
        toolbar = plt.get_current_fig_manager().toolbar

        if event.key == 'shift':
            shift_pressed = False

            # Si on avait désactivé le zoom, on le remet
            if zoom_was_active:
                toolbar.zoom()  # toggles zoom *on*
                zoom_was_active = False
            return
        elif event.key == 'x':
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

                    # Ajouter le cercle final
                    final_circle = Circle((cx, cy), r, fill=False, color='lime', linewidth=2)
                    ax.add_patch(final_circle)

                    # Redessiner
                    fig.canvas.draw_idle()

                    # --- RESET VIEW (Home) après chaque cercle ---
                    ax.set_xlim(0, image_data.shape[1])
                    ax.set_ylim(0, image_data.shape[0])
                    fig.canvas.draw_idle()

                    if len(finished_circles) >= max_num_circles:
                        print("All circles done. Close the window to continue...")
                        plt.close(fig)
                        return
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
    fig.canvas.mpl_connect('key_release_event', handle_keyrelease)
    plt.show(block=True)
    return finished_circles



# Main script starts here
print("=== Fiducials Fitting Tool ===")





folder_path = input("️  Enter folder path containing NPY files: ").strip().strip("'\"")


# --- Lock system to avoid multiple users writing simultaneously ---
lock_path = os.path.join(folder_path, ".fiducials_lock")
with open(lock_path, "w") as f:
    f.write(USER)
LOCK_TIMEOUT = 3600 # 1 heure en secondes

if os.path.exists(lock_path):
    lock_age = time.time() - os.path.getmtime(lock_path)
    if lock_age > LOCK_TIMEOUT:
        print("Lock file is old. Removing orphan lock.")
        os.remove(lock_path)

# Create the lock file
try:
    if os.path.exists(lock_path):
        with open(lock_path, "r") as f:
            lock_user = f.read().strip()

        if lock_user == USER:
            print("You already hold the lock. Continuing...")
        else:
            print(f"Folder is locked by another user: {lock_user}. Waiting...")
            while os.path.exists(lock_path):
                time.sleep(2)
except Exception as e:
    print(f"Could not create lock file: {e}")
    exit()
if not os.path.isdir(folder_path):
    print("Folder not found")
    exit()

circles_per_image = 3
excel_output_path = os.path.join(folder_path, "fiducials_measurements.xlsx")

# --- Charger les anciens résultats si l'Excel existe ---
if os.path.isfile(excel_output_path):
    try:
        previous_results = pd.read_excel(excel_output_path)
        processed_files = set(previous_results["filename_npy"].tolist())
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

    data_row = {
        "filename_npy": current_file,
        "filename": current_file.replace("_qpsix_max.npy", ".datx").replace("_", "/", 1)
    }
    for fiducial_idx, (center_x, center_y) in enumerate(measured_fiducials, start=1):
        data_row[f"xc_{fiducial_idx}"] = center_x
        data_row[f"yc_{fiducial_idx}"] = center_y

    measurement_results.append(data_row)
    processed_files.add(current_file)

    # Sauvegarde après chaque image
    results_df = pd.DataFrame(measurement_results)
    results_df.to_excel(excel_output_path, index=False)
    print(f" Saved current results to Excel: {excel_output_path}")
# --- Release lock ---
try:
    if os.path.exists(lock_path):
        with open(lock_path, "r") as f:
            lock_user = f.read().strip()
        if lock_user == USER:
            os.remove(lock_path)
            print("Lock released.")
except Exception as e:
    print(f"Could not remove lock file: {e}")
print("\n All files processed!")


