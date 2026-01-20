import os
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # Had issues with other backends on my machine
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
from optical.utils import fit_circle_to_points
import time

import getpass
USER = getpass.getuser()



def interactive_fiducial_measurement(image_path, file_name, max_num_circles=3):
    """
    Shows an image and lets user click points to fit fiducials manually.
    SHIFT = click points
    X     = validate circle
    BACKSPACE = undo last point
    """
    try:
        image_data = np.load(image_path)
    except Exception as e:
        print(f"Couldn't load image: {e}")
        return []

    fig, ax = plt.subplots(figsize=(15, 15))

    # --- Force zoom ON at startup (cross-platform) ---
    toolbar = plt.get_current_fig_manager().toolbar
    if toolbar and toolbar.mode != 'zoom rect':
        toolbar.zoom()

    vmin, vmax = np.percentile(image_data, (5, 95))
    ax.imshow(image_data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(
        f"Working on: {file_name}\n"
        "SHIFT + Click = add points | X = validate circle | Backspace = undo"
    )
    plt.axis("image")

    clicked_points = []
    finished_circles = []
    point_plots = []
    preview_lines = []

    shift_pressed = False
    zoom_was_active = False

    # ------------------------------------------------------------
    def redraw_preview():
        # Remove previous preview patches
        for patch in ax.patches[:]:
            patch.remove()

        nonlocal preview_lines
        for line in preview_lines:
            line.remove()
        preview_lines = []

        if len(clicked_points) >= 3:
            try:
                x = np.array([p[0] for p in clicked_points])
                y = np.array([p[1] for p in clicked_points])
                cx, cy, r = fit_circle_to_points(x, y)

                preview_circle = Circle(
                    (cx, cy), r,
                    fill=False, color='cyan',
                    linestyle='--', linewidth=2
                )
                ax.add_patch(preview_circle)

                h, = ax.plot([cx - r, cx + r], [cy, cy], 'm--', lw=1)
                v, = ax.plot([cx, cx], [cy - r, cy + r], 'm--', lw=1)
                preview_lines = [h, v]

            except Exception:
                pass

        fig.canvas.draw_idle()

    # ------------------------------------------------------------
    def handle_mouse_click(event):
        if event.inaxes != ax or event.button != 1:
            return

        if shift_pressed:
            clicked_points.append((event.xdata, event.ydata))
            p, = ax.plot(event.xdata, event.ydata, 'bo', ms=2)
            point_plots.append(p)
            redraw_preview()

    # ------------------------------------------------------------
    def handle_keypress(event):
        nonlocal shift_pressed, zoom_was_active

        if event.key == 'shift':
            shift_pressed = True
            if toolbar and toolbar.mode == 'zoom rect':
                zoom_was_active = True
                toolbar.zoom()  # turn zoom OFF temporarily
            else:
                zoom_was_active = False

    # ------------------------------------------------------------
    def handle_keyrelease(event):
        nonlocal shift_pressed, zoom_was_active

        if event.key == 'shift':
            shift_pressed = False
            if zoom_was_active and toolbar:
                toolbar.zoom()  # restore zoom
                zoom_was_active = False

        elif event.key == 'x':
            if len(clicked_points) < 3:
                print("Need at least 3 points to fit a circle")
                return

            try:
                x = np.array([p[0] for p in clicked_points])
                y = np.array([p[1] for p in clicked_points])
                cx, cy, r = fit_circle_to_points(x, y)

                idx = len(finished_circles) + 1
                print(
                    f"{file_name} | Circle #{idx}: "
                    f"center=({cx:.2f}, {cy:.2f}), radius={r:.2f}"
                )
                finished_circles.append((cx, cy))

                # Clear current points
                clicked_points.clear()
                for p in point_plots:
                    p.remove()
                point_plots.clear()

                final_circle = Circle((cx, cy), r, fill=False,
                                      color='lime', linewidth=2)
                ax.add_patch(final_circle)

                # --- RESET VIEW ---
                ax.set_xlim(0, image_data.shape[1])
                ax.set_ylim(0, image_data.shape[0])
                fig.canvas.draw_idle()

                # --- FORCE zoom ON (Windows/macOS safe) ---
                if toolbar and toolbar.mode != 'zoom rect':
                    toolbar.zoom()

                if len(finished_circles) >= max_num_circles:
                    print("All circles done. Close the window to continue.")
                    plt.close(fig)

            except Exception as e:
                print(f"Error fitting circle: {e}")

        elif event.key == 'backspace':
            if clicked_points:
                clicked_points.pop()
                if point_plots:
                    point_plots.pop().remove()
                redraw_preview()
            else:
                print("No points to undo")

    # ------------------------------------------------------------
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
        "filename_npy": current_file,"mirror":current_file.split("_")[0],
        "filename": current_file.replace("_qpsix_max.npy", ".datx").replace("_", "/", 1)
    }
    for fiducial_idx, (center_x, center_y) in enumerate(measured_fiducials, start=1):
        data_row[f"xc_{fiducial_idx}"] = center_x
        data_row[f"yc_{fiducial_idx}"] = center_y
    data_row['filename']= current_file.replace("_qpsix_max.npy", ".datx").replace("_", "/", 1)
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


