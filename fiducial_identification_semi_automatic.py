import os
import shutil
import numpy as np
import cv2
import matplotlib


matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import pandas as pd
from optics.zygo import ZygoData
from registration_2D import register

IGNORED_FOLDERS = {"shutter"}

# Diametre approximatif d'un fiduciaire (pixels) et demi-taille de la zone d'interet
# autour du clic (pixels). half_size=50 -> sous-image de 100 x 100 px.
FIDUCIAL_DIAMETER = 12
ROI_HALF_SIZE = 50

# register() renvoie le centre avec un biais de +0.5 px en x et en y : le template
# (largeur 2*diameter) est centré sur l'indice (largeur - 1) / 2, alors que register
# ajoute largeur / 2.
# Mettre 0.0 pour retrouver exactement la sortie brute de register().
REGISTER_CENTER_SHIFT = -0.5


# ================================================================
# ===================== CONVERSION STEP ==========================
# ================================================================

def convert_qpsix_to_npy(input_folder, output_folder):
    """
    Converts all .qpsix files in input_folder to .npy files in output_folder.
    Skips files that have already been converted.
    """
    os.makedirs(output_folder, exist_ok=True)
    any_processed = False

    for dirpath, dirnames, filenames in os.walk(input_folder):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_FOLDERS]

        for filename in filenames:
            if filename.lower().endswith(".qpsix"):
                filepath = os.path.join(dirpath, filename)

                rel_path = os.path.relpath(filepath, input_folder)
                safe_name = rel_path.replace(os.sep, "_").replace(".qpsix", "_qpsix_max.npy")
                output_path = os.path.join(output_folder, safe_name)

                if os.path.exists(output_path):
                    print(f"Already converted: {safe_name}")
                    continue

                print(f"Converting: {filepath}")
                try:
                    zdata = ZygoData(filepath)
                    img = np.std(zdata.data, axis=0)
                    np.save(output_path, img)
                    print(f"  -> Saved: {output_path}")
                    any_processed = True
                except Exception as e:
                    print(f"  Error for {filepath}: {e}")

    if not any_processed:
        print("No new files to convert.")


# ================================================================
# ======================= REGISTRATION ===========================
# ================================================================


def locate_fiducial(image_data, click_x, click_y, diameter, half_size):
    """
    Extracts a sub-image centered on the click, runs register() on it and
    converts the result back to full-image coordinates.

    Returns (xc, yc, (x0, y0, x1, y1)) where (x0, y0) is the position of the
    sub-image's top-left pixel in the full image.
    """
    h, w = image_data.shape
    x0 = max(int(round(click_x)) - half_size, 0)
    y0 = max(int(round(click_y)) - half_size, 0)
    x1 = min(int(round(click_x)) + half_size, w)
    y1 = min(int(round(click_y)) + half_size, h)

    sub = image_data[y0:y1, x0:x1]
    if min(sub.shape) <= 2 * diameter:
        raise ValueError("Zone trop petite (clic trop pres du bord ou half_size trop faible)")

    dx, dy = register(sub, diameter)
    return dx + x0 + REGISTER_CENTER_SHIFT, dy + y0 + REGISTER_CENTER_SHIFT, (x0, y0, x1, y1)


# ================================================================
# =================== FIDUCIAL MEASUREMENT =======================
# ================================================================

def interactive_fiducial_measurement(image_path, file_name, max_num_circles=3,
                                     percentile_range=(0, 100),
                                     diameter=FIDUCIAL_DIAMETER,
                                     half_size=ROI_HALF_SIZE):
    """
    Shows an image; the user clicks near a fiducial to define the region of interest,
    register() is run on that region and the result is shown.
    SHIFT + click = define region (re-click to move it)
    X             = validate the detected fiducial
    BACKSPACE     = cancel pending detection, or remove last validated fiducial
    """
    try:
        image_data = np.load(image_path)
    except Exception as e:
        print(f"Couldn't load image: {e}")
        return []

    fig, ax = plt.subplots(figsize=(15, 15))

    toolbar = plt.get_current_fig_manager().toolbar
    if toolbar and toolbar.mode != 'zoom rect':
        toolbar.zoom()

    vmin, vmax = np.percentile(image_data, percentile_range)
    ax.imshow(image_data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(
        f"Working on: {file_name}\n"
        "SHIFT + Click = define region around a fiducial | X = validate | Backspace = undo"
    )
    plt.axis("image")

    finished_circles = []  # (cx, cy) in full-image coordinates
    finished_artists = []  # list of lists of artists, one per validated fiducial
    pending = None  # (cx, cy)
    pending_artists = []

    shift_pressed = False
    zoom_was_active = False

    def clear_pending():
        nonlocal pending
        for a in pending_artists:
            a.remove()
        pending_artists.clear()
        pending = None

    def handle_mouse_click(event):
        nonlocal pending
        if event.inaxes != ax or event.button != 1 or not shift_pressed:
            return

        try:
            cx, cy, (x0, y0, x1, y1) = locate_fiducial(
                image_data, event.xdata, event.ydata, diameter, half_size
            )
        except Exception as e:
            print(f"Registration failed: {e}")
            return

        clear_pending()
        pending = (cx, cy)

        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                         fill=False, color='yellow', linestyle='--', linewidth=1)
        ax.add_patch(rect)
        circ = Circle((cx, cy), diameter / 2, fill=False, color='cyan',
                      linestyle='--', linewidth=2)
        ax.add_patch(circ)
        cross, = ax.plot(cx, cy, 'r+', ms=12)
        pending_artists.extend([rect, circ, cross])

        print(f"  Detected center = ({cx:.2f}, {cy:.2f})  | press X to validate")
        fig.canvas.draw_idle()

    def handle_keypress(event):
        nonlocal shift_pressed, zoom_was_active
        if event.key == 'shift':
            shift_pressed = True
            if toolbar and toolbar.mode == 'zoom rect':
                zoom_was_active = True
                toolbar.zoom()
            else:
                zoom_was_active = False

    def handle_keyrelease(event):
        nonlocal shift_pressed, zoom_was_active, pending

        if event.key == 'shift':
            shift_pressed = False
            if zoom_was_active and toolbar:
                toolbar.zoom()
                zoom_was_active = False

        elif event.key == 'x':
            if pending is None:
                print("Nothing to validate: SHIFT + click near a fiducial first")
                return

            cx, cy = pending
            idx = len(finished_circles) + 1
            print(f"{file_name} | Circle #{idx}: center=({cx:.2f}, {cy:.2f})")
            finished_circles.append((cx, cy))

            # keep the detection on screen as a validated (green) circle
            final_circle = Circle((cx, cy), diameter / 2, fill=False,
                                  color='lime', linewidth=2)
            ax.add_patch(final_circle)
            finished_artists.append([final_circle])

            clear_pending()

            ax.set_xlim(0, image_data.shape[1])
            ax.set_ylim(0, image_data.shape[0])
            fig.canvas.draw_idle()

            if toolbar and toolbar.mode != 'zoom rect':
                toolbar.zoom()

            if len(finished_circles) >= max_num_circles:
                print("All circles done. Close the window to continue.")
                plt.close(fig)

        elif event.key == 'backspace':
            if pending is not None:
                clear_pending()
                fig.canvas.draw_idle()
            elif finished_circles:
                finished_circles.pop()
                for a in finished_artists.pop():
                    a.remove()
                fig.canvas.draw_idle()
            else:
                print("Nothing to undo")

    fig.canvas.mpl_connect('button_press_event', handle_mouse_click)
    fig.canvas.mpl_connect('key_press_event', handle_keypress)
    fig.canvas.mpl_connect('key_release_event', handle_keyrelease)

    plt.show(block=True)
    return finished_circles


# ================================================================
# ===================== EXCEL / TEMPLATE =========================
# ================================================================

def fill_template_with_fiducials(fiducials_excel_path, template_dest):
    try:
        import xlwings as xw

        fid_df = pd.read_excel(fiducials_excel_path)
        data = fid_df.iloc[:, 1:].values  # skip first column

        app = xw.App(visible=False)
        app.display_alerts = False
        app.screen_updating = False

        try:
            wb = app.books.open(template_dest, update_links=False)

            if "fit" not in [s.name for s in wb.sheets]:
                print(f"  Sheet 'fit' not found. Available: {[s.name for s in wb.sheets]}")
                wb.close()
                return

            ws = wb.sheets["fit"]

            START_ROW = 34
            n_rows, n_cols = data.shape
            top_left = ws.range(f"A{START_ROW}")
            ws.range(top_left, top_left.offset(n_rows - 1, n_cols - 1)).value = data.tolist()

            wb.save()
            wb.close()
            print(f"Template filled with {len(data)} rows starting at row {START_ROW}.")
        finally:
            app.quit()

    except Exception as e:
        print(f"Could not fill template: {e}")


def save_results_to_excel(measurement_results, excel_output_path):
    """
    filename_npy | mirror | xc_1 | yc_1 | ... | xc_N | yc_N | roll | binning | filename
    """
    results_df = pd.DataFrame(measurement_results)

    yc_cols = [c for c in results_df.columns if c.startswith("yc_")]
    if yc_cols:
        insert_pos = results_df.columns.get_loc(yc_cols[-1]) + 1
    else:
        insert_pos = len(results_df.columns)

    if "roll" not in results_df.columns:
        results_df.insert(insert_pos, "roll", 0)
        insert_pos += 1

    if "binning" not in results_df.columns:
        results_df.insert(insert_pos, "binning", 1)

    if "filename" in results_df.columns:
        cols = [c for c in results_df.columns if c != "filename"] + ["filename"]
        results_df = results_df[cols]

    results_df.to_excel(excel_output_path, index=False)


# ================================================================
# ========================= MAIN SCRIPT ==========================
# ================================================================

if __name__ == "__main__":
    print("=== Fiducials Fitting Tool ===")

    folder_path = input("Enter folder path containing datx/qpsix files: ").strip().strip("'\"")
    npy_folder = os.path.join(folder_path, "npy_exports")

    TEMPLATE_PATH = input("Enter template path (substrates_template_FM_form_casquette.xlsm): ").strip().strip("'\"")
    template_dest = os.path.join(folder_path, os.path.basename(TEMPLATE_PATH))

    if os.path.exists(template_dest):
        print(f"Template already present: {os.path.basename(TEMPLATE_PATH)}")
    else:
        try:
            shutil.copy2(TEMPLATE_PATH, template_dest)
            print(f"Template copied to: {template_dest}")
        except Exception as e:
            print(f"Could not copy template: {e}")

    image_type = input("Image type (tilt / sfe): ").strip().lower()
    percentile_range = (5, 95) if image_type == "sfe" else (0, 100)
    print(f"Using percentile range: {percentile_range[0]} - {percentile_range[1]}")

    print("\n=== Checking / Running conversion step ===")
    convert_qpsix_to_npy(folder_path, npy_folder)

    circles_per_image = 3
    excel_output_path = os.path.join(npy_folder, "fiducials_measurements.xlsx")

    if os.path.isfile(excel_output_path):
        previous_results = pd.read_excel(excel_output_path)
        processed_files = set(previous_results["filename_npy"].tolist())
        measurement_results = previous_results.to_dict(orient="records")
        print(f"Loaded existing Excel with {len(processed_files)} measured files.")
    else:
        processed_files = set()
        measurement_results = []
        print("No existing Excel found, starting a new one.")

    npy_file_list = [f for f in os.listdir(npy_folder) if f.lower().endswith("qpsix_max.npy")]
    if not npy_file_list:
        print("No NPY files found in the npy_exports folder.")
        raise SystemExit

    print(f"Found {len(npy_file_list)} NPY files ({len(processed_files)} already measured).")
    print("\nHow to use:")
    print("  * SHIFT + Click near a fiducial = define the region, the center is detected automatically")
    print(f"  * X = validate the detected fiducial (need {circles_per_image} total)")
    print("  * Backspace = cancel the pending detection / remove the last validated one\n")

    for current_file in npy_file_list:
        if current_file in processed_files:
            print(f"  Skipping already processed: {current_file}")
            continue

        print(f"\nProcessing: {current_file}")
        full_image_path = os.path.join(npy_folder, current_file)

        measured_fiducials = interactive_fiducial_measurement(
            full_image_path, current_file, circles_per_image, percentile_range=percentile_range
        )

        data_row = {
            "filename_npy": current_file,
            "mirror": current_file.split("_")[0],
        }
        for i, (cx, cy) in enumerate(measured_fiducials, start=1):
            data_row[f"xc_{i}"] = cx
            data_row[f"yc_{i}"] = cy
        data_row["roll"] = 0
        data_row["binning"] = 1
        data_row["filename"] = current_file.replace("_qpsix_max.npy", ".datx").replace("_", "/", 1)

        measurement_results.append(data_row)
        processed_files.add(current_file)
        save_results_to_excel(measurement_results, excel_output_path)

    print("\nAll files processed!")
    fill_template_with_fiducials(excel_output_path, template_dest)
