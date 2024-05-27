import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from surface import create_surface_from_NX_file
import pandas as pd

def nx_to_zernike(nx_file, grid_points = 300, mesh_points = 100, max_degree = 9):
    path, base_name = os.path.split(nx_file)
    zernike_file = os.path.join(path, 'zernike_coefficients_' + os.path.splitext(base_name)[0] + '.dat')
    optic_surface = create_surface_from_NX_file(nx_file, grid_points)
    optic_surface.calculate_zernike_fit('Noll', max_degree, mesh_points, 'linear')
    c_nm = optic_surface.get_zernike_fit().get_c_nm()
    n_tot = int(((max_degree + 1) * (max_degree + 2)) / 2)

    radius = optic_surface.get_surface_radius()
    data = np.vstack(('{}'.format(n_tot), radius, c_nm))
    zernike_coeffs = pd.DataFrame(data)
    zernike_coeffs.to_csv(zernike_file, header=False, index=False)

def read_distortion_file(distortion_file):

    distortion = []
    waves = []
    with open(distortion_file, encoding='utf-16-le') as f:
        for line in f:
            if line[0:11] == ';Wavelength':
                waves.append(float(line.split('=')[1][:-3]))
                distortion.append([])
            elif line[0] != ';' and len(line) > 3:
                distortion[-1].append([float(value) for value in line.split()])

    return np.array(distortion), waves

def print_macro_lines(files):
    for i, zernike_file in enumerate(files):
        print(f'callsetstr {i}, "{os.path.basename(zernike_file)}"')

def case_from_file(file):
    if 'Cas' in file:
        return file[file.index('Cas'):].split('.')[0]

def get_file_suffix(file):
    return file.split('_')[-1].split('.')[0]

def get_null_file(file):
    null_files = glob.glob(os.path.join(input_path, '*null*.txt'))
    null_suffix = [get_file_suffix(f) for f in null_files]
    suffix = get_file_suffix(file)
    return null_files[null_suffix.index(suffix)]


# input_path = r'Y:\02- Engineering\04- Thermique\1_EtudeGLQ\DefortmationThermoElastique'
# files = glob.glob(os.path.join(input_path, 'Grating*.xlsx'))
#
# for nx_file in files:
#     nx_to_zernike(nx_file)

# files = glob.glob(os.path.join(input_path, 'zernike*.dat'))
# print_marco_lines(files)


plt.close('all')
plt.ioff()

input_path = r'C:\\'
files = glob.glob(os.path.join(input_path, '*Cas*.txt'))
unique_cases = set([case_from_file(f) for f in files])

detectors = ['SW', 'LW1', 'LW2', 'LW3']
names = ['RMS variation [arcseconds]',  #'RMS radius variation [% / °]',
         'Geo variation [arcseconds]',  #'Geometric radius variation [% / °]',
         'Centroid deviation\n[$\\bigcirc$ = 1 $\mu$m]']
parameters = [2, 3, 4, 5]
y_lims = [(-0.05, 0.05), (-0.3, 0.3), (-1, 1)]

for case in unique_cases:
    case_files = [f for f in files if case == case_from_file(f)]
    suffixes = [get_file_suffix(f) for f in case_files]

    fig, axes = plt.subplots(len(names), len(detectors), figsize=(17, 6),
                             sharex=True, sharey=False, gridspec_kw={'height_ratios':[1, 1, 1]})

    for d, detector in enumerate(detectors):
        detector_file = case_files[suffixes.index(detector)]
        null_file = get_null_file(detector_file)

        null, _ = read_distortion_file(null_file)
        distortion, waves = read_distortion_file(detector_file)

        delta = distortion - null

#        delta[:, :, 2:4] /= null[:, :, 2:4]
        delta[:, :, 2:4] = np.degrees(1e-3 * delta[:, :, 2:4] / 20078) * 3600  # convert to arcseconds based on focal length
        delta[:, :, 4:] *= 1000  # convert delta centoidx, centroidy, chiefx, chiefy to microns

        colors = np.linspace(0, 1, len(delta))
        fields = np.linspace(1, len(delta[0, :, 0]), len(delta[0, :, 0]), dtype=int)

        for p, name, ax, y_lim in zip(parameters[0:2], names[0:2], axes[0:2, d].flatten(), y_lims[0:2]):
            for w, c, wave in zip(delta, colors, waves):
                ax.plot(fields, w[:, p], 'o-', color=plt.cm.RdYlBu(c), label=f'{wave} nm')
            ax.set_xticks(fields)
            ax.set_xticklabels('')
            ax.set_ylim(y_lim)
            ax.grid(axis='y')
            if d == 0:
                ax.set_ylabel(name)
            else:
                ax.set_yticklabels('')

            ax.legend(ncol=3, bbox_to_anchor=(1, -0.05))

        arrow_scale = 3
        origin = np.meshgrid(fields, np.zeros_like(fields))
        x, _ = np.indices(delta[:, :, 0].shape)
        x = x / x.max()
        colors = plt.cm.RdYlBu(x.flatten())
        for f in fields:
            circle = Circle((f, 0), 1 / arrow_scale, color='grey', fill=False)
            axes[2, d].add_patch(circle)
        axes[2, d].quiver(*origin, delta[:, :, parameters[-2]], delta[:, :, parameters[-1]],
                          color=colors, angles='xy', scale_units='xy', scale=arrow_scale)
        axes[2, d].axis('scaled')
        axes[2, d].set_xlabel('Field #')
        axes[2, d].set_xticks(fields)
        axes[2, d].set_xticklabels(fields)
        axes[2, d].set_yticks([])
        axes[2, d].set_yticklabels('')
        axes[2, d].set_ylim(y_lims[2])
        if d == 0:
            axes[2, d].set_ylabel(names[2])
        axes[0, d].set_title(detector)

    plt.tight_layout()

    fig.savefig(os.path.join('output', f'zernike_{case}_arcseconds.png'), dpi=200)

plt.ion()