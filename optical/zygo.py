import copy
from optics.zygo import Fit, SagData
from optics import surfaces
from optics.geometry import Point, Polygon
from optical.fiducials import ega_from_fiducials
from optical.surfaces import EGASubstrate
from optical import mirror_crown_angles
import os
from openpyxl import load_workbook
import matplotlib as mp
from pathlib import Path
import numpy as np


class EGAFit(Fit):

    @classmethod
    def from_xlsx(cls, xlsx_file):

        def read_table():
            fields = next(rows)
            table_keys = []
            column = 0
            while fields[column].value is not None:
                table_keys.append(fields[column].value.split(' ')[0].lower())
                column += 1
            table = []
            while True:
                row = next(rows)
                if row[0].value is None:
                    break
                table.append({key:cell.value for key, cell in zip(table_keys, row)})

            return table

        wb = load_workbook(xlsx_file, keep_vba=True)
        rows = wb.active.rows
        while True:
            try:
                row = next(rows)
            except StopIteration:
                break
            if row[0].value is None:
                continue
            if "1. Options" in row[0].value:
                options = read_table()[0]
                tolerance = options.pop('tolerance')
                floating_reference = options.pop('reference') == 'floating'
                parameters = options.pop('parameters')
                fitted_parameters = [] if parameters is None else [p.strip() for p in parameters.split(',')]
            elif "2.1 Surface" in row[0].value:
                substrate_surface = read_table()[0]
                surface_type = substrate_surface.pop('type')
                substrate_surface = surfaces.ParametricSurface.factory.create(surface_type, degrees=True, **substrate_surface)
            elif "2.2 Aperture" in row[0].value:
                substrate_aperture = read_table()[0]
                shape = substrate_aperture.pop('shape')
                substrate_aperture = surfaces.Aperture.factory.create(shape + 'Aperture', **substrate_aperture)
            elif "2.3 Useful area" in row[0].value:
                substrate_useful_area = read_table()[0]
                shape = substrate_useful_area.pop('shape')
                substrate_useful_area = surfaces.Aperture.factory.create(shape + 'Aperture', **substrate_useful_area)
            elif "2.4 Fiducials" in row[0].value:
                xy = read_table()[0]
                substrate = EGASubstrate(substrate_surface, substrate_aperture, substrate_useful_area,
                                         fiducials=((xy['x1'], xy['y1']), (xy['x2'], xy['y2']), (xy['x3'], xy['y3'])))
            elif "3. Reference" in row[0].value:
                reference = read_table()[0]
                surface_type = reference.pop('type')
                reference = surfaces.ParametricSurface.factory.create(surface_type, **reference)
            elif "4.1 Path" in row[0].value:
                row = next(rows)
                path = os.path.dirname(xlsx_file) if row[0].value is None else row[0].value
            elif "4.2 Data" in row[0].value:
                table = read_table()
                fiducials = []
                crown_indices = []
                for row in table:
                    crown_index = row.pop('crown')
                    crown_indices.append(crown_index if crown_index > 0 else None)
                    fiducials.append(Polygon((Point(row.pop('x1'), row.pop('y1'), 0),
                                              Point(row.pop('x2'), row.pop('y2'), 0),
                                              Point(row.pop('x3'), row.pop('y3'), 0))))
                mirror_crown = [mirror_crown_angles[c - 1] for c in crown_indices] if all(crown_indices) else None

                to_normal = type(reference) is not surfaces.Flat
                geometries = ega_from_fiducials(fiducials, substrate, to_normal, offset_angles=mirror_crown)
                for geometry in geometries:
                    print(geometry)

                sag_data = []
                for row, geometry in zip(table, geometries):
                    # file = os.path.join(path, row.pop('file'))
                    filename = row.pop('file').replace("\\", os.sep).replace("/", os.sep)
                    file = (Path(path) / filename).resolve()
                    row.update(geometry)

                    roll = row.pop('roll')
                    for key in ['gx_std', 'gy_std', 'dx_std', 'dy_std', 'theta_std']:
                        row.pop(key)  # Unused by sag_data

                    binning = row.pop('binning')
                    sd = SagData(file, auto_crop=False, binning=1, **row)

                    cos = np.cos(np.radians(roll))
                    sin = np.sin(np.radians(roll))
                    rotation = np.array([[cos, sin, 0],
                                         [-sin, cos, 0],
                                         [0, 0, 1]])
                    if roll == 270:
                        pivot_x = substrate.aperture.dx
                        pivot_y = substrate.aperture.dy - (substrate.aperture.y_width - substrate.aperture.x_width) / 2
                    elif roll == 90:
                        pivot_x = substrate.aperture.dx - (substrate.aperture.y_width - substrate.aperture.x_width) / 2
                        pivot_y = substrate.aperture.dy
                    elif roll == 0 or roll == 180:
                        pivot_x = substrate.aperture.dx
                        pivot_y = substrate.aperture.dy  # substrate center
                    else:
                        raise ValueError('Invalid tilt mount angle')
                    # pivot_x += -0.13
                    # pivot_y -= 0.04
                    to_pivot = [[1, 0, -pivot_x],
                                 [0, 1, -pivot_y],
                                 [0, 0, 1]]
                    to_ega = [[1, 0, pivot_x],
                              [0, 1, pivot_y],
                              [0, 0, 1]]
                    x, y, _ = to_ega @ rotation @ to_pivot @ (0, 0, 1)
                    row['dx'], row['dy'] = sd.to_data(x, y)
                    row['theta']  = (row['theta'] - roll) % 360
                    sag_data.append(SagData(file, auto_crop=False, binning=binning, **row))

        return cls(sag_data,
                   copy.deepcopy(substrate),
                   fitted_parameters,
                   reference,
                   floating_reference=floating_reference, tol=tolerance, **options)


def sfe_measure(**kwargs):
    extension = os.path.splitext(kwargs['file'])[1]
    if extension == '.xlsx':
        fitter = EGAFit.from_xlsx(kwargs['file'])
    elif extension == '.json':
        fitter = EGAFit.from_json(kwargs['file'])
    else:
        raise ValueError('File extension not supported')

    mp.use('Agg')
    fitter.fit()
    fitter.make_report(kwargs['output'])
