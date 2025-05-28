import copy

from optics.zygo import Fit, SagData
from optics import surfaces
from optics.geometry import Point, Polygon

from solar_c.optical.fiducials import ega_from_fiducials
from solar_c.optical.surfaces import EGASubstrate
from solar_c.optical import mirror_crown_angles
import os
from openpyxl import load_workbook
import matplotlib as mp


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

        wb = load_workbook(xlsx_file)
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
                substrate_surface = surfaces.ParametricSurface.factory.create(surface_type, **substrate_surface)
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
                path = '' if row[0].value is None else row[0].value
            elif "4.2 Data" in row[0].value:
                sag_data = []
                table = read_table()
                fiducials = []
                mirror_crown= []
                for row in table:
                    mirror_crown.append(mirror_crown_angles[row.pop('crown') - 1])
                    fiducials.append(
                        Polygon((Point(row.pop('x1'), row.pop('y1'), 0),
                                 Point(row.pop('x2'), row.pop('y2'), 0),
                                 Point(row.pop('x3'), row.pop('y3'), 0)))
                    )
                geometries = ega_from_fiducials(fiducials, substrate, offset_angles=mirror_crown)
                for row, geometry in zip(table, geometries):
                    file = os.path.join(path, row.pop('file'))
                    row.update(geometry)
                    sag_data.append(SagData(file, **row))

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
