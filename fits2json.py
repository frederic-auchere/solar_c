import numpy as np
import cv2
import os
from astropy.io import fits
from tqdm import tqdm
import json
import argparse
import csv


def read_header(file):
    with open(file, encoding="utf-8", errors="ignore") as f:
        while True:
            if 'Timestamps' in f.readline():
                break

        lines = f.readlines()
        dates = []
        for line in lines:
            _, _, date, time = line.split()
            dates.append(f'20{date[4:6]}-{date[2:4]}-{date[0:2]}T{time[0:2]}:{time[2:4]}:{float(time[4:]):.3f}')

    return dates


def register(reference, image, edge=(10, 10)):
    """
    Returns the offset between image and reference computed by cross correlation
    """
    def parabolic_interpolation(cc):
        """
        Returns the position of the extremum of the cross_correlation matrix cc
        assuming a parabolic shape
        """
        cy, cx = np.unravel_index(np.argmax(cc, axis=None), cc.shape)
        if cx == 0 or cy == 0 or cx == cc.shape[1]-1 or cy == cc.shape[0]-1:
            return cx, cy
        else:
            xi = [cx-1, cx, cx+1]
            yi = [cy-1, cy, cy+1]
            ccx2 = cc[[cy, cy, cy], xi]**2
            ccy2 = cc[yi, [cx, cx, cx]]**2
    
            xn = ccx2[2] - ccx2[1]
            xd = ccx2[0] - 2*ccx2[1] + ccx2[2]
            yn = ccy2[2] - ccy2[1]
            yd = ccy2[0] - 2*ccy2[1] + ccy2[2]
    
            dx = cx if xd == 0 else xi[2] - xn/xd - 0.5
            dy = cy if yd == 0 else yi[2] - yn/yd - 0.5
    
            return dx, dy    

    edge_x = int(round(image.shape[1] * edge[0] / 100.0))
    edge_y = int(round(image.shape[0] * edge[1] / 100.0))

    cc = cv2.matchTemplate(reference,
                           np.copy(image[edge_y:-edge_y, edge_x:-edge_x]),
                           cv2.TM_CCOEFF_NORMED)
    dx, dy = parabolic_interpolation(cc)

    dx -= edge_x
    dy -= edge_y

    return dx, dy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="FITS2JSON",
                                     description="Converts FITS cube to JSON or CSV stability data")
    parser.add_argument("filename",
                        help="Input FITS file",
                        default="",
                        type=str)

    parser.add_argument("-p", "--plate_scale",
                        help="Detector plate scale (arcsecond/pixel)",
                        default=np.degrees(0.0024 / 2895) * 3600,
                        type=float)

    parser.add_argument("-e", "--edge",
                        help="Cross-correlation edge in % of the shape",
                        default=10,
                        type=int)

    parser.add_argument("-t", "--type",
                        help="Output file type",
                        default='json',
                        type=str)

    args = parser.parse_args()

    if args.type not in ['json', 'csv']:
        raise TypeError('Type must be json or csv')

    full_file_path = args.filename
    header_file = os.path.splitext(full_file_path)[0] + '.txt'
    full_header_path = os.path.join(os.path.dirname(full_file_path), header_file)
    dates = read_header(full_header_path)

    data = fits.getdata(full_file_path).astype(np.float32)

    reference = data[0]
    x , y = [], []
    for d in tqdm(data[1:]):
        dx, dy = register(reference, d, edge=(args.edge, args.edge))
        x.append(dx * args.plate_scale)
        y.append(dy * args.plate_scale)
    x = np.array(x)
    y = np.array(y)
    r = np.sqrt(x ** 2 + y ** 2)

    with open(os.path.join(os.path.splitext(full_file_path)[0] + '.' + args.type), 'w', newline='') as fp:
        if args.type == 'json':
            data = {
                'unit': 'arcsec',
                'data': {d: {'x': x, 'y': y} for d, x, y in zip(dates[1:], x, y)}
            }
            json.dump(data, fp, indent=4)
        elif args.type == 'csv':
            fieldnames = ['date', 'x', 'y', 'r']
            writer = csv.DictWriter(fp, delimiter=',', quoting=csv.QUOTE_NONE, fieldnames=fieldnames, dialect='excel')
            writer.writeheader()
            writer.writerows([{'date': d, 'x': x, 'y': y, 'r':r} for d, x, y, r in zip(dates[1:], x, y, r)])

    x_std = np.std(x)
    y_std = np.std(y)
    r_std = np.std(r)
    print(f'Standard deviation: x:{x_std:.3f}" RMS y:{x_std:.3f}" RMS r:{r_std:.3f}" RMS')
