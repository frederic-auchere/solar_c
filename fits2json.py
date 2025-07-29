import numpy as np
import cv2
import os
from astropy.io import fits
from tqdm import tqdm
import json
import argparse


def read_header(file):
    with open(file) as f:
        while True:
            if 'Timestamps' in f.readline():
                break

        lines = f.readlines()
        dates = []
        for line in lines:
            _, _, date, time = line.split()
            dates.append(f'20{date[4:6]}-{date[2:4]}-{date[0:2]}T{time[0:2]}:{time[2:4]}:{float(time[4:]):.3f}')

    return dates


def register(reference, image, edge=512):
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

    cc = cv2.matchTemplate(reference,
                           np.copy(image[edge:-edge, edge:-edge]),
                           cv2.TM_CCOEFF_NORMED)
    dx, dy = parabolic_interpolation(cc)

    dx -= edge
    dy -= edge

    return dx, dy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="FITS2JSON",
                                     description="Converts FITS cube to JSON stability data")
    parser.add_argument("filename",
                        help="Input FITS file",
                        default="",
                        type=str)

    parser.add_argument("-p", "--plate_scale",
                        help="Detector plate scale (arcsecond/pixel)",
                        default=np.degrees(0.0024 / 2540) * 3600,
                        type=float)

    parser.add_argument("-e", "--edge",
                        help="Cross-correlation edge",
                        default=40,
                        type=int)

    args = parser.parse_args()

    full_file_path = args.filename
    header_file = os.path.splitext(full_file_path)[0] + '.txt'
    full_header_path = os.path.join(os.path.dirname(full_file_path), header_file)
    dates = read_header(full_header_path)

    data = fits.getdata(full_file_path)

    reference = data[0]
    x , y = [], []
    for d in tqdm(data[1:]):
        dx, dy = register(reference, d, edge=args.edge)
        x.append(dx)
        y.append(dy)

    data = {
        'unit': 'arcsec',
        'data': {d: {'x': x * args.plate_scale, 'y': y * args.plate_scale} for d, x, y in zip(dates[1:], x, y)}
    }
    with open(os.path.join(os.path.splitext(full_file_path)[0] + '.json'), 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=4)
