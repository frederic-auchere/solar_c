import numpy as np
import datetime
import math
import ctypes
import struct

ZYGO_INVALID_PHASE = 2147483640
ZYGO_ENC = 'utf-8'  # may be ASCII, cp1252...
ZYGO_PHASE_RES_FACTORS = {
    0: 4096,    # 12-bit
    1: 32768,   # 15-bit
    2: 131072,  # 17-bit
}
ZYGO_DEFAULT_WVL = 6.327999813038332e-07

def parabolic_interpolation(cc, extremum=np.nanargmax):
    """
    Returns the position of the extremum of matrix cc
    assuming a parabolic shape
    """
    cy, cx = np.unravel_index(extremum(cc, axis=None), cc.shape)
    if cx == 0 or cy == 0 or cx == cc.shape[1] - 1 or cy == cc.shape[0] - 1:
        return cx, cy
    else:
        xi = [cx - 1, cx, cx + 1]
        yi = [cy - 1, cy, cy + 1]
        ccx2 = cc[[cy, cy, cy], xi] ** 2
        ccy2 = cc[yi, [cx, cx, cx]] ** 2

        xn = ccx2[2] - ccx2[1]
        xd = ccx2[0] - 2 * ccx2[1] + ccx2[2]
        yn = ccy2[2] - ccy2[1]
        yd = ccy2[0] - 2 * ccy2[1] + ccy2[2]

        dx = cx if xd == 0 else xi[2] - xn / xd - 0.5
        dy = cy if yd == 0 else yi[2] - yn / yd - 0.5

        return dx, dy


def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)



def fit_circle_to_points(x_coords, y_coords):
    """
    Fits a circle to 3 or more points using least squares

    """

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    if len(x_coords) < 3:
        raise ValueError("Need at least 3 points to fit a circle")

    # Setting up the linear system
    A = np.c_[2 * x_coords, 2 * y_coords, np.ones(x_coords.size)]
    b = x_coords ** 2 + y_coords ** 2
    coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)  # keeping all outputs just in case

    center_x, center_y = coeffs[0], coeffs[1]  # circle center coordinates
    radius = np.sqrt(coeffs[2] + center_x ** 2 + center_y ** 2)

    return center_x, center_y, radius





def _zygo_metadata_helper():
    """Returns a dict of [name] -> [struct code, low index, high index, default]."""
    IB16 = '>H'
    IL16 = '<H'
    IB32 = '>I'
    IL32 = '<I'
    FB32 = '>f'
    FL32 = '<f'
    LE = '<'
    uint8 = 'B'
    return {
        'magic_number': (IB32, 0, 4, 0x881B036F),
        'header_format': (IB16, 4, 6, 1),
        'header_size': (IB32, 6, 10, 834),
        'swtype': (IB16, 10, 12, 1),
        'swdate': (LE+'30'+'s', 12, 42, ' '*30),  # 30 blank spaces
        'swmaj': (IB16, 42, 44, 0),
        'swmin': (IB16, 44, 46, 0),
        'swpatch': (IB16, 46, 48, 0),
        'ac_x': (IB16, 48, 50, 0),
        'ac_y': (IB16, 50, 52, 0),
        'ac_width': (IB16, 52, 54, 0),
        'ac_height': (IB16, 54, 56, 0),
        'ac_n_buckets': (IB16, 56, 58, 0),
        'ac_range': (IB16, 58, 60, 0),
        'ac_n_bytes': (IB32, 60, 64, 0),
        'cn_x': (IB16, 64, 66, 0),
        'cn_y': (IB16, 66, 68, 0),
        'cn_width': (IB16, 68, 70, 0),
        'cn_height': (IB16, 70, 72, 0),
        'cn_n_bytes': (IB32, 72, 76, 0),
        'timestamp': (IB32, 76, 80, 0),  # TODO: use unix now here, unless it would overflow
        'comment': (LE+'82'+'s', 80, 162, ' '*82),
        'source': (IB16, 162, 164, 0),
        'scale_factor': (FB32, 164, 168, 0.5),
        'wavelength': (FB32, 168, 172, ZYGO_DEFAULT_WVL),
        'numerical_aperture': (FB32, 172, 176, 0),
        'obliquity_factor': (FB32, 176, 180, 1.),
        'magnification': (FB32, 180, 184, 0),
        'lateral_resolution': (FB32, 184, 188, 1.),
        'acq_type': (IB16, 188, 190, 0),
        'intensity_average_count': (IB16, 190, 192, 0),
        'sfac_limit': (IB16, 194, 196, 3),
        'ramp_cal': (IB16, 192, 194, 0),
        'ramp_gain': (IB16, 196, 198, 1753),
        'part_thickness': (FB32, 198, 202, 0),
        'sw_llc': (IB16, 202, 204, 1),
        'target_range': (FB32, 204, 208, 0.1),
        'rad_crv_measure_seq': (IL16, 208, 210, 0),
        'min_mod': (IB32, 210, 214, 17),
        'min_mod_count': (IB32, 214, 218, 50),
        'phase_res': (IB16, 218, 220, 1),
        'min_area': (IB32, 220, 224, 20),
        'discontinuity_action': (IB16, 224, 226, 1),
        'discontinuity_filter': (FB32, 226, 230, 60.),
        'connect_order': (IB16, 230, 232, 0),
        'sign': (IB16, 232, 234, 0),
        'camera_width': (IB16, 234, 236, 0),  # TODO: will Mx be happy?
        'camera_height': (IB16, 236, 238, 0),
        'sys_type': (IB16, 238, 240, 23),
        'sys_board': (IB16, 240, 242, 0),
        'sys_serial': (IB16, 242, 244, 0),
        'sys_inst_id': (IB16, 244, 246, 0),
        'obj_name': (LE+'12'+'s', 246, 258, ' '*12),
        'part_name': (LE+'40'+'s', 258, 298, ' '*80),
        'codev_type': (IB16, 298, 300, 0),
        'phase_avg_count': (IB16, 300, 302, 1),
        'sub_sys_err': (IB16, 302, 304, 0),
        '__pad0': ('16x', 304, 320, '\x00'*16),
        'part_sn': (LE+'40'+'s', 320, 360, ' '*40),
        'refractive_index': (FB32, 360, 364, 1.),
        'remove_tilt': (IB16, 364, 366, 0),
        'remove_fringes': (IB16, 366, 368, 0),
        'max_area': (IB32, 368, 372, 9999999),
        'setup_type': (IB16, 372, 374, 0),
        'wrapped': (IB16, 374, 376, 0),
        'pre_connect_filter': (FB32, 376, 380, 0.),
        '__pad1': ('6x', 380, 386, '\x00'*6),
        'wavelength_in_1': (FB32, 386, 390, ZYGO_DEFAULT_WVL),
        'wavelength_in_2': (FB32, 390, 394, ZYGO_DEFAULT_WVL),
        'wavelength_in_3': (FB32, 394, 398, ZYGO_DEFAULT_WVL),
        'wavelength_select': ('<8s', 398, 406, '1       '),
        'fda_res': (IB16, 406, 408, 0),
        'scan_description': (LE+'20'+'s', 408, 428, ' '*20),
        'n_fiducials': (IB16, 428, 430, 0),
        'fiducial_1':  (FB32, 430, 434, 0.),
        'fiducial_2':  (FB32, 434, 438, 0.),
        'fiducial_3':  (FB32, 438, 442, 0.),
        'fiducial_4':  (FB32, 442, 446, 0.),
        'fiducial_5':  (FB32, 446, 450, 0.),
        'fiducial_6':  (FB32, 450, 454, 0.),
        'fiducial_7':  (FB32, 454, 458, 0.),
        'fiducial_8':  (FB32, 458, 462, 0.),
        'fiducial_9':  (FB32, 462, 466, 0.),
        'fiducial_10':  (FB32, 466, 470, 0.),
        'fiducial_11': (FB32, 470, 474, 0.),
        'fiducial_12': (FB32, 474, 478, 0.),
        'fiducial_13': (FB32, 478, 482, 0.),
        'fiducial_14': (FB32, 482, 486, 0.),
        'pixel_width': (FB32, 486, 490, 7.4e-6),
        'pixel_height': (FB32, 490, 494, 7.4e-6),
        'exit_pupil_diameter': (FB32, 494, 498, 0.),
        'light_level_percent': (FB32, 498, 502, 55.),
        'coords_state': (IL32, 502, 506, 0),
        'coords_x': (FL32, 506, 510, 0.),
        'coords_y': (FL32, 510, 514, 0.),
        'coords_z': (FL32, 514, 518, 0.),
        'coords_a': (FL32, 518, 522, 0.),
        'coords_b': (FL32, 522, 526, 0.),
        'coords_c': (FL32, 526, 530, 0.),
        'cohrence_mode': (IL16, 530, 532, 0),
        'surface_filter': (IL16, 532, 534, 0),
        'sys_err_filename': (LE+'28'+'s', 534, 562, ' '*28),
        'zoom_descr': (LE+'8'+'s', 562, 570, '   1X '),
        'alpha_part': (FL32, 570, 574, 0),
        'beta_part': (FL32, 574, 578, 0),
        'dist_part': (FL32, 578, 582, 0),
        'cam_split_loc_x': (IL16, 582, 584, 0),
        'cam_split_loc_y': (IL16, 584, 586, 0),
        'cam_split_trans_x': (IL16, 586, 588, 0),
        'cam_split_trans_y': (IL16, 588, 590, 0),
        'material_a': (LE+'24'+'s', 590, 614, ' '*24),
        'material_b': (LE+'24'+'s', 614, 638, ' '*24),
        '__pad2': ('4x', 638, 642, '\x00'*4),
        'dmi_center_x': (FL32, 642, 646, 0.),
        'dmi_center_y': (FL32, 646, 650, 0.),
        'sph_distortion_correction': (IL16, 650, 652, 0),
        'sph_dist_part_na': (FL32, 654, 658, 0.),
        'sph_dist_part_radius': (FL32, 658, 662, 0.),
        'sph_dist_cal_na': (FL32, 662, 666, 0.),
        'sph_dist_cal_radius': (FL32, 666, 670, 0.),
        'surface_type': (IL16, 670, 672, 0),
        'ac_surface_type': (IL16, 672, 674, 0),
        'z_pos': (FL32, 674, 678, 0.),
        'power_mul': (FL32, 678, 682, 0.),
        'focus_mul': (FL32, 682, 686, 0.),
        'roc_focus_cal_factor': (FL32, 686, 690, 0.),
        'roc_power_cal_factor': (FL32, 690, 694, 0.),
        'ftp_pos_left': (FL32, 694, 698, 0.),
        'ftp_pos_right': (FL32, 698, 702, 0.),
        'ftp_pos_pitch': (FL32, 702, 706, 0.),
        'ftp_pos_roll': (FL32, 706, 710, 0.),
        'min_mod_percent': (FL32, 710, 714, 7.),
        'max_intens': (IL32, 714, 718, 0),
        'ring_of_fire': (IL16, 718, 720, 0),
        '__pad3': ('x', 720, 721, '\x00'),
        'rc_orientation': ('c', 721, 722, ' '),
        'rc_distance': (FL32, 722, 726, 0.),
        'rc_angle': (FL32, 726, 730, 0.),
        'rc_diameter': (FL32, 730, 734, 0.),
        'rem_fringes_mode': (IB16, 734, 736, 0),
        '__pad4': ('x', 736, 737, '\x00'),
        'ftpsi_phase_res': (uint8, 737, 738, 0),
        'frames_acquired': (IL16, 738, 740, 0),
        'cavity_type': (IL16, 740, 742, 0),
        'cam_frame_rate': (FL32, 742, 746, 0.),
        'tune_range': (FL32, 746, 750, 0.),
        'cal_pix_x': (IL16, 750, 752, 0),
        'cal_pix_y': (IL16, 752, 754, 0),
        'test_cal_pts_1': (FL32, 758, 762, 0.),
        'test_cal_pts_2': (FL32, 762, 766, 0.),
        'test_cal_pts_3': (FL32, 766, 770, 0.),
        'test_cal_pts_4': (FL32, 770, 774, 0.),
        'ref_cal_pts_1': (FL32, 774, 778, 0.),
        'ref_cal_pts_2': (FL32, 778, 782, 0.),
        'ref_cal_pts_3': (FL32, 782, 786, 0.),
        'ref_cal_pts_4': (FL32, 786, 790, 0.),
        'test_cal_pix_opd': (FL32, 790, 794, 0.),
        'test_ref_pix_opd': (FL32, 794, 798, 0.),
        'flash_phase_cd_mask': (FL32, 798, 802, 9.139576869988608e-40),
        'flash_phase_alias_mask': (FL32, 802, 806, 0.),
        'flash_phase_filter': (FL32, 806, 810, 0.),
        'scan_direction': (uint8, 810, 811, 0),
        'ftpsi_res_factor': (IL16, 814, 816, 0),
        }


def write_zygo_dat(file, phase, dx, wavelength=0.6328, intensity=None):
    """Write a Zygo .DAT interferogram file.

    Parameters
    ----------
    file : path_like
        filename
    phase : ndarray
        array of phase values, nm
    dx : ndarray
        inter-sample spacing, mm
    wavelength : float, optional
        wavelength of light, um
    intensity : ndarray, optional
        intensity data

    """
    defaults = _zygo_metadata_helper()
    for k, v in defaults.items():
        defaults[k] = list(v)

    all_keys_pad = [k for k in defaults.keys() if '__pad' in k]
    for key in all_keys_pad:
        del defaults[key]
    ny, nx = phase.shape

    x0 = - (nx - 1) / 2 * dx
    print(x0)
    y0 = - (ny - 1) / 2 * dx
    timestamp = datetime.datetime.now()
    ts = math.floor(timestamp.timestamp())  # unix timestamp
    buf = ctypes.create_string_buffer(834)
    # need to modify cn_x, cn_y, cn_width, cn_height, cn_n_bytes
    defaults['scale_factor'][3] = 1.
    defaults['obliquity_factor'][3] = 1.
    defaults['lateral_resolution'][3] = dx/1e3  # mm -> m
    defaults['timestamp'][3] = ts
    defaults['coords_state'][3] = 1
    defaults['coords_x'][3] =20* x0 / 1e3  # mm → m
    defaults['coords_y'][3] = y0 / 1e3
    defaults['cn_x'][3] = 0
    defaults['cn_y'][3] = 0
    defaults['cn_width'][3] = phase.shape[1]
    defaults['cn_height'][3] = phase.shape[0]
    defaults['cn_n_bytes'][3] = phase.size*4  # data gets packed to int32
    defaults['wavelength'][3] = wavelength/1e6  # um -> m

    defaults['phase_res'][3] = 1  # um -> m
    phase_res_fctr = ZYGO_PHASE_RES_FACTORS[1]

    for k, (T, lo, hi, val) in defaults.items():
        try:
            if 's' in T or T == 'c':
                # str -> bytes
                val = val.encode(ZYGO_ENC)

            struct.pack_into(T, buf, lo, val)
        except Exception as e:
            print(k, T, lo, hi, '"', val, '"', len(val.encode(ZYGO_ENC)))
            raise e

    # reverse conversion from nm into "zygos"
    # zygos -> nm
    # (raw*scale_factor*obliquity*wvl)/phase_res_fctr * 1e9
    # so nm -> zygos
    # (1e9*wvl/phase_res_factor/z)  # 1e9/1e6; I use um, they use m
    phase = np.flipud(phase)
    mask = np.isnan(phase)

    W = wavelength/1e6
    S = 1.
    O = 1.
    R = phase_res_fctr
    sf_m = (W * S * O)/R  # Metropro manual, pg 12-6
    sf_nm = sf_m
    im = (phase/1e9*(1/sf_nm)).astype(np.int32)
    im[mask] = ZYGO_INVALID_PHASE

    dt = np.dtype(np.int32).newbyteorder('>')
    bufphs = im.astype(dt).tobytes(order='C')
    if not hasattr(file, 'write'):
        file = open(file, 'wb')

    try:
        file.write(buf)
        file.write(bufphs)
    finally:
        file.close()

    return

