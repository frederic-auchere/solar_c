import argparse
from optical.zygo import EGAFit
import matplotlib

matplotlib.use('Agg')


def main(report_type, xlsx_file, outfile=None):
    fitter = EGAFit.from_xlsx(xlsx_file)
    fitter.fit()
    if report_type == 'tilt':
        fitter.tilt_fix()

    fitter.make_report(outfile=outfile, save_stack=True)


def make_report():
    """
    Command line interface
    :return:
    """
    parser = argparse.ArgumentParser(prog="Surface Fitting",
                                     description="Processes a sequence of files as described in a xlsx file")
    parser.add_argument("type", choices=['sfe', 'tilt'], help="'sfe' or 'tilt'")
    parser.add_argument("xlsx_file", help="Input excel filename")
    parser.add_argument("-o", "--outfile",
                        help="Output filename.",
                        type=str)

    args = vars(parser.parse_args())

    report_type = args.pop('type')
    xlsx_file = args.pop('xlsx_file')
    main(report_type, xlsx_file, **args)
