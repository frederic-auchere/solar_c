import argparse
from optical.zygo import EGAFit


def main(xlsx_file, outfile=None):
    fitter = EGAFit.from_xlsx(xlsx_file)
    fit = fitter.fit()
    fitter.make_report(outfile=outfile)


def sfe():
    """
    Command line interface
    :return:
    """
    parser = argparse.ArgumentParser(prog="Surface Fitting",
                                     description="Processes a sequence of files based as described in a xls file")
    parser.add_argument("xlsx_file", help="Input .excel filename")
    parser.add_argument("-o", "--outfile",
                        help="Output filename.",
                        type=str)

    args = parser.parse_args()

    main(**vars(args))