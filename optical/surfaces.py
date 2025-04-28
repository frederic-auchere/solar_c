from optics.surfaces import Substrate
from optics.geometry import Point, Polygon


class EGASubstrate(Substrate):

    def __init__(self, *args, fiducials=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.fiducials = None if fiducials is None else Polygon([Point(*xy, float(self.sag(xy).data)) for xy in fiducials])
