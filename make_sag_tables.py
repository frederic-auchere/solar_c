from optical import sw_substrate, lw_substrate
import numpy as np


for substrate, name in zip([lw_substrate, sw_substrate], ['lw_sag.txt', 'sw_sag.txt']):
    substrate.useful_area = None
    sag = substrate.sag(substrate.grid())
    table = np.stack([substrate.grid()[0].flatten(), substrate.grid()[1].flatten(), sag.flatten()])

    with open(name, 'w') as f:
        np.savetxt(name, table.T, header='x [mm] y [mm] z [mm]')
