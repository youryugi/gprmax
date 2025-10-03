# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
from datetime import datetime

import h5py
import numpy as np
import matplotlib.pyplot as plt

from gprMax.exceptions import CmdInputError
from .outputfiles_merge import get_output_data


def mpl_plot(filename, outputdata, dt, rxnumber, rxcomponent):
    """Creates a plot (with matplotlib) of the B-scan.

    Args:
        filename (string): Filename (including path) of output file.
        outputdata (array): Array of A-scans, i.e. B-scan data.
        dt (float): Temporal resolution of the model.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.

    Returns:
        plt (object): matplotlib plot object.
    """

    (path, filename) = os.path.split(filename)

    fig = plt.figure(num=filename + ' - rx' + str(rxnumber), 
                     figsize=(20, 10), facecolor='w', edgecolor='w')
    plt.imshow(outputdata, 
               extent=[0, outputdata.shape[1], outputdata.shape[0] * dt, 0], 
               interpolation='nearest', aspect='auto', cmap='seismic', 
               vmin=-np.amax(np.abs(outputdata)), vmax=np.amax(np.abs(outputdata)))
    plt.xlabel('Trace number')
    plt.ylabel('Time [s]')
    # plt.title('{}'.format(filename))

    # Grid properties
    ax = fig.gca()
    ax.grid(which='both', axis='both', linestyle='-.')

    cb = plt.colorbar()
    if 'E' in rxcomponent:
        cb.set_label('Field strength [V/m]')
    elif 'H' in rxcomponent:
        cb.set_label('Field strength [A/m]')
    elif 'I' in rxcomponent:
        cb.set_label('Current [A]')

    # Save a PDF/PNG of the figure
    # savefile = os.path.splitext(filename)[0]
    # fig.savefig(path + os.sep + savefile + '.pdf', dpi=None, format='pdf', 
    #             bbox_inches='tight', pad_inches=0.1)
    # fig.savefig(path + os.sep + savefile + '.png', dpi=150, format='png', 
    #             bbox_inches='tight', pad_inches=0.1)

    return plt


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plots a B-scan image.', 
                                     usage='cd gprMax; python -m tools.plot_Bscan outputfile output')
    parser.add_argument('outputfile', help='name of output file including path')
    parser.add_argument('rx_component', help='name of output component to be plotted', 
                        choices=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz'])
    # NEW: 可选静音窗口（纳秒）
    parser.add_argument('--mute_ns', type=float, default=None,
                        help='Early-time mute length in ns (zero out data from 0..mute_ns)')
    args = parser.parse_args()

    # Open output file and read number of outputs (receivers)
    f = h5py.File(args.outputfile, 'r')
    nrx = f.attrs['nrx']
    f.close()

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError('No receivers found in {}'.format(args.outputfile))

    for rx in range(1, nrx + 1):
        outputdata, dt = get_output_data(args.outputfile, rx, args.rx_component)

        # NEW: 应用静音（把前 mute_ns 的样本置零）
        if args.mute_ns is not None and args.mute_ns > 0:
            n_end = int(round((args.mute_ns * 1e-9) / dt))
            n_end = max(0, min(n_end, outputdata.shape[0]))
            if n_end > 0:
                outputdata[:n_end, :] = 0.0

        plthandle = mpl_plot(args.outputfile, outputdata, dt, rx, args.rx_component)

    plthandle.show()
    
    # Generate filename with current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_name = os.path.splitext(os.path.basename(args.outputfile))[0]
    filename = f"bscan_{base_output_name}_{current_time}.png"

    # Save into the same folder as the .out/.in file
    out_dir = os.path.dirname(os.path.abspath(args.outputfile))
    savepath = os.path.join(out_dir, filename)

    plt.savefig(savepath, dpi=300)
    print(f"B-scan saved as: {savepath}")

