# !/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Created on 25 May 2020

@author: Philip Winkler

Copyright © 2020 Philip Winkler, Delmic

This script provides a command line interface for the alignment of the beamshift controller to the
multiprobe axes. It determines 1) the angle between the beam shift and the multiprobe coordinate system,
and 2) the gain (pixels per ampere). Both parameters need to be set as metadata to the beamshift controller.

The code is very similar to the scan_to_multiprobe.py script.
This script implements the following steps:
* Acquire image of multiprobe, get position of grid spots. Move the multiprobe pattern by setting a
deflection in amperes to the beam shift controller, get the new image and the new position of the grid spots.
Repeat several times with different beamshifts.
* Calculate angle of beam shift relative to angle of multiprobe
* Calculate the number of pixels per ampere

Prerequisites
* scan_to_multiprobe alignment has been completed (there is no rotation in the multiprobe wrt the diagnostic camera)

Information on how to run this can be found in the sonic-engineering repo:
sonic-engineering/doc/useful commands.txt

To use this script first run the following line or add it at the end of .bashrc.
export PYTHONPATH="$HOME/development/sonic-engineering/:$PYTHONPATH"

"""
from __future__ import division

import argparse
import logging
import math
import sys

import numpy
from scipy.stats import linregress

from odemis import model
from odemis.acq.align.spot import FindGridSpots
from odemis.driver import tfsbc
from odemis.util.driver import get_backend_status, BACKEND_RUNNING


def line_to_angle(coordinates):
    """
    Fit a line through a set of coordinates and calculate the angle between that
    line and the x-axis. Positive angle is clockwise, negative angle is counter clockwise.

    Parameters
    ----------
    coordinates: (ndarray of shape nx2)
        (x, y) center coordinates of the moved spot grids.

    Returns
    -------
    phi: (float)
        Angle in radians.
    """
    # Construct a system of linear equations to fit the line a*x + b*y + c = 0
    n = coordinates.shape[0]
    if n < 2:
        raise ValueError("Need 2 or more coordinates, cannot find deflection "
                         "angle for {} coordinates.".format(n))
    elif n == 2:
        x = coordinates[1, :] - coordinates[0, :]
        # Determine the angle of this line to the x-axis, -pi/2 < phi < pi/2
        phi = numpy.arctan2(x[0], x[1])
    else:
        A = numpy.hstack((coordinates, numpy.ones((n, 1))))
        # Solve the equation A*x = 0; i.e. find the null space of A. The solution is
        # the eigenvector corresponding to the smallest singular value.
        U, s, V = numpy.linalg.svd(A, full_matrices=False)
        x = numpy.transpose(V[-1])
        # Determine the angle of this line to the x-axis, -pi/2 < phi < pi/2
        phi = numpy.arctan2(-x[0], x[1])
    phi = (phi + math.pi / 2) % math.pi - math.pi / 2
    return phi


def get_beamshift_parameters(ccd, beamshifter):
    """
    Find the angle and the gain of the beamshift controller.

    Parameters
    ----------
    ccd: (odemis.model.DigitalCamera)
        A camera object of the diagnostic camera.
    beamshifter: (odemis.model.BeamShiftController)
        An instance of the beamshift controller driver.

    Returns
    -------
    phi: (float)
        The angle between the x direction of the beamshift and the orientation of the multiprobe pattern
    gain: (float)
        The conversion factor pixels per ampere.
    """
    n_spots = (8, 8)
    shifts_amps = [-16e-3, -12e-3, -8e-3, -4e-3, 0]  # amps to set in x direction (y shift remains at 0)
    shifts_pxs = []  # corresponding pixels
    rotations = []  # registered rotation
    for offset in shifts_amps:
        conversion_factor = 0xFFFF / (tfsbc.RANGE_AMPS[1] - tfsbc.RANGE_AMPS[0])
        valuePos = round((offset - tfsbc.RANGE_AMPS[0]) * conversion_factor)
        valueNeg = round((-offset - tfsbc.RANGE_AMPS[0]) * conversion_factor)
        logging.warning("Writing value: %s" % valuePos)
        beamshifter._write_registers([valueNeg, 30768, valuePos, 34768])  # [-x, -y, x, y], y = 0

        image = ccd.data.get(asap=False)
        # Calculate the translation and rotation in the left-handed diagnostic camera coordinate system.
        _, translation, *_ = FindGridSpots(image, n_spots)
        # Flip translation to go to the right-handed diagnostic camera coordinate system.
        translation[1] = image.shape[1] - translation[1]
        logging.warning("Found shift %s" % translation)
        shifts_pxs.append(translation)

    rotation = line_to_angle(numpy.array(shifts_pxs))
    pxs = [math.hypot(x, y) for (x, y) in shifts_pxs]
    slope, _, _, _, std_err = linregress(pxs, shifts_amps)  # TODO: check standard error
    return rotation, slope


def main(args):
    """
    Handles the command line arguments.

    Parameters
    ----------
    args: The list of arguments passed.

    Returns
    -------
    (int)
        value to return to the OS as program exit code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument("--ccd-role", dest="ccd_role", default="diagnostic-ccd",
                        metavar="<component>",
                        help="Role of the camera to connect to via the Odemis back-end. Ex: 'ccd'.")
    parser.add_argument("--beamshift-role", dest="beamshift_role", default="beamshift-control",
                        metavar="<component>",
                        help="Role of the camera to connect to via the Odemis back-end. Ex: 'ccd'.")

    options = parser.parse_args(args[1:])
    if options.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    if get_backend_status() != BACKEND_RUNNING:
        raise ValueError("Backend is not running.")

    ccd = model.getComponent(role=options.ccd_role)

    if options.beamshift_role == 'None':
        beamshifter = tfsbc.BeamShiftController("DC Offset", None, "/dev/ttyUSB*", "RS485")
    else:
        beamshifter = model.getComponent(role=options.beamshift_role)
    try:
        phi, gain = get_beamshift_parameters(ccd, beamshifter)
        print("\nBeamshift Parameters\nphi:\t%.8f rad\ngain:\t%.8f amp/px" % (phi, gain))
    except Exception as exp:
        logging.error("%s", exp, exc_info=True)
        return 128
    return 0


if __name__ == '__main__':
    ret = main(sys.argv)
    exit(ret)
