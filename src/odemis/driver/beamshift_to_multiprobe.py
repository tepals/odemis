# !/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Created on 25 May 2020

@author: Philip Winkler

Copyright © 2020 Philip Winkler, Delmic

This script provides a command line interface for the alignment of the beamshift controller to the
multiprobe axes. It determines 1) the angle between the beam shift and the multiprobe coordinate system,
and 2) the gain (pixels per ampere). Both parameters need to be set as metadata to the beamshift controller.

This script implements the following steps:
* Acquire image of multiprobe using the diagnostic camera, get position of grid spots. Move the multiprobe pattern by setting a
deflection in amperes on the beam shift controller, get the new image and the new position of the grid spots.
Repeat several times with different beamshifts.
* Calculate angle between the axes of the beam shift coordinate system and the axes of the mp coordinate system
* Calculate the number of pixels on the diagnostic camera per ampere

Prerequisites
* scan_to_multiprobe alignment has been completed (there is no rotation rotation of mp with respect to diagnostic camera
    coordinate axes)

Information on how to run this script can be found in the sonic-engineering repo:
sonic-engineering/doc/useful commands.txt

To use this script first run the following line or add it at the end of .bashrc.
export PYTHONPATH="$HOME/development/sonic-engineering/:$PYTHONPATH"

"""
from __future__ import division

import argparse
import logging
import math
import sys
import time

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


def get_beamshift_rotation(scan_rotation, px_shifts):
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
        The angle in rad between the x direction of the beamshift coordinate system and the orientation of the multiprobe pattern.
    gain: (float)
        The conversion factor pixels per ampere.
    """
    rotation = line_to_angle(numpy.array(px_shifts))
    # Movement (approximately 100 deg) wrt diagnostic camera frame --> coordinate system of dc shift rotated by
    # 360 - angle. Now subtract shift of multiprobe coordinate system wrt diagnostic camera coordinate system
    # --> - scan_rotation
    rotation_dc_frame_cam_frame = 2 * math.pi - rotation
    rotation_dc_frame_multiprobe = rotation_dc_frame_cam_frame - scan_rotation
    return rotation_dc_frame_multiprobe

def get_beamshift_gain(pxshifts, ampshifts):
    # Get gain (amps per px)
    # Difference of the position in pixels between first and last acquisition divided by difference in amperes.
    xdiff = abs(pxshifts[-1][0] - pxshifts[0][0])
    ydiff = abs(pxshifts[-1][1] - pxshifts[0][1])
    pxdiff = math.hypot(xdiff, ydiff)

    xdiff_amps = abs(ampshifts[-1][0] - ampshifts[0][0])
    ydiff_amps = abs(ampshifts[-1][1] - ampshifts[0][1])
    ampdiff = math.hypot(xdiff_amps, ydiff_amps)
    gain = ampdiff / pxdiff
    # TODO: this is a very simple calculation, it would be better to make use of all the acquired shifts and
    #  use linear regression to get slope (and check standard error)

    return gain

def _get_shifts(ccd, beamshifter, amp_shifts):
    n_spots = (8, 8)
    shifts_pxs = []  # corresponding pixel positions
    for shift in amp_shifts:
        logging.debug("Current register values: %s" % beamshifter._read_registers())
        # Move the mp pattern along the x axis of the beam shift coordinate system by using the beam shift control
        # y=0 is a good position (near center of camera image)
        registers = beamshifter._amp_to_diff_int64(shift)
        logging.debug("Writing values: %s" % registers)
        beamshifter._write_registers(registers)  # [-x, -y, x, y], y value at position near the middle

        time.sleep(3)  # wait for firmware to adjust beam
        logging.debug("New register values: %s" % beamshifter._read_registers())
        image = ccd.data.get(asap=False)
        # Calculate the translation and rotation in the left-handed diagnostic camera coordinate system.
        _, translation, *_ = FindGridSpots(image, n_spots)
        # Flip translation to go to the right-handed diagnostic camera coordinate system.
        translation[1] = image.shape[1] - translation[1]
        logging.error("Found translation of %s." % translation)
        shifts_pxs.append(translation)
    return shifts_pxs

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
    parser.add_argument("--beamshift-role", dest="beamshift_role", default="beamshift-control",
                        metavar="<component>",
                        help="Role of the beamshift controller to connect to via the Odemis back-end. Ex: 'beamshift-control'.")
    parser.add_argument('--px-size-diagcam', dest="px_size_diagcam", type=float, default=75,
                        help="The pixel size in nm/px on the diagnostic camera.")
    parser.add_argument('--scan-rotation', dest="scan_rot", type=float, default=0.9,
                        help="Scan rotation in rad as determined with scan_to_multiprobe.py script.")

    options = parser.parse_args(args[1:])
    if options.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    if get_backend_status() != BACKEND_RUNNING:
        raise ValueError("Backend is not running.")

    ccd = model.getComponent(role="diagnostic-ccd") # TODO

    if options.beamshift_role == 'None':
        # TODO: For testing, remove later
        beamshifter = tfsbc.BeamShiftController("DC Offset", None, "/dev/ttyUSB*", "RS485")
    else:
        beamshifter = model.getComponent(role=options.beamshift_role)

    try:
        xshifts_amps = [(-8e-3, 0), (-6e-3, 0), (-4e-3, 0), (-2e-3, 0), (0, 0)]
        # Move x to the middle, so there is more room for y movement (it's almost diagonal)
        yshifts_amps = [(-4e-3, -32e-3), (-4e-3, -16e-3), (-4e-3, 0e-3, 0), (-4e-3, 16e-3),
                        (-4e-3, 32e-3)]
        xshifts_pxs = _get_shifts(ccd, beamshifter, xshifts_amps)
        yshifts_pxs = _get_shifts(ccd, beamshifter, yshifts_amps)
        phi = get_beamshift_rotation(options.scan_rot, xshifts_pxs)  # expected to be around 3.6
        xgain = get_beamshift_gain(xshifts_pxs, xshifts_amps)/ (options.px_size_diagcam * 10e-9)
        ygain = get_beamshift_gain(yshifts_pxs, yshifts_amps)/ (options.px_size_diagcam * 10e-9)
        print("\nBeamshift Parameters\nphi:\t%.8f rad\nxgain:\t%.8f amp/px\nygain:\t%.8f amp/px" % (phi, xgain, ygain))
    except Exception as exp:
        logging.error("%s", exp, exc_info=True)
        return 128

    # Update metadata of beamshift driver
    if options.beamshift_role != 'None':
        # Get parameters for beamshift driver
        # gain in ampere / px, px_size in nm / px
        beamshift_gain = (xgain / (options.px_size_diagcam * 10e-9),  # amp/m
                          ygain / (options.px_size_diagcam * 10e-9))
        beamshifter.updateMetadata({model.MD_CALIB_SCALE: beamshift_gain,
                                    model.MD_CALIB_TRANSLATION: [0, 0],
                                    model.MD_CALIB_ROTATION: phi})  # rad

    return 0


if __name__ == '__main__':
    ret = main(sys.argv)
    exit(ret)
