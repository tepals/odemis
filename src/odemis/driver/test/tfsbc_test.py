# -*- coding: utf-8 -*-
"""
Created on 11 May 2020

@author: Philip Winkler

Copyright © 2020 Philip Winkler, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License version 2 as published by the Free Software
Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.
"""
from __future__ import division

from odemis.driver import tfsbc
from odemis import model
import unittest
import logging
import os
import time
import math

TEST_NOHW = (os.environ.get("TEST_NOHW", 0) != 0)  # Default to Hw testing

logging.getLogger().setLevel(logging.DEBUG)
logging.basicConfig(format="%(asctime)s  %(levelname)-7s %(module)s:%(lineno)d %(message)s")

if TEST_NOHW:
    PORT = "/dev/fake"
else:
    PORT = "/dev/ttyUSB*"


# @skip("skip")
class TestBeamShiftController(unittest.TestCase):
    """
    Tests the beam controller driver.
    """

    @classmethod
    def setUpClass(cls):
        cls.bc = tfsbc.BeamShiftController("DC Offset", None, PORT, "RS485")
        cls.bc.updateMetadata({model.MD_CALIB_SCALE: [300, 300],  # amp/m
                               model.MD_CALIB_TRANSLATION: [0, 0],
                               model.MD_CALIB_ROTATION: 0})  # rad

    @classmethod
    def tearDownClass(cls):
        pass

    def assertTupleAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """
        check two tuples are almost equal (value by value)
        """
        for f, s in zip(first, second):
            self.assertAlmostEqual(f, s, places=places, msg=msg, delta=delta)

    def test_read_write(self):
        vals = [27000, 37000, 20000, 44000]
        self.bc._write_registers(vals)
        ret = self.bc._read_registers()
        self.assertTupleAlmostEqual(vals, list(ret), places=1)

    def test_shifts(self):
        """
        Move to different shifts, test if .shift VA is updated correctly. Wait in between,
        so the effect can be seen on the hardware.
        """
        self.bc.updateMetadata({model.MD_CALIB_ROTATION: 0})  # rad
        self.bc.updateMetadata({model.MD_CALIB_SCALE: [130, 130]})  # rad

        shift = (0, 0)
        self.bc.shift.value = shift
        self.assertTupleAlmostEqual(self.bc.shift.value, shift, places=9)
        time.sleep(1)

        shift = (-50e-6, 0)  # 50 µm ~ 75 px
        self.bc.shift.value = shift
        self.assertTupleAlmostEqual(self.bc.shift.value, shift, places=9)
        time.sleep(1)

        shift = (-50e-6, -50e-6)
        self.bc.shift.value = shift
        self.assertTupleAlmostEqual(self.bc.shift.value, shift, places=9)
        time.sleep(1)

        shift = (0, -50e-6)
        self.bc.shift.value = shift
        self.assertTupleAlmostEqual(self.bc.shift.value, shift, places=9)

        shift = (0, 0)
        self.bc.shift.value = shift
        self.assertTupleAlmostEqual(self.bc.shift.value, shift, places=9)

    def test_write_time(self):
        startt = time.time()
        shift = (-3e-6, 1e-6)
        self.bc.shift.value = shift
        self.assertLess(time.time() - startt, 0.03, "Reading/writing took more than 30 ms.")
        logging.debug("Shift value set to %s", self.bc.shift.value)
        self.assertTupleAlmostEqual(self.bc.shift.value, shift, places=9)

    def test_amp_int64_conversion(self):
        # 0, 0 --> signal at half the value (rounded up)
        ret = self.bc._amp_to_diff_int64((0, 0))
        for i in ret:
            self.assertEqual(i, int(round(0xFFFF / 2)))

        # Same x, y --> differential low and high the same in both directions
        ret = self.bc._amp_to_diff_int64((5e-6, 5e-6))
        self.assertEqual(ret[0], ret[1])  # low x, low y
        self.assertEqual(ret[2], ret[3])  # upper x, upper y

        # x = -y --> differential low and high reversed
        ret = self.bc._amp_to_diff_int64((-5e-6, 5e-6))
        self.assertEqual(ret[0], ret[3])  # low x, upper y
        self.assertEqual(ret[1], ret[2])  # upper x, low y

        # Back to original state
        self.bc._amp_to_diff_int64((0, 0))

    def test_transformation_function(self):
        # Rotation by pi
        ret = self.bc._transform((10e-6, 0), (1, 1), math.pi, (0, 0))
        self.assertTupleAlmostEqual(ret, (-10e-6, 0), places=9)
        ret = self.bc._reverse_transform(ret, (1, 1), math.pi, (0, 0))  # Reverse rotation
        self.assertTupleAlmostEqual(ret, (10e-6, 0), places=9)
        ret = self.bc._transform((0, 10e-6), (1, 1), math.pi, (0, 0))
        self.assertTupleAlmostEqual(ret, (0, -10e-6), places=9)

        # Rotation by pi/2
        ret = self.bc._transform((10e-6, 0), (1, 1), math.pi / 2, (0, 0))
        # Coordinate system moved by pi/2, so we expect a move in the opposite direction
        self.assertTupleAlmostEqual(ret, (0, -10e-6), places=9)

        # Rotation by pi/4 = 45 deg
        # Shift angle also 45 deg --> expecting sqrt(x**2 + y**2) in x, 0 in y
        ret = self.bc._transform((10e-6, 10e-6), (1, 1), math.pi / 4, (0, 0))
        self.assertTupleAlmostEqual(ret, (math.hypot(10e-6, 10e-6), 0), places=9)

        # Gain
        ret = self.bc._transform((10e-6, 0), (100, 100), math.pi, (0, 0))
        # 100 amps/m * 10e-6 m
        self.assertTupleAlmostEqual(ret, (-1e-3, 0), places=9)
        ret = self.bc._transform((0, 10e-6), (100, 100), math.pi, (0, 0))
        self.assertTupleAlmostEqual(ret, (0, -1e-3), places=9)
        ret = self.bc._reverse_transform(ret, (100, 100), math.pi, (0, 0))
        self.assertTupleAlmostEqual(ret, (0, 10e-6), places=9)

        # Realistic rotation
        ret = self.bc._transform((10e-6, 0), (130, 130), 3.66, (0, 0))
        # expected to be in second quadrant (x<0, y>0)
        self.assertLess(ret[0], 0)
        self.assertGreater(ret[1], 0)

    def test_rotation_shift(self):
        """
        Test if setting shift works with non-zero rotation.
        """
        # Previously set rotation: shift should return value that was set (with rounding error)
        self.bc.updateMetadata({model.MD_CALIB_ROTATION: math.pi / 2})  # rad
        shift = (-0.5e-9, -0.2e-9)
        self.bc.shift.value = shift
        self.assertTupleAlmostEqual(self.bc.shift.value, shift, places=9)

        # Setting value again --> shift is given in rotated coordinate system
        # 360 deg turn (from pi/2 to 2pi + pi/2) --> same value
        self.bc.updateMetadata({model.MD_CALIB_ROTATION: 2.5 * math.pi})  # rad
        self.assertTupleAlmostEqual(self.bc.shift.value, shift, places=9)

        # 180 deg turn --> negative value
        self.bc.updateMetadata({model.MD_CALIB_ROTATION: 3 * math.pi / 2})  # rad
        self.assertTupleAlmostEqual(self.bc.shift.value, (-shift[0], -shift[1]), places=9)

        # Back to normal rotation for other tests
        self.bc.updateMetadata({model.MD_CALIB_ROTATION: 0})  # rad

    def test_large_shift(self):
        """
        A large shift should be clipped to the device maximum. The function ._write_registers should
        raise an error if any values larger than 0xFFFF is written.
        """
        shift = (20e-6, 2.3e-6)
        self.bc.shift.value = shift
        # Maximum shift: 14.1 nm
        self.assertTupleAlmostEqual(self.bc.shift.value, (14.1e-9, 2.3e-9), places=9)

        with self.assertRaises(ValueError):
            self.bc._write_registers([70000, 49990, 20000, 4])


if __name__ == "__main__":
    unittest.main()

