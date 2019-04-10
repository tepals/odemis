# -*- coding: utf-8 -*-
'''
Created on 25 April 2014

@author: Kimon Tsitsikas

Copyright © 2013-2014 Kimon Tsitsikas, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License version 2 as published by the Free Software
Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.
'''
from concurrent.futures._base import CancelledError
import logging
import numpy
from odemis import model
import odemis
from odemis.acq import align
from odemis.acq.align import autofocus
from odemis.dataio import hdf5, tiff
from odemis.util import test, timeout
import os
from scipy import ndimage
import time
import unittest


# logging.basicConfig(format=" - %(levelname)s \t%(message)s")
logging.getLogger().setLevel(logging.INFO)
# _frm = "%(asctime)s  %(levelname)-7s %(module)-15s: %(message)s"
# logging.getLogger().handlers[0].setFormatter(logging.Formatter(_frm))

CONFIG_PATH = os.path.dirname(odemis.__file__) + "/../../install/linux/usr/share/odemis/"
MBSEM_CONFIG = CONFIG_PATH + "sim/mbsem-full-sim.odm.yaml"


class TestAutofocus(unittest.TestCase):
    """
    Test autofocus functions
    """
    backend_was_running = False

    @classmethod
    def setUpClass(cls):

        # try:
        #     test.start_backend(MBSEM_CONFIG)
        # except LookupError:
        #     logging.info("A running backend is already found, skipping tests")
        #     cls.backend_was_running = True
        #     return
        # except IOError as exp:
        #     logging.error(str(exp))
        #     raise

        # find components by their role
        cls.ebeam = model.getComponent(role="e-beam")
        cls.efocus = model.getComponent(role="ebeam-focus")
        cls.sed = model.getComponent(role="se-detector")  # SEM image
        cls.diagnostic_cam = model.getComponent(role="diagnostic-ccd")
        cls.ofocus = model.getComponent(role="diagnostic-cam-focus")
        cls.stage = model.getComponent(role="stage")

        # The good focus positions are at the start up positions
        cls._opt_good_focus = cls.ofocus.position.value["z"]
        cls._sem_good_focus = cls.efocus.position.value["z"]

    @classmethod
    def tearDownClass(cls):
        # if cls.backend_was_running:
        #     return
        # test.stop_backend()
        pass

    def setUp(self):

        if self.backend_was_running:
            self.skipTest("Running backend found")

    @timeout(1000)
    def test_autofocus_opt(self):
        """
        Test AutoFocus on CCD
        """
        # The way to measure focus is a bit different between CCD and SEM
        self.ofocus.moveAbs({"z": self._opt_good_focus + 10e-6}).result()
        self.diagnostic_cam.exposureTime.value = self.diagnostic_cam.exposureTime.range[0]
        future_focus = align.AutoFocus(self.diagnostic_cam, self.ebeam, self.ofocus)
        foc_pos, foc_lev = future_focus.result(timeout=900)
        self.assertAlmostEqual(foc_pos, self._opt_good_focus, 3)
        self.assertGreater(foc_lev, 0)


    @timeout(1000)
    def test_autofocus_sem(self):
        """
        Test AutoFocus on e-beam
        """
        self.efocus.moveAbs({"z": self._sem_good_focus - 100e-06}).result()
        self.ebeam.dwellTime.value = self.ebeam.dwellTime.range[0]
        future_focus = align.AutoFocus(self.sed, self.ebeam, self.efocus)
        foc_pos, foc_lev = future_focus.result(timeout=900)
        self.assertAlmostEqual(foc_pos, self._sem_good_focus, 3)
        self.assertGreater(foc_lev, 0)

    @timeout(1000)
    def test_autofocus_optical_mbsem(self):
        """
        Test auto-focus of optical and multi-beam SEM combined.

        1. Get the z-position of the stage and the z position of the sem focus
        2. Run SEM auto-focus.
        3. Run optical auto-focus, this focuses by adjusting the z-position of the stage.
        4. Feed back the difference in z position to the SEM, this focuses by adjusting the voltage.
        5. Go back to step 1

        todo good optical focus needs to be different from zero
        todo in focus ebeam image needs to be assigned to the new good "position" when the stage is moved

        todo write convergence loop: while delta_z > stepsize stage,
        todo move code outside of tests
        todo write assertions
        todo think about cases where things go wrong. maybe look at spectograph
        todo read paper from Ryan Lane
        todo use microscope simulator image.
        todo tear down test
        Blurring in simulator is done setting the sigma of the gaussian filter to the difference between the good
        position and the current position times 1e4. So if position is equal to good position sigma is 0, so no
        blurring, this also means that there are no discrete steps in blurring.
        """
        self.efocus.good_focus.value = 0.1
        # Move the z position of the stage to go out of optical focus.
        self.ofocus.moveAbs({"z": self._opt_good_focus}).result()
        self.ofocus.moveAbs({"z": self._opt_good_focus + 400e-6}).result()
        time.sleep(2)
        self.diagnostic_cam.exposureTime.value = self.diagnostic_cam.exposureTime.range[0]

        # Move the z (working distance) of the ebeam to go out of focus.
        ebeam_movement = 200e-6
        self.efocus.moveAbs({"z": self._sem_good_focus + ebeam_movement}).result()
        time.sleep(2)
        self.ebeam.dwellTime.value = self.ebeam.dwellTime.range[0]

        # 1. get the z-pos of stage, this corresponds to the focus_position of the SEM.
        init_stage_zpos = self.stage.position.value['z']
        init_ebeam_zpos = self.efocus.position.value['z']

        # 2. Run SEM auto-focus.
        logging.info('sem')
        ebeam_focus_position = focus_sem(self.sed, self.ebeam, self.efocus)

        logging.info('optical')
        # 3. Run optical auto-focus. (optical should be out of focus here)
        delta_z = focus_optical(self.diagnostic_cam, self.ofocus, init_stage_zpos)
        logging.info(delta_z)
        # 3b. update good focus of ebeam (only for simulation).
        self.efocus.good_focus.value = self.efocus.good_focus.value + delta_z + 1e-9
        # 4. Feed back the difference in z position to the SEM.
        logging.info('sem')
        ebeam_focus_position = focus_sem(self.sed, self.ebeam, self.efocus, good_focus=ebeam_focus_position + delta_z)
        logging.info("ebeam focus position " + str(ebeam_focus_position))
        # Iterate: update ofocus good focus with difference
        # self.ofocus.moveAbs({"z": init_ebeam_zpos - ebeam_focus_position}).result()  # ?

        # numpy.testing.assert_allclose(focus_position, init_ebeam_zpos - ebeam_movement, atol=1e-4)


def focus_sem(sed, ebeam, efocus, good_focus=None):
    future_focus = align.AutoFocus(sed, ebeam, efocus, good_focus=good_focus)
    ebeam_focus_position, foc_lev = future_focus.result(timeout=900)
    return ebeam_focus_position


def focus_optical(diagnostic_cam, ofocus, init_stage_zpos):
    future_focus = align.AutoFocus(diagnostic_cam, None, ofocus)
    new_stage_zpos, foc_lev = future_focus.result(timeout=900)
    delta_z = new_stage_zpos - init_stage_zpos
    return delta_z


if __name__ == '__main__':
    unittest.main()
