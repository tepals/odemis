import logging
import os
import time
import unittest

import numpy

import odemis
from odemis import model
from odemis.util import timeout

from odemis.acq import align
from odemis.driver import pigcs, simulated


class TestFocus(unittest.TestCase):
    def setUp(self):
        self.focus = pigcs.Bus("test", "focus", port="/dev/ttyUSB0", axes={"z": [None, "Z", True]})
        # self.focus = simulated.Stage("test", "focus", axes=["z"], rng={"z": [0, 100e-6]})

        self.axis = self.focus._axis_to_cc["z"][1]
        self.controler = self.focus._axis_to_cc["z"][0]

    @timeout(1000)
    def test_movement(self):
        """
        Test AutoFocus on CCD
        """
        # Test if it is referenced
        self.assertTrue(self.focus.referenced.value)
        # Test move absolute
        new_pos = 1.5e-5
        f = self.focus.moveAbs({"z": new_pos})
        f.result()
        self.assertAlmostEqual(self.focus.position.value["z"], new_pos, places=7)

        # Test move relative
        original_position = self.focus.position.value["z"]
        shift = 1e-6
        f = self.focus.moveRel({"z": shift})
        f.result()
        self.assertAlmostEqual(self.focus.position.value["z"], original_position + shift, places=7)

    def test_autofocus(self):
        # self.diagnostic_cam.exposureTime.value = self.diagnostic_cam.exposureTime.range[0]
        # future_focus = align.AutoFocus(self.diagnostic_cam, self.ebeam, self.ofocus)
        # foc_pos, foc_lev = future_focus.result(timeout=900)
        # self.assertAlmostEqual(foc_pos, self._opt_good_focus, 3)
        # self.assertGreater(foc_lev, 0)
        pass


# logging.basicConfig(format=" - %(levelname)s \t%(message)s")
logging.getLogger().setLevel(logging.INFO)
# _frm = "%(asctime)s  %(levelname)-7s %(module)-15s: %(message)s"
# logging.getLogger().handlers[0].setFormatter(logging.Formatter(_frm))

CONFIG_PATH = os.path.dirname(odemis.__file__) + "/../../install/linux/usr/share/odemis/"
MBSEM_CONFIG = CONFIG_PATH + "sim/mbsem-autofocus.odm.yaml"


class TestAutofocus(unittest.TestCase):
    """
    Test autofocus functions
    """
    backend_was_running = False

    @classmethod
    def setUpClass(cls):
        # todo raise unittest.skip if (no) running backend
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
        cls.diagnostic_cam = model.getComponent(role="diagnostic-ccd")
        cls.stage = model.getComponent(role="stage")
        cls.ofocus = model.getComponent(role="diagnostic-cam-focus")

        # The good focus positions are at the start up positions
        # cls.ofocus.moveAbs({"z": 0}).result()
        good_focus = 0
        cls.diagnostic_cam.good_focus.value = good_focus
        cls._opt_good_focus = good_focus  # cls.ofocus.position.value["z"]

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
        # self.assertAlmostEqual(self.ofocus.position.value["z"], self._opt_good_focus)
        # Move the stage so that the image is out of focus
        center_position = 80e-6
        self.ofocus.moveAbs({"z": center_position}).result()
        numpy.testing.assert_allclose(self.ofocus.position.value["z"], center_position, atol=1e-7)

        time.sleep(4)
        self.diagnostic_cam.exposureTime.value = self.diagnostic_cam.exposureTime.range[0]
        future_focus = align.AutoFocus(self.diagnostic_cam, None, self.ofocus)
        foc_pos, foc_lev = future_focus.result(timeout=900)
        logging.info("found focus at {}".format(foc_pos))
        logging.info("good focus at {}".format(self._opt_good_focus))
        # todo check the minimum step we want to move.
        numpy.testing.assert_allclose(foc_pos, self._opt_good_focus, atol=1e-6)
        self.assertGreater(foc_lev, 0)
