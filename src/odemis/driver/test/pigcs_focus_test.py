from __future__ import division

import logging
import numpy
import os
import time
import unittest

import odemis
from odemis import model
from odemis.acq import align
from odemis.driver import pigcs, ueye
from odemis.util import timeout


class TestStage(unittest.TestCase):
    def setUp(self):
        self.focus = pigcs.Bus("test", "focus", port="/dev/ttyUSB0", axes={"z": [None, "Z", True]})

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


class TestDiagnosticCam(unittest.TestCase):
    def setUp(self):
        self.diagnostic_cam = ueye.Camera(name="camera", role="ccd")

    def test_diagnostic(self):
        self.diagnostic_cam.SetFrameRate(2)


logging.getLogger().setLevel(logging.INFO)

CONFIG_PATH = os.path.dirname(odemis.__file__) + "/../../install/linux/usr/share/odemis/"
MBSEM_CONFIG = CONFIG_PATH + "sim/mbsem-autofocus.odm.yaml"


class TestAutofocusSim(unittest.TestCase):
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
        # good_focus = numpy.random.randint(100) * 1e-6
        good_focus = 40e-6
        cls.diagnostic_cam.good_focus.value = good_focus
        cls._opt_good_focus = good_focus

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
        # Move the stage so that the image is out of focus
        center_position = 65e-6
        self.ofocus.moveAbs({"z": center_position}).result()
        numpy.testing.assert_allclose(self.ofocus.position.value["z"], center_position, atol=1e-7)
        # Run autofocus
        future_focus = align.autofocus.CLSpotsAutoFocus(self.diagnostic_cam, self.ofocus)
        foc_pos, foc_lev = future_focus.result(timeout=900)
        logging.info("found focus at {}".format(foc_pos))
        logging.info("good focus at {}".format(self._opt_good_focus))
        # todo check the minimum step we want to move.
        numpy.testing.assert_allclose(foc_pos, self._opt_good_focus, atol=0.5e-6)
        self.assertGreater(foc_lev, 0)

    def test_autofocus_different_starting_positions(self):
        for k in range(50):
            # Move the stage to a random starting position.
            start_position = numpy.random.randint(100) * 1e-6
            self.ofocus.moveAbs({"z": start_position}).result()
            # Set the good focus to a random value.
            good_focus = numpy.random.randint(100) * 1e-6
            self.diagnostic_cam.good_focus.value = good_focus
            # run autofocus
            future_focus = align.autofocus.CLSpotsAutoFocus(self.diagnostic_cam, self.ofocus)
            foc_pos, foc_lev = future_focus.result(timeout=900)
            logging.info("found focus at {}".format(foc_pos))
            logging.info("good focus at {}".format(good_focus))
            numpy.testing.assert_allclose(foc_pos, good_focus, atol=0.5e-6)


class TestAutofocusHW(unittest.TestCase):
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
        cls._optimal_focus = 43e-6  # TODO update with actual value

    @classmethod
    def tearDownClass(cls):
        # if cls.backend_was_running:
        #     return
        # test.stop_backend()
        pass

    def setUp(self):
        if self.backend_was_running:
            self.skipTest("Running backend found")

    def test_autofocus_optical_hardware(self):
        """
        Test AutoFocus on CCD
        """
        # Move the stage so that the image is out of focus
        center_position = 80e-6
        self.ofocus.moveAbs({"z": center_position}).result()
        numpy.testing.assert_allclose(self.ofocus.position.value["z"], center_position, atol=1e-7)

        time.sleep(3)
        future_focus = align.AutoFocus(self.diagnostic_cam, None, self.ofocus)
        foc_pos, foc_lev = future_focus.result(timeout=900)

        logging.info("found focus at {}".format(foc_pos))
        logging.info("good focus at {}".format(self._optimal_focus))
        self.assertGreater(foc_lev, 0)

    def test_autofocus_optical_hardware_multiple_runs(self):
        """
        Test AutoFocus on CCD
        """
        # Move the stage so that the image is out of focus
        start_positions = [0, 8e-6, 17e-6, 32e-6, 45e-6, 51e-6, 69e-6, 76e-6, 83e-6, 99e-6]
        for start_position in start_positions:
            self.ofocus.moveAbs({"z": start_position}).result()
            numpy.testing.assert_allclose(self.ofocus.position.value["z"], start_position, atol=1e-7)

            future_focus = align.AutoFocus(self.diagnostic_cam, None, self.ofocus)
            foc_pos, foc_lev = future_focus.result(timeout=900)

            logging.info("found focus at {}".format(foc_pos))
            logging.info("good focus at {}".format(self._optimal_focus))
            numpy.testing.assert_allclose(foc_pos, self._optimal_focus, atol=0.5e-6)

    def test_autofocus_optical_hardware_start_at_good_focus(self):
        """
        Test AutoFocus on CCD
        """
        # Move the stage so that the image is out of focus
        start_position = self._optimal_focus
        self.ofocus.moveAbs({"z": start_position}).result()
        # check that it moved to the correct starting position
        numpy.testing.assert_allclose(self.ofocus.position.value["z"], start_position, atol=1e-7)

        # Run autofocus
        future_focus = align.AutoFocus(self.diagnostic_cam, None, self.ofocus)
        foc_pos, foc_lev = future_focus.result(timeout=900)
        # Test that the correct focus has been found.
        logging.info("found focus at {}".format(foc_pos))
        logging.info("good focus at {}".format(self._optimal_focus))
        numpy.testing.assert_allclose(foc_pos, self._optimal_focus, atol=0.5e-6)
