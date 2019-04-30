from __future__ import division

import logging
import numpy
import os
import unittest

from odemis import model
from odemis.acq import align
from odemis.util import test, timeout

TEST_NOHW = (os.environ.get("TEST_NOHW", "0") != "0")  # Default to Hw testing
TEST_NOHW = True

logging.getLogger().setLevel(logging.INFO)

MBSEM_CONFIG = "/home/pals/development/sonic-odm-yaml/mbsem-optical-autofocus-sim.odm.yaml"


class TestAutofocusSim(unittest.TestCase):
    """
    Test auto focus functions
    """
    backend_was_running = False

    @classmethod
    def setUpClass(cls):
        try:
            test.start_backend(MBSEM_CONFIG)
        except LookupError:
            logging.info("A running backend is already found, skipping tests")
            cls.backend_was_running = True
            return
        except IOError as exp:
            logging.error(str(exp))
            raise

        # find components by their role
        cls.diagnostic_cam = model.getComponent(role="diagnostic-ccd")
        cls.stage = model.getComponent(role="stage")
        cls.ofocus = model.getComponent(role="diagnostic-cam-focus")

    @classmethod
    def tearDownClass(cls):
        if cls.backend_was_running:
            return
        test.stop_backend()

    def setUp(self):
        if self.backend_was_running:
            self.skipTest("Running backend found")

    @timeout(1000)
    def test_autofocus_opt(self):
        """
        Test AutoFocus on CCD
        """
        # set the position where the image is in focus.
        good_focus = 0
        self.diagnostic_cam.good_focus.value = good_focus
        # Move the stage so that the image is out of focus
        center_position = 17e-6
        self.ofocus.moveAbs({"z": center_position}).result()
        # Run auto focus
        future_focus = align.autofocus.CLSpotsAutoFocus(self.diagnostic_cam, self.ofocus)
        foc_pos, foc_lev = future_focus.result(timeout=900)
        # Check that the auto focus converged to the correct position and the stage moved to the correct position
        numpy.testing.assert_allclose(foc_pos, good_focus, atol=1e-6)
        numpy.testing.assert_allclose(self.ofocus.position.value["z"], foc_pos, atol=1e-6)

    def test_autofocus_different_starting_positions(self):
        for k in range(20):
            # Move the stage to a random starting position.
            start_position = numpy.random.randint(100) * 1e-6
            self.ofocus.moveAbs({"z": start_position}).result()
            # Set the good focus to a random value.
            good_focus = numpy.random.randint(100) * 1e-6
            logging.info("start {}, good focus {}".format(start_position, good_focus))
            self.diagnostic_cam.good_focus.value = good_focus
            # run auto focus
            future_focus = align.autofocus.CLSpotsAutoFocus(self.diagnostic_cam, self.ofocus)
            foc_pos, foc_lev = future_focus.result(timeout=900)
            logging.info("found focus at {}".format(foc_pos))
            logging.info("good focus at {}".format(good_focus))
            numpy.testing.assert_allclose(foc_pos, good_focus, atol=0.5e-6)


class TestAutofocusHW(unittest.TestCase):
    """
    Test auto focus functions
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

        # if TEST_NOHW:
        #     raise unittest.SkipTest('No HW present. Skipping tests.')

        # find components by their role
        cls.diagnostic_cam = model.getComponent(role="diagnostic-ccd")
        cls.stage = model.getComponent(role="stage")
        cls.ofocus = model.getComponent(role="diagnostic-cam-focus")
        cls._optimal_focus = 50e-6  # TODO update with actual value

    @classmethod
    def tearDownClass(cls):
        # if cls.backend_was_running:
        #     return
        # test.stop_backend()
        pass

    def setUp(self):
        # if self.backend_was_running:
        #     self.skipTest("Running backend found")
        pass

    def test_autofocus_optical_hardware(self):
        """
        Test AutoFocus on CCD
        """
        # Move the stage so that the image is out of focus
        self.diagnostic_cam._setExposureTime(0.01)
        center_position = 40e-6
        self.ofocus.moveAbs({"z": center_position}).result()
        center_position = 74e-6
        self.ofocus.moveAbs({"z": center_position}).result()
        # time.sleep(1)
        numpy.testing.assert_allclose(self.ofocus.position.value["z"], center_position, atol=1e-7)

        # time.sleep(3)
        # future_focus = align.AutoFocus(self.diagnostic_cam, None, self.ofocus)
        # foc_pos, foc_lev = future_focus.result(timeout=900)
        #
        # logging.info("found focus at {}".format(foc_pos))
        # logging.info("good focus at {}".format(self._optimal_focus))
        # numpy.testing.assert_allclose(foc_pos, self._optimal_focus, atol=0.5e-6)

    def test_autofocus_optical_hardware_multiple_runs(self):
        """
        Test AutoFocus on CCD
        """
        # Move the stage so that the image is out of focus
        results = []
        for start_position in range(100):
            try:
                start_position = start_position * 1e-6
                logging.info("start pos {}".format(start_position))
                self.ofocus.moveAbs({"z": start_position}).result()
                numpy.testing.assert_allclose(self.ofocus.position.value["z"], start_position, atol=1e-6)

                future_focus = align.AutoFocus(self.diagnostic_cam, None, self.ofocus)
                foc_pos, foc_lev = future_focus.result(timeout=900)

                logging.info("found focus at {}".format(foc_pos))
                logging.info("good focus at {}".format(self._optimal_focus))
                # numpy.testing.assert_allclose(foc_pos, self._optimal_focus, atol=0.5e-6)
                result = numpy.allclose(foc_pos, self._optimal_focus, atol=1e-6)
                numpy.testing.assert_allclose(self.ofocus.position.value["z"], foc_pos, atol=1e-6)
                logging.info(result)
                results.append(result)
            except Exception as e:
                logging.info("{}".format(e))
                continue
        self.assertTrue(numpy.all(results))

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
