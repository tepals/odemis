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

import logging
import math

import numpy
import serial
import serial.tools.list_ports
from pymodbus.client.sync import ModbusSerialClient

from odemis import model
from odemis.model import HwError
from odemis.util.transform import SimilarityTransform

# Parameters for connection
BAUDRATE = 230400
TIMEOUT = 1
BYTESIZE = 8
PARITY = serial.PARITY_NONE
STOPBITS = 1

# Modbus level addresses
SLAVE_UNIT = 2

# Modbus registers
BEAMDEFL_LX = 0  # lower x beam deflection element control
BEAMDEFL_LY = 1  # lower y beam deflection element control
BEAMDEFL_UX = 2  # upper x beam deflection element control
BEAMDEFL_UY = 3  # upper y beam deflection element control

RANGE_AMPS = [-42.2e-3, 42.2e-3]  # range of currents accepted by device, correspond to deflection of beam
RANGE_AMPS_REDUCED = [-20e-3, 20e-3]  # range of currents that can be seen in the diagnostic camera
DEFAULT_GAIN = [3e6, 30e6]  # 3e6 amp/m = 3 mamp / nm

RH = 0  # right-handed coordinate system
LH = 1  # left-handed coordinate system


def transform_coordinates(value, rotation, offset=(0,0), source_orientation=RH, target_orientation=RH):
    """
    Transform 2D coordinates from source coordinate system to target coordinate system.
    The coordinate systems can be rotated and translated with respect to each other and may have different orientations.
    This function does not handle scaling. In the use cases here, the scales are different in the x and y axes, so scaling before and after the
    rotation will return different results. Depending on which option is appropriate, the scaling needs to be carried
    out before or after calling this function.
    :param value (float, float): x, y value in the source coordinate system
    :param rotation (float): rotation of the target coordinate system with respect to the source coordinate system in rad
    :param offset (float, float): x, y offset to add to the transformed value (in the target coordindate system and target unit)
    :param source_orientation (RH or LH): orientation (handedness) of the source coordinate system
    :param target_orientation (RH or LH): orientation (handedness) of the target coordinate system
    :return (float, float): transformed value in target units
    """
    # Transform value to right-handed version of source coordinate system
    if source_orientation == LH:
        value = (value[0], -value[1])

    # Rotation angle of vector is opposite of rotation angle of coordinate systems
    rotation = -rotation

    # Regular right-handed transformation to right-handed target system
    transform = SimilarityTransform(rotation=rotation, translation=offset)
    value = transform(value)

    # Transform result to left-handed target system
    if target_orientation == LH:
        value = (value[0], -value[1])

    return value


class BeamShiftController(model.HwComponent):
    """
    Driver for the Thermofischer beam deflection controller.
    This class provides the .shift VA containing a tuple of two floats which describe
    the x and y beam offset in m in the stage coordinate system.

    The device accepts ampere values. The conversion from meter to amperes depends on
    three metadata:
        * MD_CALIB_SCALE (float, float): conversion factor ampere/meter (gain)
        * MD_CALIB_TRANSLATION (float, float): offset in amps (already in the beam coordinate system, will not be transformed)
        * MD_CALIB_ROTATION (float): angle in radians
    """

    def __init__(self, name, role, port=None, serialnum=None, **kwargs):
        """
        :param port (str): port (e.g. /dev/ttyUSB0)
        :param serialnum (str): serial number of RS485 adapter
        """
        # .hwVersion, .swVersion not available
        model.HwComponent.__init__(self, name, role, **kwargs)

        # Find port by RS485 adapter serial number
        self._port = self._findDevice(port, serialnum)
        self._serial = self._openSerialPort(self._port)

        # Shift VA
        # Range depends on metadata and will be checked in ._write_registers
        self.shift = model.TupleContinuous((0, 0), range=((-1, -1), (1, 1)),
                                           cls=(int, float), unit="m",
                                           setter=self._setShift)

    def _findDevice(self, ports=None, serialnum=None):
        """
        Look for a compatible device. Requires at least one of the arguments ports and serialnum.
        ports (str): port (e.g. "/dev/ttyUSB0") or pattern for port ("/dev/ttyUSB*"), "/dev/fake" will start the simulator
        serialnum (str): serial number
        return (str): the name of the port used
        raises:
            ValueError: if no device on the ports with the given serial number is found
        """
        # At least one of the arguments ports and serialnum must be specified
        if not ports and not serialnum:
            raise ValueError("At least one of the arguments 'ports' and 'serialnum' must be specified.")

        # For debugging purpose
        if ports == "/dev/fake":
            return ports

        # If no ports specified, check all available ports
        if ports:
            names = list(serial.tools.list_ports.grep(ports))
        else:
            names = serial.tools.list_ports.comports()  # search all serial ports

        # Look for serial number if available, otherwise make sure only one port matches the port pattern.
        if serialnum:
            for port in names:
                if serialnum in port.description or serialnum in port.hwid:
                    # "RS485" is in port.description, .hwid presumably contains serial number, TODO: check this!
                    return port.device  # Found it!
            else:
                raise HwError("Beam controller device with serial number %s not found for port %s. " % (serialnum, names) +
                              "Check the connection.")
        else:
            if len(names) == 1:
                port = names[0]
                return port.device
            elif len(names) > 1:
                raise HwError("Multiple ports detected for beam controller. Please specify a serial number.")
            else:
                raise HwError("Beam controller device not found for port %s. Check the connection." % ports)

    def _openSerialPort(self, port):
        if self._port == "/dev/fake":
            return BeamShiftControllerSimulator()
        else:
            return ModbusSerialClient(method='rtu', port=port,
                                      baudrate=BAUDRATE, timeout=TIMEOUT,
                                      stopbits=STOPBITS, parity=PARITY,
                                      bytesize=BYTESIZE)

    def _setShift(self, value):
        """
        :param value (float, float): x, y shift from the center (in m)
        """
        gain = self._metadata.get(model.MD_CALIB_SCALE)
        if gain is None:
            gain = DEFAULT_GAIN
            logging.error("No gain specified, assuming default of %s." % gain +
                          "Calibration is required to set shift in a meaningful way.")

        offset = self._metadata.get(model.MD_CALIB_TRANSLATION)
        if offset is None:
            logging.warning("No deflection offset specified, assuming 0 amps.")
            offset = (0, 0)

        # It's the rotation between coordinate systems, so when transforming a move, we need the inverse!
        rotation = self._metadata.get(model.MD_CALIB_ROTATION)
        if rotation is None:
            logging.warning("No rotation specified, assuming 0 rad.")
            rotation = 0

        # Transform (rotation/translation)
        value = transform_coordinates(value, rotation, offset, RH, LH)

        # Convert (m-->amp-->int64), scales are different in x and y, this is why we can't directly
        # do it inside the previous function. The scaling has to be done in the dc shift coordinate system,
        # so in this case after the transformation.
        value = (value[0] * gain[0], value[1] * gain[1])
        register_values = self._amp_to_diff_int64(value)

        # Read previous value of registers for debugging purpose
        # Note on duration: a write instruction takes about 14 ms, a read instruction about 20 ms
        ret = self._read_registers()
        logging.debug("Register values before writing: %s." % ret)

        logging.debug("Writing register values %s" % register_values)
        self._write_registers(register_values)

        # Scale back to multiprobe units and convert back to multiprobe coordinates
        value = (value[0] / gain[0], value[1] / gain[1])
        value = transform_coordinates(value, -rotation, (-offset[0], -offset[1]), LH, RH)
        return value

    @staticmethod
    def _amp_to_diff_int64(value):
        """
        Convert value in amps to differential output signal (64-bit ints).
        :param value (float, float): x, y shift in amps
        :return (int, int, int, int): differential output (x low, y low, x high, y high)
        """
        value = numpy.array(value, dtype=numpy.float64)
        amp_range = numpy.array(RANGE_AMPS, dtype=numpy.float64)

        conversion_factor = 0xFFFF / (amp_range[1] - amp_range[0])
        valuePos = (value - amp_range[0]) * conversion_factor
        valueNeg = (-value - amp_range[0]) * conversion_factor
        register_values = [round(valueNeg[0]), round(valueNeg[1]), round(valuePos[0]), round(valuePos[1])]
        return [int(val) for val in register_values]  # round doesn't return int

    def _write_registers(self, values):
        """
        Write to all four registers. Try to reconnect to device in case connection was lost.
        :values (list of 4 ints): register values (-x, -y, x, y)
        """
        if len(values) != 4:
            raise ValueError("write_registers received payload of invalid length %s != 4." % len(values))

        # Check if values are in allowed range
        if not all(0 <= val <= 0xFFFF for val in values):
            raise ValueError("Register values %s not in range [0, 65535]." % values)

        try:
            # write all registers together (starting at lower x register (=0x01))
            rq = self._serial.write_registers(BEAMDEFL_LX, values, unit=SLAVE_UNIT)
            #if rq.function_code < 0x80:  # test that we are not an error
            #    raise model.HwError(rq.function_code)
        except IOError:
            self._reconnect()
            raise IOError("Failed to write registers of beam control firmware, "
                          "restarted serial connection.")

    def _read_registers(self):
        """
        Read all four registers. Try to reconnect to device in case connection was lost.
        :return (list of 4 ints): register values (-x, -y, x, y)
        """
        try:
            # write all registers together (starting at lower x register (=0x01))
            rr = self._serial.read_holding_registers(BEAMDEFL_LX, 4, unit=SLAVE_UNIT)
            return rr.registers
        except IOError:
            self._reconnect()
            raise IOError("Failed to write registers of beam control firmware, "
                          "restarted serial connection.")

    def _reconnect(self):
        """
        Attempt to reconnect the camera. It will block until this happens.
        On return, the hardware should be ready to use as before, excepted it
        still needs the settings to be applied.
        """
        num_it = 5
        self.state._set_value(model.HwError("Beam deflection controller disconnected"), force_write=True)
        logging.warning("Failed to write registers, trying to reconnect...")
        for i in range(num_it):
            try:
                self._serial.close()
                self._serial.connect()
                logging.info("Recovered device.")
                break
            except IOError:
                continue
        else:
            raise IOError("Failed to reconnect to beam deflection controller.")
        self.state._set_value(model.ST_RUNNING, force_write=True)

    def updateMetadata(self, md):
        logging.debug("Updating metadata %s." % md)
        prev_rotation = self._metadata.get(model.MD_CALIB_ROTATION)
        # Update .shift if rotation is changed (but don't set value in hardware)
        if model.MD_CALIB_ROTATION in md.keys():
            rotation = md[model.MD_CALIB_ROTATION] - prev_rotation if prev_rotation else md[model.MD_CALIB_ROTATION]
            shift = transform_coordinates(self.shift.value, rotation)
            logging.debug("Shift after metadata update: %s" % shift)
            self.shift._value = shift
            self.shift.notify(shift)
        model.HwComponent.updateMetadata(self, md)


class BeamShiftControllerSimulator(object):

    def __init__(self):
        self.r0 = 0
        self.r1 = 0
        self.r2 = 0
        self.r3 = 0

    def write_registers(self, start_register, values, unit=None):
        """
        Writes four values in the registers r0-r3.
        """
        self.r0 = values[0]
        self.r1 = values[1]
        self.r2 = values[2]
        self.r3 = values[3]
        return SimplifiedModbusObject([])

    def read_holding_registers(self, start_register, num_registers, unit=None):
        return SimplifiedModbusObject([self.r0, self.r1, self.r2, self.r3][:num_registers])


class SimplifiedModbusObject(object):
    """
    Simulate a modbus object (has .registers and .function_code attributes).
    """
    def __init__(self, registers):
        self.function_code = 0x80
        self.registers = registers
