import serial
import time

class DuoMAG:
    def __init__(self, com_port):
        """
        Initializes a serial connection to the DuoMAG device.

        Args:
            com_port (str): The COM port to connect to (e.g., 'COM3').
        """
        self._comPort = com_port
        self._serialPort = self.Open()

    def Open(self):
        """
        Opens a USB serial connection with the DuoMAG device.

        Returns:
            serial.Serial: The serial port object.
        """
        try:
            ser = serial.Serial(self._comPort)
        except serial.SerialException:
            raise Exception(f"Cannot open the serial port: {self._comPort}")

        ser.baudrate = 1000000
        ser.stopbits = serial.STOPBITS_TWO
        ser.timeout = 1

        if not ser.is_open:
            ser.open()

        return ser

    def Close(self):
        """
        Closes the serial connection if it's open.
        """
        if self._serialPort and self._serialPort.is_open:
            self._serialPort.close()

    def __del__(self):
        """
        Destructor to ensure the serial connection is closed when the object is deleted.
        """
        self.Close()

    def Pulse(self, intensity=None, pulse=None):
        """
        Sets the intensity and/or requests a pulse from the DuoMAG device.

        Args:
            intensity (int, optional): Intensity value between 0-100.
            pulse (bool, optional): Set to True to request a pulse.

        Raises:
            ValueError: If intensity is not within the valid range or pulse is not 1.
        """
        if intensity is None and pulse is None:
            self._RequestPulse()
            return

        if intensity is not None:
            if isinstance(intensity, (int, float)) and 0 <= intensity <= 100:
                self._SendCommand([int(intensity), int(intensity)])
            else:
                raise ValueError("The input argument INTENSITY must be a scalar between 0-100.")

        if isinstance(pulse, bool):
            if pulse:
                self._RequestPulse()
        elif pulse is not None:
            raise ValueError("Set input argument pulse to true to request a pulse.")

    def Recharge(self, steps, stepSize):
        """
        Sets the recharge delay for the DuoMAG device.

        Args:
            steps (int): Scalar value between 0-127 representing the number of steps.
            stepSize (float): Value in milliseconds used to multiply steps to the desired delay value.

        Raises:
            ValueError: If steps is not between 0-127 or stepSize is not a supported value.
        """
        # Supported step sizes in ms
        supported_step_sizes = {
            0.05: '000',
            0.10: '001',
            0.20: '010',
            0.50: '011',
            1.00: '100',
            2.00: '101',
            5.00: '110',
            10.00: '111'
        }

        if not (0 <= steps <= 127):
            raise ValueError("Steps must be a scalar value between 0-127.")

        if stepSize not in supported_step_sizes:
            raise ValueError("Not a supported step size value. See documentation for supported values.")

        # Get the corresponding binary code for the step size
        rechargeExp = supported_step_sizes[stepSize]

        # Convert fraction value to binary
        frc = format(steps, '07b')  # 7-bit binary representation for steps

        # Add the fraction value to upper/lower bits
        upper = int('100' + frc[:5], 2)  # upper bits take bits 1-5
        lower = int('101' + rechargeExp + frc[5:], 2)  # lower bits take bits 6-7

        # Send commands to the device
        self._SendCommand([upper, upper])  # Set upper bits

        self._SendCommand([lower, lower])  # Set lower bits

    def TTL(self, steps, stepSize):
        """
        Sets the TTL OUT delay for the DuoMAG device.

        Args:
            steps (int): Scalar value between 0-127 representing the number of steps.
            stepSize (float): Value in milliseconds used to multiply steps to the desired delay value.

        Raises:
            ValueError: If steps is not between 0-127 or stepSize is not a supported value.
        """
        # Supported step sizes in ms
        supported_step_sizes = {
            0.05: '000',
            0.10: '001',
            0.20: '010',
            0.50: '011',
            1.00: '100',
            2.00: '101',
            5.00: '110',
            10.00: '111'
        }

        if not (0 <= steps <= 127):
            raise ValueError("Steps must be a scalar value between 0-127.")

        if stepSize not in supported_step_sizes:
            raise ValueError("Not a supported step size value. See documentation for supported values.")
        if not self._serialPort.is_open:
            raise Exception("Serial connection is not open.")

        exp = ["000", "001", "010", "011", "100", "101", "110", "111"]

        vals = [00.05, 00.10, 00.20, 00.50, 01.00, 02.00, 05.00, 10.00]

        # Check the exp input for a match
        if stepSize in vals:
            # Get the corresponding binary code
            stepSize_index = vals.index(stepSize)
            rechargeExp = exp[stepSize_index]

            # Convert fraction value to binary
            # Fraction code (blank to start); frc = '0000000'
            frc = format(steps, '07b')  # for 7-bit array

            # Add the fraction value to upper bits
            upper = int('110' + frc[:5], 2)  # upper bits take bits 1-5

            # Add the exp value to the lower bits
            lower = int('111' + rechargeExp + frc[5:7], 2)  # lower bits take bits 6-7

            # Send commands to the device
            self._SendCommand([upper, upper, lower, lower])  # Set bits

    def _SendCommand(self, data):
        """
        Sends a command to the DuoMAG device.

        Args:
            data (list): A list of integers to send as a command.
        """

        if self._serialPort.is_open:
            byteArray = bytearray(data)
            self._serialPort.write(byteArray)
            time.sleep(0.05)
        else:
            raise Exception("Serial connection is not open.")

    def _RequestPulse(self):
        """
        Sends a pulse command to the DuoMAG device.
        """
        while self.ReadStatus()["isCharged"] != 1:
            time.sleep(0.01)
        self._SendCommand([121, 121])

    def ReadStatus(self):
        """
        Reads the current status of the DuoMAG device.

        Returns:
            dict: A dictionary containing the current status of the device.
        """
        if not self._serialPort.is_open:
            raise Exception("Serial connection is not open.")

        nchar = 32  # bytes to read
        # Read the data stream from the device

        self._serialPort.close()
        self._serialPort.open()

        # Read in data stream
        nchar = 32  # bytes to read
        iRaw = self._serialPort.read(nchar)

        # Convert to binary to allow us to read individual bits
        iBin = ''.join(format(byte, '08b') for byte in iRaw)

        # Find index for sync bytes (i.e 255)
        getSync = [index for index, value in enumerate(iRaw) if value == 255]

        # Read input between the most recent sync indexes
        iBin = iBin[getSync[-2] * 8 + 8: getSync[-1] * 8]

        # Calculate change intensity
        changeIntensity = iBin[3:8]  # get the binary code as character array
        changeIntensity = DuoMAG.sbin2dec(changeIntensity)  # use sbin2dec to convert to signed integer

        # Match data into corresponding titles with a dictionary
        DuoSTATUS = {}
        DuoSTATUS['demandOnStimTTL'] = int(iBin[2])
        DuoSTATUS['changeIntensity'] = changeIntensity
        DuoSTATUS['currentIntentity'] = int(''.join(iBin[8:16]), 2)
        DuoSTATUS['coilTemperatureADC'] = int(''.join(iBin[16:24]), 2)  # ADC output; units not specified
        DuoSTATUS['resistorTemperatureADC'] = int(''.join(iBin[24:32]), 2)  # ADC output; units not specified
        DuoSTATUS['dischargeTemperatureADC'] = int(''.join(iBin[32:40]), 2)  # ADC output; units not specified
        DuoSTATUS['omittedPulseCount'] = int(''.join(iBin[40:48]), 2)
        DuoSTATUS['isPowerOverheat'] = int(iBin[49])
        DuoSTATUS['isCoilDisconnected'] = int(iBin[50])
        DuoSTATUS['isVoltageHoldError'] = int(iBin[51])
        DuoSTATUS['isVoltageRechargeError'] = int(iBin[52])
        DuoSTATUS['isPowerFanSeized'] = int(iBin[57])
        DuoSTATUS['isCoilOverheat'] = int(iBin[59])
        DuoSTATUS['isResistorOverheat'] = int(iBin[60])
        DuoSTATUS['isDischargeOverheat'] = int(iBin[61])
        DuoSTATUS['isCoilDataOK'] = int(iBin[62])
        DuoSTATUS['isReceivingUSB'] = int(iBin[63])
        DuoSTATUS['isIdleTime'] = int(iBin[69])
        DuoSTATUS['isCharged'] = int(iBin[71])

        # Exponent values (from Deymed manual)
        vals = [00.05, 00.10, 00.20, 00.50, 01.00, 02.00, 05.00, 10.00]  # in ms

        # Exponent codes (from Deymed manual)
        exp = ["000", "001", "010", "011", "100", "101", "110", "111"]

        # Match exponent with equivalent value for recharge delay
        rechargeExp = sum(exp[i] == iBin[89:92] for i in range(len(exp)))
        rechargeExpVal = vals[rechargeExp]

        # Match exponent with equivalent value for ISI delay
        TTLExp = sum(exp[i] == iBin[93:96] for i in range(len(exp)))
        TTLExpVal = vals[TTLExp]

        # Calculate recharge and ISI delays in ms
        DuoSTATUS['delayTTL'] = int(int(''.join(iBin[73:80]), 2) * TTLExpVal * 10)
        DuoSTATUS['delayRecharge'] = int(int(''.join(iBin[81:88]), 2) * rechargeExpVal * 10)

        # Add coil ID if newer Deymed model
        if len(iBin) > 103:
            typeCoil = int(''.join(iBin[112:120]), 2)

            # Swap in coil identity if known
            # Coil type is saved as an integer, which corresponds to one of the coils
            # below. Array is ordered as in Deymed manual.
            # Known coil types:
            coilsKnown = ['70BF', '70BF-COOL', '120BFV', '100R', '125R', '50BF', '50BFT']

            # Check the coil number corresponds to one of the coils above
            if typeCoil > 0 and typeCoil <= len(coilsKnown):
                DuoSTATUS['typeCoil'] = coilsKnown[typeCoil - 1]
            else:
                DuoSTATUS['typeCoil'] = "Unknown"

        return DuoSTATUS

    @staticmethod
    # To read change in intensity parameter, which is in two's complement (i.e signed binary)
    def sbin2dec(x):
        # The leftmost digit determines if the binary string is signed or not
        if x[0] == '0':
            y = int(x[1:], 2)
        else:
            # The leftmost digit is '1'
            y = int(x[1:], 2) - 2 ** (len(x) - 1)

        return y