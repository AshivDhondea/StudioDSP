import src.radar_constants as constants
from typing import Union
import logging
import math
import numpy as np

logging.basicConfig(level=logging.INFO)


def bistatic_received_power(power_tx: float, gain_tx: float, gain_rx: float, rho_rx: Union[float, np.ndarray], rho_tx: Union[float, np.ndarray],
                            wavelength: float, radar_cross_section: float) -> Union[float, np.ndarray]:
    if rho_rx == 0. or rho_tx == 0.:
        dbz = "Distance to receiver or transmitter cannot be zero."
        logging.error(dbz)
        raise ZeroDivisionError(dbz)

    denominator: float = (4 * math.pi) ** 3 * (rho_rx ** 2) * (rho_tx ** 2)
    numerator: float = power_tx * gain_tx * gain_rx * radar_cross_section * (wavelength ** 2)
    power_rx: float = numerator / denominator
    return power_rx


def bistatic_receiver_snr(power_rx: Union[float, np.ndarray], t0: float, bandwidth: float, radar_loss: float) -> float:
    if bandwidth == 0. or t0 == 0. or radar_loss == 0.:
        dbz = "Bandwidth or t0 or radar loss cannot be zero."
        logging.error(dbz)
        raise ZeroDivisionError(dbz)

    return power_rx / (constants.BOLTZMANN_CONSTANT * bandwidth * t0 * radar_loss)


def bistatic_pulse_width(power_tx: float, gain_tx: float, gain_rx: float, rho_rx: Union[float, np.ndarray], rho_tx: Union[float, np.ndarray],
                          wavelength: float, radar_cross_section: float, snr: float, t0: float, radar_loss: float) -> Union[float, np.ndarray]:
    # equation 1.57
    if any(x == 0 for x in (power_tx, gain_tx, gain_rx, radar_cross_section, wavelength)):
        dbz = "Power Tx, Gain Tx, Gain Rx, RCS or wavelength cannot be 0."
        logging.error(dbz)
        raise ZeroDivisionError(dbz)

    numerator = (4 * math.pi) ** 3 * (rho_rx ** 2) * (rho_tx ** 2) * snr * constants.BOLTZMANN_CONSTANT * t0 * radar_loss
    denominator = power_tx * gain_tx * gain_rx * radar_cross_section * (wavelength ** 2)
    pulse_width: float = numerator / denominator
    return pulse_width



