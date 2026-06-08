import src.radar_constants as constants
from typing import Union
import logging
import math
import numpy as np

logging.basicConfig(level=logging.INFO)


# TODO: add docstrings and test cases.


def power_to_decibel(p: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    if isinstance(p, np.ndarray):
        return 10. * np.log10(p)  # power in watts.
    else:
        return 10. * math.log10(p)


def decibel_to_power(decibel: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    if isinstance(decibel, np.ndarray):
        return np.power(10. * np.ones_like(decibel), 0.1 * decibel)
    else:
        return math.pow(10., 0.1 * decibel)


def wavelength_or_frequency(speed_light: float, frequency: float) -> float:
    if frequency == 0.:
        dbz = "Cannot have frequency or wavelength equal to zero."
        logging.error(dbz)
        raise ZeroDivisionError(dbz)

    return speed_light / frequency


def monostatic_range_resolution(speed_light: float, bandwidth: float) -> float:
    if bandwidth == 0.:
        dbz = "Cannot have zero bandwidth."
        logging.error(dbz)
        raise ZeroDivisionError(dbz)
    return speed_light / (2. * bandwidth)


def monostatic_velocity_resolution(wavelength: float, pulse_width: float) -> float:
    if pulse_width == 0.:
        dbz = "Cannot divide by zero pulse_width."
        logging.error(dbz)
        raise ZeroDivisionError(dbz)
    return wavelength / (2. * pulse_width)


def time_bandwidth_product(pulse_width: float, bandwidth: float) -> float:
    return pulse_width * bandwidth


def calculate_search_volume(azimuth_angle: float, elevation_angle: float) -> float:
    """
    az,el in deg
    eqn 1.61 in Mahafza book
    """
    return azimuth_angle * elevation_angle / (57.296 ** 2)  # steradians


def calculate_power_aperture(snr: float, tsc: float,radar_cross_section: float, rho: float, noise_temp: float, nf: float, loss: float, az_angle: float, el_angle: float):
    """
    implements Listing 1.5. MATLAB Function power_aperture.
    % This program implements Eq. (1.67)
    """
    omega = calculate_search_volume(az_angle,el_angle) # compute search volume in steradians

    # implement Eq. (1.67)
    power_aperture: float = snr + power_to_decibel(4. * math.pi) + power_to_decibel(constants.BOLTZMANN_CONSTANT) + power_to_decibel(noise_temp) + nf + loss + power_to_decibel(rho **4) + power_to_decibel(omega) - power_to_decibel(radar_cross_section) - power_to_decibel(tsc)
    return power_aperture


def linear_chirp(start_frequency: float, end_frequency: float, time: float, sampling_frequency: float, amplitude: float) -> np.ndarray:
    if time == 0. or sampling_frequency == 0.:
        dbz = "Duration or sampling frequency cannot be zero."
        logging.error(dbz)
        raise ZeroDivisionError(dbz)
    end_freq_comp = (end_frequency - start_frequency) / (2 * time)
    dt = 1. / sampling_frequency  # set sampling interval
    time_array = np.arange(0, time, dt, dtype=np.float64)  # 0:dt:dur; % create vector of time samples
    psi = 2 * math.pi * (
                100. * np.ones_like(time_array) + start_frequency * time_array + end_freq_comp * time_array * time_array)  # set argument for chirp function
    xx = amplitude * np.cos(psi)  # modulate signal
    return xx.astype(complex)