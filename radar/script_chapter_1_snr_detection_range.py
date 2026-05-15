"""
implements Listing 1.2. MATLAB Program “fig1_12.m”
in Mahafza radar book
"""
import src.bistatic_radar as bistatic_eq
import src.radar_constants as constants
import src.radar_equations as radar_eq
import logging
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(level=logging.INFO)



if __name__ == '__main__':
    # Set the radar parameters
    # Dev note: in a production scenario, these magic variables have to be passed as parameters.
    # Radar parameters from book
    power_tx = 1.5e6  # [W]
    centre_frequency = 5.6e9  # [Hz]
    gain_tx_db = 45. # [dB]
    gain_tx = radar_eq.decibel_to_power(gain_tx_db)
    gain_rx = gain_tx
    radar_cross_section = 0.1  # [m^2]
    bandwidth = 5e6  # [Hz]
    te = 290.  # [K]
    nf = 3  # [dB]
    t0 = radar_eq.decibel_to_power(nf) * te
    radar_loss = radar_eq.decibel_to_power(6.)

    wavelength = radar_eq.wavelength_or_frequency(constants.SPEED_OF_LIGHT, centre_frequency)
    rho_tx = np.linspace(25e3, 165e3, 1000)  # target range 25 -165 km, 1000 points

    p_rx1 = np.zeros([np.shape(rho_tx)[0]], dtype=np.float64)
    p_rx2 = np.zeros([np.shape(rho_tx)[0]], dtype=np.float64)
    p_rx3 = np.zeros([np.shape(rho_tx)[0]], dtype=np.float64)
    #snr_rx_1 = np.zeros([np.shape(rho_tx)[0]], dtype=np.float64)
    #snr_rx_2 = np.zeros([np.shape(rho_tx)[0]], dtype=np.float64)
    #snr_rx_3 = np.zeros([np.shape(rho_tx)[0]], dtype=np.float64)

    #snr_rx_2_04 = np.zeros([np.shape(rho_tx)[0]], dtype=np.float64)
    #snr_rx_3_18 = np.zeros([np.shape(rho_tx)[0]], dtype=np.float64)

    for i in range(len(rho_tx)):
        p_rx1[i] = bistatic_eq.bistatic_received_power(power_tx, gain_tx, gain_rx, rho_tx[i], rho_tx[i],
                            wavelength, radar_cross_section)
        p_rx2[i] = bistatic_eq.bistatic_received_power(power_tx, gain_tx, gain_rx, rho_tx[i], rho_tx[i],
                                                       wavelength, radar_cross_section / 10.)
        p_rx3[i] = bistatic_eq.bistatic_received_power(power_tx, gain_tx, gain_rx, rho_tx[i], rho_tx[i],
                                                       wavelength, radar_cross_section * 10.)

    snr_rx_1 = bistatic_eq.bistatic_receiver_snr(p_rx1, t0, bandwidth, radar_loss)
    snr_rx_2 = bistatic_eq.bistatic_receiver_snr(p_rx2, t0, bandwidth, radar_loss)
    snr_rx_3 = bistatic_eq.bistatic_receiver_snr(p_rx3, t0, bandwidth, radar_loss)

    snr_rx_2_04 = bistatic_eq.bistatic_receiver_snr(p_rx1 * 0.4, t0, bandwidth, radar_loss)
    snr_rx_3_18 = bistatic_eq.bistatic_receiver_snr(p_rx1 * 1.8, t0, bandwidth, radar_loss)

    snr_rx_1_db = radar_eq.power_to_decibel(snr_rx_1)
    snr_rx_2_db = radar_eq.power_to_decibel(snr_rx_2)
    snr_rx_3_db = radar_eq.power_to_decibel(snr_rx_3)

    rcs1 = radar_eq.power_to_decibel(radar_cross_section)
    rcs2 = radar_eq.power_to_decibel(radar_cross_section / 10.)
    rcs3 = radar_eq.power_to_decibel(radar_cross_section * 10.)

    snr_rx_2_04_db = radar_eq.power_to_decibel(snr_rx_2_04)
    snr_rx_3_18_db = radar_eq.power_to_decibel(snr_rx_3_18)

    fig = plt.figure(1)
    ax = fig.gca()
    fig.suptitle("SNR versus detection range for three different values of RCS", fontsize=12)
    plt.plot(rho_tx / 1000., snr_rx_3_db, label=r"$\sigma = %f~\mathrm{dBsm}$" % rcs3)
    plt.plot(rho_tx / 1000., snr_rx_1_db, linestyle='-.', label=r"$\sigma = %f~\mathrm{dBsm}$" % rcs1)
    plt.plot(rho_tx / 1000., snr_rx_2_db, linestyle='--', label=r"$\sigma = %f~\mathrm{dBsm}$" % rcs2)
    ax.set_ylabel("SNR [dB]")
    ax.set_xlabel('Detection range [km]')
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle=(0, [0.7, 0.7]), lw=0.4, color='black')
    fig.savefig('script_chapter01_snr_detection_range_12a.pdf', bbox_inches='tight', pad_inches=0.11, dpi=10)

    fig = plt.figure(2);    ax = fig.gca()
    fig.suptitle("SNR versus detection range for three different values of radar peak power", fontsize=12)
    plt.plot(rho_tx / 1000., snr_rx_3_18_db, label=r"$P_\text{Tx} = 2.16~\mathrm{MW}$")
    plt.plot(rho_tx / 1000., snr_rx_1_db, linestyle='-.', label=r"$P_\text{Tx} = 1.5~\mathrm{MW}$")
    plt.plot(rho_tx / 1000., snr_rx_2_04_db, linestyle='--', label=r"$P_\text{Tx} = 0.6~\mathrm{MW}$")
    ax.set_ylabel(r"SNR $[\mathrm{dB}]$")
    ax.set_xlabel(r'Detection range $[\mathrm{km}]$')
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle=(0, [0.7, 0.7]), lw=0.4, color='black')
    fig.savefig('script_chapter01_snr_detection_range_12b.pdf', bbox_inches='tight', pad_inches=0.11, dpi=10)