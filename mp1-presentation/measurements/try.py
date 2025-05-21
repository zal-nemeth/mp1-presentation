from moku.instruments import WaveformGenerator
import os
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,169.254.112.97/16'

# # Replace with your Moku:Go's IP address
# moku_ip = '169.254.112.97'

# # Connect to the Moku:Go device
# with WaveformGenerator(moku_ip) as wg:
#     # Configure Channel 1 to output a 300 Hz square wave with 50% duty cycle
#     wg.generate_waveform(
#         channel=1,
#         type='Square',
#         frequency=400,  # Frequency in Hz
#         amplitude=1.0,  # Amplitude in Vpp (peak-to-peak)
#         duty=50.0,      # Duty cycle in percentage
#         offset=0.0      # DC offset in V
#     )
#     # Enable the output on Channel 1
#     wg.set_output(channel=1, enabled=True)
#     print("Channel 1 is now outputting a 300 Hz square wave.")


from moku.instruments import WaveformGenerator


moku_ip = '169.254.112.97'  # Replace with your device's IP address

# Create the connection
wg = WaveformGenerator(moku_ip, force_connect=True)

# Your setup code for the waveform generator here
wg.generate_waveform(channel=2,
                     amplitude=5,
                     type='Square',
                     frequency=600,
                     duty=50)
# wg.
# wg.(channel=1, enabled=True)
print("Waveform generator set up successfully.")

