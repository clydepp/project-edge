import wave
import numpy as np
from serial import Serial
from serial.tools import list_ports
import os
import sys

# Find available COM ports
available_ports = [port.device for port in list_ports.comports()]
if not available_ports:
    print("No COM ports available. Please check your connections.")
    sys.exit(1)

# Print available ports
print("Available COM ports:")
for i, port in enumerate(available_ports):
    print(f"{i+1}. {port}")

# Let user select a port
port_idx = int(input(f"Select COM port (1-{len(available_ports)}): ")) - 1
SERIAL_PORT = available_ports[port_idx]
BAUD_RATE = 115200
CHUNK_SIZE = 256
SAMPLE_RATE = 8000

# Check if wav file exists and let user provide path
wav_path = input("Enter path to WAV file: ")
if not os.path.exists(wav_path):
    print(f"File not found: {wav_path}")
    sys.exit(1)

try:
    # Open the wav file
    wav_file = wave.open(wav_path, 'rb')
    sample_width = wav_file.getsampwidth()
    num_channels = wav_file.getnchannels()
    
    # Open serial connection
    try:
        ser = Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT}")
        
        # Read and send chunks of audio data
        try:
            while True:
                frames = wav_file.readframes(CHUNK_SIZE)
                if not frames:
                    break
                audio_data = np.frombuffer(frames, dtype=np.int16)
                for sample in audio_data:
                    ser.write(sample.to_bytes(2, byteorder='big', signed=True))
                ser.write(b'\xFF\xFF')
                
            print("Audio data sent successfully")
        except Exception as e:
            print(f"Error sending data: {e}")
        finally:
            ser.close()
    except Exception as e:
        print(f"Error opening serial port: {e}")
finally:
    wav_file.close()