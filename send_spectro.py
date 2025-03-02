import subprocess
import os
import numpy as np

# Path to the Nios II Command Shell (configurable via environment variable)
NIOS_CMD_SHELL_BAT = os.getenv("NIOS_CMD_SHELL_PATH", "C:/intelFPGA_lite/18.1/nios2eds/Nios II Command Shell.bat")

def send_on_jtag(cmd):
    """
    Sends a command to the Nios II processor via JTAG.
    """
    # Validate the command
    if not cmd or not isinstance(cmd, str):
        raise ValueError("Command must be a non-empty string.")

    # Check if the batch file exists
    if not os.path.exists(NIOS_CMD_SHELL_BAT):
        raise FileNotFoundError(f"Nios II Command Shell not found at {NIOS_CMD_SHELL_BAT}")

    # Create a subprocess to run the Nios II Command Shell
    process = subprocess.Popen(
        NIOS_CMD_SHELL_BAT,
        bufsize=0,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Send the command to the Nios II terminal
    try:
        input_data = f"nios2-terminal\n{cmd}\n".encode("utf-8")
        vals, err = process.communicate(input=input_data, timeout=10)
        process.terminate()
    except subprocess.TimeoutExpired:
        process.terminate()
        raise RuntimeError("Command execution timed out.")
    except Exception as e:
        process.terminate()
        raise RuntimeError(f"An error occurred: {e}")

    # Decode the output
    if vals:
        vals = vals.decode("utf-8")
    return vals

def prepare_spectrogram_data(spectrogram):
    """
    Prepares the spectrogram data for transmission.
    - Flattens the 2D array.
    - Normalizes and quantizes the data to 8-bit integers.
    - Serializes the data into a byte stream.
    """
    # Flatten the spectrogram
    flattened_data = spectrogram.flatten()

    # Normalize and quantize to 8-bit integers
    max_value = np.max(flattened_data)
    normalized_data = (flattened_data / max_value) * 255
    quantized_data = np.round(normalized_data).astype(np.uint8)

    # Serialize into bytes
    byte_data = quantized_data.tobytes()
    return byte_data

def send_spectrogram(spectrogram):
    """
    Sends the spectrogram data to the Nios II processor.
    """
    # Prepare the spectrogram data
    byte_data = prepare_spectrogram_data(spectrogram)

    # Convert the byte data to a hex string for transmission
    hex_data = byte_data.hex()

    # Send the data to the Nios II processor
    response = send_on_jtag(hex_data)
    print("Nios II Response:", response)

def main():
    # Example spectrogram (2D array)
    spectrogram = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])

    # Send the spectrogram to the Nios II processor
    send_spectrogram(spectrogram)

if __name__ == "__main__":
    main()