import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# -------------------------------
# 1. Define Vocabulary and Helpers
# -------------------------------
# Here we define a simple character set. In a real ASR system, you might need punctuation and more.
vocab = "abcdefghijklmnopqrstuvwxyz '"
char_to_num = {c: i for i, c in enumerate(vocab)}
num_to_char = {i: c for i, c in enumerate(vocab)}
num_classes = len(vocab)

def text_to_int(text):
    """Converts a text string to a list of integer indices."""
    text = text.lower()
    return [char_to_num[c] for c in text if c in char_to_num]

def int_to_text(ints):
    """Converts a list of integer indices back to a text string."""
    return ''.join([num_to_char[i] for i in ints if i in num_to_char])

# -------------------------------
# 2. Audio Preprocessing
# -------------------------------
def preprocess_audio(waveform, sample_rate, num_mel_bins=40):
    """
    Converts a raw audio waveform into a log–Mel spectrogram.
    Parameters:
      waveform: 1-D tensor of audio samples.
      sample_rate: Sampling rate of the audio.
      num_mel_bins: Number of Mel frequency bins.
    Returns:
      A tensor of shape (time, num_mel_bins).
    """
    # Compute the Short-Time Fourier Transform (STFT)
    stfts = tf.signal.stft(waveform, frame_length=256, frame_step=128)
    spectrogram = tf.abs(stfts)

    # Create a Mel filter bank matrix and apply it to the spectrogram.
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, spectrogram.shape[-1], sample_rate, 80.0, 7600.0)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    
    # Take logarithm to obtain log-Mel spectrogram
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    return log_mel_spectrogram

def load_audio(file_path):
    """Loads a WAV file and returns the waveform and sample rate."""
    audio_binary = tf.io.read_file(file_path)
    waveform, sample_rate = tf.audio.decode_wav(audio_binary)
    waveform = tf.squeeze(waveform, axis=-1)
    return waveform, sample_rate

# -------------------------------
# 3. Model Definition
# -------------------------------
def build_model(input_dim, num_classes):
    """
    Builds a simple CNN-RNN model for speech recognition.
    Parameters:
      input_dim: Number of features per time step (e.g., number of Mel bins).
      num_classes: Number of output classes (characters).
    Returns:
      A tf.keras.Model instance.
    """
    input_data = tf.keras.Input(shape=(None, input_dim))  # (time, features)
    # Add a channel dimension for convolution layers
    x = tf.keras.layers.Reshape((-1, input_dim, 1))(input_data)
    
    # Convolutional layers for feature extraction
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Reshape for RNN layers: combine spatial dims while preserving time
    # Here, we flatten the last two dimensions.
    x = tf.keras.layers.Reshape((-1, x.shape[-2]*x.shape[-1]))(x)
    
    # Bidirectional LSTM layers
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    
    # Output layer with softmax (note: +1 for the CTC blank token)
    y_pred = tf.keras.layers.Dense(num_classes + 1, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=input_data, outputs=y_pred)
    return model

# -------------------------------
# 4. Dataset Preparation Using LibriSpeech (Subset)
# -------------------------------
def prepare_dataset():
    """
    Loads a small subset of the LibriSpeech dataset and preprocesses each example.
    Each example is mapped to a tuple (log_mel_spectrogram, transcript_as_int).
    """
    # For demonstration we load 'train-clean-100'. Adjust as needed.
    ds, ds_info = tfds.load('librispeech', split='train-clean-100', with_info=True, as_supervised=True)
    
    def map_fn(audio, transcript):
        # Squeeze to remove extra dimensions if present.
        waveform = tf.squeeze(audio, axis=-1)
        sample_rate = 16000  # LibriSpeech audio is at 16kHz
        # Compute log–Mel spectrogram
        log_mel = preprocess_audio(waveform, sample_rate)
        # Convert transcript text to integer sequence
        # Note: .numpy() is used here within a tf.py_function.
        transcript_str = transcript.numpy().decode('utf-8')
        label = text_to_int(transcript_str)
        label = tf.convert_to_tensor(label, dtype=tf.int32)
        return log_mel, label

    def tf_map_fn(audio, transcript):
        log_mel, label = tf.py_function(map_fn, [audio, transcript], [tf.float32, tf.int32])
        log_mel.set_shape([None, 40])  # 40 Mel bins
        label.set_shape([None])
        return log_mel, label

    ds = ds.map(tf_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(8, padded_shapes=([None, 40], [None]), drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------------------
# 5. Define CTC Loss for Training
# -------------------------------
def ctc_loss(y_true, y_pred):
    """
    Computes the CTC (Connectionist Temporal Classification) loss.
    y_true: tensor of shape (batch, label_length)
    y_pred: tensor of shape (batch, time, num_classes+1)
    """
    batch_len = tf.cast(tf.shape(y_pred)[0], dtype=tf.int32)
    input_length = tf.fill([batch_len, 1], tf.shape(y_pred)[1])
    label_length = tf.fill([batch_len, 1], tf.shape(y_true)[1])
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# -------------------------------
# 6. Training the Model
# -------------------------------
def train_model():
    dataset = prepare_dataset()
    
    # Use one batch to determine the input feature dimensions.
    for batch in dataset.take(1):
        sample_features, sample_labels = batch
        break
    input_dim = sample_features.shape[-1]
    
    model = build_model(input_dim, num_classes)
    model.compile(optimizer='adam', loss=ctc_loss)
    
    # For demonstration, we train for just a few epochs.
    model.fit(dataset, epochs=5)
    
    # Save the trained model.
    model.save('asr_model.h5')
    return model

# -------------------------------
# 7. Inference Function
# -------------------------------
def transcribe_audio(file_path, model):
    """
    Given a 15-second (or any length) WAV audio file, this function:
      - Loads and preprocesses the audio.
      - Runs inference with the trained model.
      - Uses greedy CTC decoding to obtain the transcription.
    """
    waveform, sample_rate = load_audio(file_path)
    log_mel = preprocess_audio(waveform, sample_rate)
    # Add batch dimension: shape becomes (1, time, 40)
    log_mel = tf.expand_dims(log_mel, 0)
    y_pred = model.predict(log_mel)
    
    # Decoding: Use TensorFlow's built-in CTC decoder (greedy decoding here).
    input_len = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    decoded, log_prob = tf.keras.backend.ctc_decode(y_pred, input_length=input_len, greedy=True)
    
    # Convert the first decoded output (assumed batch size 1) from integers to text.
    transcription = int_to_text(tf.squeeze(decoded[0]).numpy().tolist())
    return transcription

# -------------------------------
# 8. Main Execution
# -------------------------------
if __name__ == "__main__":
    # Step 1: Train the model (this may take a while even on a subset).
    model = train_model()
    
    # Step 2: Perform inference on a 15-second audio file.
    # Replace 'sample_audio.wav' with the path to your WAV file.
    transcription = transcribe_audio('Recording.wav', model)
    print("Transcription:", transcription)
