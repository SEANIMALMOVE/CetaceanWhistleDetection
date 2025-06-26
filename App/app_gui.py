import os
import json
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import time
import sys

from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QLineEdit, QFileDialog, QCheckBox, QProgressBar, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Configuration file to store last used paths and option
CONFIG_FILE = 'config.json'

def save_config(audio_path, output_folder, model_folder, load_previous):
    config = {
        'audio_path': audio_path,
        'output_folder': output_folder,
        'model_folder': model_folder,
        'load_previous': load_previous
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

# Define class mappings
CLASSES_MAPPING = {
    'Background': 0,
    'Whistle': 1,
}
INV_CLASSES_MAPPING = {v: k for k, v in CLASSES_MAPPING.items()}

# Class to recursively search for .wav files
class AudioDataset:
    def __init__(self, audio_path, sample_rate=48000):
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.files = []
        self._process_audio()

    def _process_audio(self):
        if os.path.isfile(self.audio_path):
            self.files = [self.audio_path]
        elif os.path.isdir(self.audio_path):
            self.files = []
            for root, _, files in os.walk(self.audio_path):
                for f in files:
                    if f.lower().endswith('.wav'):
                        self.files.append(os.path.join(root, f))
            print(f"Number of audio files to be processed: {len(self.files)}")
        else:
            raise ValueError("Invalid path. Provide a valid file or folder containing audio files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]

# Function to load the model from a given folder
def load_model_keras(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to split an audio file into 3-second segments (with padding if needed)
def process_audio_segments(file, sample_rate=48000, segment_length=3):
    segments = []
    signal, sr = librosa.load(file, sr=sample_rate, mono=True)
    max_length = sample_rate * segment_length
    if len(signal) <= max_length:
        padding = max_length - len(signal)
        pad_left = padding // 2
        pad_right = padding - pad_left
        signal = np.pad(signal, (pad_left, pad_right), mode='constant')
        segments.append((file, signal, 0, segment_length))
    elif len(signal) > max_length:
        for start in range(0, len(signal), max_length):
            end = start + max_length
            segment = signal[start:end]
            if len(segment) < max_length:
                segment = np.pad(segment, (0, max_length - len(segment)), mode='constant')
            segments.append((file, segment, start // sample_rate, end // sample_rate))
    else:
        segments.append((file, signal, 0, segment_length))
    print(f"Number of segments: {len(segments)} in {len(signal)/sample_rate} seconds")
    return segments

# Function to classify a single segment
def classify_audio(model, audio_clip):
    audio_clip = np.expand_dims(audio_clip, axis=0)
    probabilities = model.predict(audio_clip, verbose=0)[0]
    top_class = np.argmax(probabilities)
    return top_class, probabilities

# Worker thread that runs the classification process
class Worker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  # current, total
    finished_signal = pyqtSignal(pd.DataFrame)  # emits combined dataframe when done

    def __init__(self, audio_path, output_folder, model_folder, load_previous):
        super().__init__()
        self.audio_path = audio_path
        self.output_folder = output_folder
        self.model_folder = model_folder
        self.load_previous = load_previous

    def run(self):
        self.log_signal.emit("Loading model...")
        try:
            model = load_model_keras(self.model_folder)
        except Exception as e:
            self.log_signal.emit("Error loading model: " + str(e))
            return

        dataset = AudioDataset(self.audio_path)
        all_results = []
        total_preprocess_time = 0
        total_inference_time = 0
        total_audio_duration = 0

        df_combined = pd.DataFrame()
        if self.load_previous:
            existing_files = [f for f in os.listdir(self.output_folder) if f.endswith('.csv')]
            for f in existing_files:
                try:
                    df_temp = pd.read_csv(os.path.join(self.output_folder, f))
                    df_combined = pd.concat([df_combined, df_temp], ignore_index=True)
                    self.log_signal.emit(f"Loaded {f} with {len(df_temp)} rows")
                except Exception as e:
                    self.log_signal.emit(f"Error loading {f}: {e}")
        else:
            self.log_signal.emit("Ignoring previously processed CSV files; starting fresh.")

        total_files = len(dataset.files)
        self.progress_signal.emit(0, total_files)
        for i, file in enumerate(dataset.files):
            base_csv = os.path.splitext(os.path.basename(file))[0] + ".csv"
            if self.load_previous and base_csv in os.listdir(self.output_folder):
                self.log_signal.emit(f"Skipping {file} (already processed)")
                self.progress_signal.emit(i+1, total_files)
                continue
            self.log_signal.emit(f"Processing {file} ({i+1}/{total_files})")
            start_preprocess = time.time()
            try:
                segments = process_audio_segments(file)
            except Exception as e:
                self.log_signal.emit(f"Error processing {file}: {e}")
                continue
            end_preprocess = time.time()
            preprocess_time = end_preprocess - start_preprocess
            total_preprocess_time += preprocess_time

            start_inference = time.time()
            file_results = []
            for original_file, segment, start_s, end_s in segments:
                try:
                    top_class, probabilities = classify_audio(model, segment)
                    result = {
                        "Path": original_file,
                        "Filename": os.path.basename(file),
                        "StartSecond": start_s,
                        "EndSecond": end_s,
                        "MainClassification": INV_CLASSES_MAPPING[top_class],
                        "ConfidenceScore": probabilities[top_class],
                        "ConfidenceVector": probabilities.tolist()
                    }
                    all_results.append(result)
                    file_results.append(result)
                except Exception as e:
                    self.log_signal.emit(f"Error classifying segment in {file}: {e}")
            end_inference = time.time()
            inference_time = end_inference - start_inference
            total_inference_time += inference_time
            total_audio_duration += len(segments) * 3  # each segment is 3 seconds

            self.log_signal.emit(f"Preprocessing time for {file}: {preprocess_time:.2f} sec")
            self.log_signal.emit(f"Inference time for {file}: {inference_time:.2f} sec")

            try:
                df_file = pd.DataFrame(file_results)
                output_csv = os.path.join(self.output_folder, base_csv)
                df_file.to_csv(output_csv, index=False)
                self.log_signal.emit(f"Saved predictions for {file} to {output_csv}")
            except Exception as e:
                self.log_signal.emit(f"Error saving CSV for {file}: {e}")

            self.progress_signal.emit(i+1, total_files)

        try:
            df_all_results = pd.DataFrame(all_results)
            df_combined = pd.concat([df_combined, df_all_results], ignore_index=True)
            combined_csv = os.path.join(self.output_folder, "predictions.csv")
            df_combined.to_csv(combined_csv, index=False)
            self.log_signal.emit(f"Combined results saved to {combined_csv}")

            # Save CSV with predictions not labeled as Background
            df_non_background = df_combined[df_combined['MainClassification'] != 'Background']
            non_background_csv = os.path.join(self.output_folder, "non_background_predictions.csv")
            df_non_background.to_csv(non_background_csv, index=False)
            self.log_signal.emit(f"Non-Background predictions saved to {non_background_csv}")
        except Exception as e:
            self.log_signal.emit("Error combining results: " + str(e))

        self.log_signal.emit("\n\n")

        avg_preprocess_time = total_preprocess_time / total_files
        avg_inference_time = total_inference_time / total_files
        avg_total_time = (total_preprocess_time + total_inference_time) / total_files

        self.log_signal.emit(f"Total time for preprocessing: {total_preprocess_time:.2f} seconds")
        self.log_signal.emit(f"Total time for inference: {total_inference_time:.2f} seconds")
        self.log_signal.emit(f"Total time for preprocessing + inference: {total_preprocess_time + total_inference_time:.2f} seconds")

        self.log_signal.emit(f"Average time per file for preprocessing: {avg_preprocess_time:.2f} seconds")
        self.log_signal.emit(f"Average time per file for inference: {avg_inference_time:.2f} seconds")
        self.log_signal.emit(f"Average time per file for preprocessing + inference: {avg_total_time:.2f} seconds")


        self.log_signal.emit(f"Total files processed: {total_files}")
        self.log_signal.emit(f"Total audio duration: {total_audio_duration/3600:.0f}:{(total_audio_duration%3600)/60:.0f}:{total_audio_duration%60:.0f}  ({total_audio_duration} seconds)")

        self.log_signal.emit("\n\n")

        self.log_signal.emit(f"Whistles detected: {len(df_non_background)}")

        self.finished_signal.emit(df_combined)

# Main window using PyQt5 for a modern look
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beautiful Audio Classifier")
        self.resize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Input fields and buttons
        input_layout = QVBoxLayout()
        self.audio_path_edit = QLineEdit()
        self.output_folder_edit = QLineEdit()
        self.model_folder_edit = QLineEdit()

        config = load_config()
        self.audio_path_edit.setText(config.get('audio_path', ''))
        self.output_folder_edit.setText(config.get('output_folder', ''))
        self.model_folder_edit.setText(config.get('model_folder', ''))

        input_layout.addWidget(QLabel("Audio Path:"))
        h_audio = QHBoxLayout()
        h_audio.addWidget(self.audio_path_edit)
        btn_audio = QPushButton("Browse")
        btn_audio.clicked.connect(self.browse_audio)
        h_audio.addWidget(btn_audio)
        input_layout.addLayout(h_audio)

        input_layout.addWidget(QLabel("Output Folder:"))
        h_output = QHBoxLayout()
        h_output.addWidget(self.output_folder_edit)
        btn_output = QPushButton("Browse")
        btn_output.clicked.connect(self.browse_output)
        h_output.addWidget(btn_output)
        input_layout.addLayout(h_output)

        input_layout.addWidget(QLabel("Model Folder:"))
        h_model = QHBoxLayout()
        h_model.addWidget(self.model_folder_edit)
        btn_model = QPushButton("Browse")
        btn_model.clicked.connect(self.browse_model)
        h_model.addWidget(btn_model)
        input_layout.addLayout(h_model)

        self.load_prev_checkbox = QCheckBox("Load previous predictions (if any)")
        self.load_prev_checkbox.setChecked(config.get('load_previous', True))
        input_layout.addWidget(self.load_prev_checkbox)

        layout.addLayout(input_layout)

        self.run_button = QPushButton("Run Classification")
        self.run_button.clicked.connect(self.run_classification)
        layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Matplotlib canvas for plotting
        self.figure, self.ax = plt.subplots(figsize=(6,4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def browse_audio(self):
        path = QFileDialog.getExistingDirectory(self, "Select Audio Folder")
        if path:
            self.audio_path_edit.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_folder_edit.setText(path)

    def browse_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if path:
            self.model_folder_edit.setText(path)

    def log(self, message):
        self.log_text.append(message)
        print(message)

    def run_classification(self):
        audio_path = self.audio_path_edit.text()
        output_folder = self.output_folder_edit.text()
        model_folder = self.model_folder_edit.text()
        load_previous = self.load_prev_checkbox.isChecked()

        if not audio_path or not output_folder or not model_folder:
            self.log("Error: Please select all paths.")
            return

        save_config(audio_path, output_folder, model_folder, load_previous)
        self.run_button.setEnabled(False)

        self.worker = Worker(audio_path, output_folder, model_folder, load_previous)
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_finished(self, df_combined):
        self.log("Processing finished.")
        self.ax.clear()
        if not df_combined.empty and 'MainClassification' in df_combined.columns:
            counts = df_combined['MainClassification'].value_counts()
            counts.plot(kind='bar', ax=self.ax)
            self.ax.set_title('Classification Counts')
            self.ax.set_ylabel('Count')
        self.canvas.draw()
        self.run_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
