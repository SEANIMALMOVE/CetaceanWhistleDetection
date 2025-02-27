import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from sklearn.model_selection import train_test_split

def load_audio_data_from_csv(csv_path):
    """
    Load audio data and labels from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing audio file paths and labels.

    Returns:
        list: Paths to audio files.
        list: Corresponding labels.
    """
    data = pd.read_csv(csv_path)
    audio_files = data['file_path'].tolist()
    labels = data['label'].tolist()
    
    return audio_files, labels

def extract_features(file_paths, sr=22050):
    """
    Extract audio features using Librosa.
    
    Args:
        file_paths (list): List of audio file paths.
        sr (int): Sampling rate for Librosa.

    Returns:
        np.array: Extracted features.
    """
    features = []
    
    for file in file_paths:
        y, _ = librosa.load(file, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.append(np.mean(mfcc.T, axis=0))

    return np.array(features)

def preprocess_data(csv_path, test_size=0.2, random_state=42):
    """
    Preprocess the dataset by extracting features and splitting into train/test sets.
    
    Args:
        csv_path (str): Path to the CSV file containing audio file paths and labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.

    Returns:
        tuple: Train and test splits (X_train, X_test, y_train, y_test).
    """
    audio_files, labels = load_audio_data_from_csv(csv_path)
    
    # Convert labels to numeric categories
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_index[label] for label in labels]

    # Extract features
    features = extract_features(audio_files)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, numeric_labels, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, unique_labels

if __name__ == "__main__":
    CSV_PATH = "../../Dataset/audio_metadata.csv"  # Update with your CSV path
    X_train, X_test, y_train, y_test, labels = preprocess_data(CSV_PATH)

    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(labels)}")
