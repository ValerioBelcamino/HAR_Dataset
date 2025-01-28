from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np


class SequenceDatasetNPY(Dataset):
    def __init__(self, sequences, labels, features, active_sensors, max_len = 150):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
        self.features = features
        self.active_sensors = active_sensors
        self.features = ['acceleration', 'gyroscope', 'magnetometer']
        self.length = -1

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.pad_sequence(self.sequences[idx]), self.labels[idx], self.lengths
    
    def pad_sequence(self, sequence_name):
        sequence_files = np.load(sequence_name)   

        feature_files = []
        for feat in self.features:
            temp = sequence_files[feat]
            feature_files.append(np.hstack([temp[:, i:i + 3] for i in self.active_sensors if i + 2 < temp.shape[1]]))
        sequence = np.hstack(feature_files)

        self.lengths = sequence.shape[0]


        if sequence.shape[0] < self.max_len:
            zeros = np.zeros((self.max_len, sequence.shape[1]), dtype=np.float32)
            zeros[:sequence.shape[0], :] = sequence
            return zeros
        else:
            return sequence