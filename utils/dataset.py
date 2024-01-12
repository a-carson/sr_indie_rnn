import numpy as np
from torch.utils.data import Dataset, Sampler
import torchaudio
import json
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class RNNDataModule(pl.LightningDataModule):

    def __init__(self,  train_input: str,
                        train_target: str,
                        val_input: str,
                        val_target: str,
                        test_input: str,
                        test_target: str,
                        train_sequence_length_seconds: float,
                        val_sequence_length_seconds: float,
                        test_sequence_length_seconds: float,
                 batch_size: int = 1,
                        pin_memory: bool = True,
                        shuffle=True):
        super().__init__()
        self.dirs = {
            "train": {"input": train_input,
                      "target": train_target},
            "val": {"input": val_input,
                    "target": val_target},
            "test": {"input": test_input,
                      "target": test_target},
        }
        self.train_sequence_length_seconds = train_sequence_length_seconds
        self.val_sequence_length_seconds = val_sequence_length_seconds
        self.test_sequence_length_seconds = test_sequence_length_seconds

        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage: str) -> None:

        train_input, sample_rate = torchaudio.load(self.dirs["train"]["input"])
        train_target, sample_rate = torchaudio.load(self.dirs["train"]["target"])
        val_input, sample_rate = torchaudio.load(self.dirs["val"]["input"])
        val_target, sample_rate = torchaudio.load(self.dirs["val"]["target"])
        test_input, sample_rate = torchaudio.load(self.dirs["test"]["input"])
        test_target, sample_rate = torchaudio.load(self.dirs["test"]["target"])

        train_seq_length_samples = int(self.train_sequence_length_seconds * sample_rate)
        val_seq_length_samples = int(self.val_sequence_length_seconds * sample_rate)
        test_seq_length_samples = int(self.test_sequence_length_seconds * sample_rate)

        self.train_ds = SequenceDataset(train_input, train_target, train_seq_length_samples)
        self.val_ds = SequenceDataset(val_input, val_target, val_seq_length_samples)
        self.test_ds = SequenceDataset(test_input, test_target, test_seq_length_samples)


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, pin_memory=self.pin_memory)



class SequenceDataset(Dataset):
    def __init__(self, input, target, sequence_length):
        self.input = input
        self.target = target
        if sequence_length is None:
            self._sequence_length = input.shape[1]
        else:
            self._sequence_length = sequence_length
        self.input_sequence = self.wrap_to_sequences(self.input, self._sequence_length)
        self.target_sequence = self.wrap_to_sequences(self.target, self._sequence_length)
        self._len = self.input_sequence.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.input_sequence[index, :, :], self.target_sequence[index, :, :]

    # wraps data from  [channels, samples] -> [sequences, samples, channels]
    def wrap_to_sequences(self, data, sequence_length):
        num_sequences = int(np.floor(data.shape[1] / sequence_length))
        truncated_data = data[:, 0:(num_sequences * sequence_length)]
        wrapped_data = truncated_data.transpose(0, 1).reshape((num_sequences, sequence_length, data.shape[0]))
        return np.float32(wrapped_data)


def tbptt_split_batch(batch, warmup, split_size):
    total_steps = batch[0].shape[1]
    splits = [[x[:, :warmup, :] for i, x in enumerate(batch[0:2])]]
    for t in range(warmup, total_steps, split_size):
        batch_split = [x[:, t: t + split_size, :] for i, x in enumerate(batch[0:2])]
        splits.append(batch_split)
    return splits


def load_dataset(input, target, root_dir=''):
    input = torchaudio.load(root_dir + input)
    sample_rate = input[1]
    target = torchaudio.load(root_dir + target)
    if sample_rate != target[1]:
        print("SAMPLE RATE MISMATCH")

    data = {"input": input[0], "target": target[0]}

    return data, sample_rate

def dataset_from_config(json_file_path):
    with open(json_file_path) as file:
        files = json.load(file)
    dataset = {}
    sample_rates = []
    for i, (key, paths) in enumerate(files.items()):
        audio_data = load_dataset(paths["input"], paths["target"])
        dataset[key] = audio_data[0]
        sample_rates.append(audio_data[1])

    assert(all(sr == sample_rates[0] for sr in sample_rates))

    return dataset, sample_rates[0]