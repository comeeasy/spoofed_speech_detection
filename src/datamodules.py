import os
import numpy as np
import pandas as pd
import random

import librosa
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, AutoFeatureExtractor

import lightning as L
import sklearn.model_selection as model_selection

from pydub import AudioSegment
from pydub.generators import WhiteNoise

from src.utils import compute_mel_spectrogram




class MFCCDataModule(L.LightningDataModule):
    def __init__(self, train_csv, test_csv, config):
        super().__init__()
        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)
        self.config = config
    
    def setup(self, stage: str):
        train_df, val_df, _, _ = model_selection.train_test_split(self.train_df, self.train_df['label'], test_size=0.2, random_state=self.config.seed)
        self.train_dset = MFCCDataset(df=train_df, sr=self.config.sr, n_mfcc=self.config.n_mfcc, n_classes=self.config.n_classes, train_mode=True)
        self.val_dset = MFCCDataset(df=val_df, sr=self.config.sr, n_mfcc=self.config.n_mfcc, n_classes=self.config.n_classes, train_mode=True)
        self.test_dset = MFCCDataset(df=self.test_df, sr=self.config.sr, n_mfcc=self.config.n_mfcc, n_classes=self.config.n_classes, train_mode=False)
    
    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.config.batch_size, num_workers=24)
    
    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.config.batch_size, num_workers=24)
    
    def test_dataloader(self):
        return DataLoader(self.test_dset, batch_size=self.config.batch_size, num_workers=24)

class MFCCDataset(torch.utils.data.Dataset):
    def __init__(self, df, sr, n_mfcc, n_classes, train_mode=True):
        self.data = df
        self.train_mode = train_mode
        
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_classes = n_classes
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        ogg_path = os.path.join("..", "dataset", row['path'])
        y, sr = librosa.load(ogg_path, sr=self.sr)
        
        # librosa패키지를 사용하여 mfcc 추출
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)

        if self.train_mode:
            label = row['label']
            label_vector = np.zeros(self.n_classes, dtype=np.float32)
            label_vector[0 if label == 'fake' else 1] = 1

        if self.train_mode:
            return mfcc, label_vector
        return mfcc


class MelSpectrogramDataModule(L.LightningDataModule):
    def __init__(self, train_csv, test_csv, config):
        super().__init__()
        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)
        self.config = config
    
    def setup(self, stage: str):
        train_df, val_df, _, _ = model_selection.train_test_split(self.train_df, self.train_df['label'], test_size=0.2, random_state=self.config.seed)
        self.train_dset = MelSpectrogramDataset(df=train_df, sr=self.config.sr, n_fft=self.config.n_fft, hop_length=self.config.hop_length, n_mels=self.config.n_mels, n_classes=self.config.n_classes, train_mode=True)
        self.val_dset = MelSpectrogramDataset(df=val_df, sr=self.config.sr, n_fft=self.config.n_fft, hop_length=self.config.hop_length, n_mels=self.config.n_mels, n_classes=self.config.n_classes, train_mode=True)
        self.test_dset = MelSpectrogramDataset(df=self.test_df, sr=self.config.sr, n_fft=self.config.n_fft, hop_length=self.config.hop_length, n_mels=self.config.n_mels, n_classes=self.config.n_classes, train_mode=False)
    
    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.config.batch_size, num_workers=24)
    
    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.config.batch_size, num_workers=24)
    
    def test_dataloader(self):
        return DataLoader(self.test_dset, batch_size=self.config.batch_size, num_workers=24)

class MelSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, df, sr, n_fft, hop_length, n_mels, n_classes, train_mode=True):
        self.data = df
        self.train_mode = train_mode
        
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_classes = n_classes
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        ogg_path = os.path.join("..", "dataset", row['path'])
        y, sr = librosa.load(ogg_path, sr=self.sr)
        
        S_dB = self.compute_mel_spectrogram(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)

        if self.train_mode:
            label = row['label']
            label_vector = np.zeros(self.n_classes, dtype=np.float32)
            label_vector[0 if label == 'fake' else 1] = 1

        if self.train_mode:
            return S_dB, label_vector
        return S_dB
    
    def compute_mel_spectrogram(y, sr=32000, n_fft=2048, hop_length=512, n_mels=128):
        # Compute mel-spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        # Convert to log scale (dB)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB
    
class AudioDataModule(L.LightningDataModule):
    def __init__(self, train_csv, test_csv, config):
        super().__init__()
        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)
        self.config = config
    
    def setup(self, stage: str):
        train_df, val_df, _, _ = model_selection.train_test_split(self.train_df, self.train_df['label'], test_size=0.2, random_state=self.config.seed)
        self.train_dset = AudioDataset(df=train_df, sr=self.config.sr, n_classes=self.config.n_classes, train_mode=True)
        self.val_dset = AudioDataset(df=val_df, sr=self.config.sr, n_classes=self.config.n_classes, train_mode=True)
        self.test_dset = AudioDataset(df=self.test_df, sr=self.config.sr, n_classes=self.config.n_classes, train_mode=False)
    
    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.config.batch_size, num_workers=24)
    
    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.config.batch_size, num_workers=24)
    
    def test_dataloader(self):
        return DataLoader(self.test_dset, batch_size=self.config.batch_size, num_workers=24)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, df, sr, train_mode=True):
        self.data = df
        self.train_mode = train_mode
        
        self.sr = sr
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        ogg_path = os.path.join("..", "dataset", row['path'])
        y, _ = librosa.load(ogg_path, sr=self.sr)
        
        if self.train_mode:
            label = row['label']
            label_vector = np.zeros(self.n_classes, dtype=np.float32)
            label_vector[0 if label == 'fake' else 1] = 1

        if self.train_mode:
            return y, label_vector
        return y

class EMDataModule(L.LightningDataModule):
    def __init__(self, train_csv, test_csv, config):
        super().__init__()
        self.config = config
        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)
    
    def setup(self, stage: str): # K-Fold 는 추후 추가..
        train_df, val_df, _, _ = model_selection.train_test_split(self.train_df, self.train_df['label'], test_size=0.05, random_state=self.config.seed)
        self.train_dset = EMDataset(df=train_df, train_mode=True)
        self.val_dset = EMDataset(df=val_df, train_mode=True)
        self.test_dset = EMDataset(df=self.test_df, train_mode=False)
        
    # Collate 함수 정의
    def _collate_fn(self, batch):
        signals, labels = zip(*batch)
        max_length = max([signal.size(0) for signal in signals])
        padded_signals = torch.zeros(len(signals), max_length)
        for i, signal in enumerate(signals):
            padded_signals[i, :signal.size(0)] = signal
        labels = torch.tensor(labels)
        return padded_signals, labels
    
    def train_dataloader(self):
        return DataLoader(self.train_dset, shuffle=True, batch_size=self.config.batch_size, collate_fn=self._collate_fn, num_workers=24)
    
    def val_dataloader(self):
        return DataLoader(self.val_dset, shuffle=True, batch_size=self.config.batch_size, collate_fn=self._collate_fn, num_workers=24)
    
    def test_dataloader(self):
        return DataLoader(self.val_dset, shuffle=False, batch_size=self.config.batch_size, collate_fn=self._collate_fn, num_workers=24)


# 데이터셋 클래스 정의
class EMDataset(torch.utils.data.Dataset):
    def __init__(self, df, train_mode=True):
        self.df = df
        self.train_mode = train_mode
        
        if train_mode:
            self.df['label'] = self.df['label'].apply(self._convert_labels)
        
        model_name_or_path = 'facebook/hubert-large-ll60k'
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
        self.sampling_rate = self.feature_extractor.sampling_rate
        

    def __len__(self):
        return self.df.shape[0]
    
    def _convert_labels(self, label):
        if label == 'fake':
            return (1., 0.)
        elif label == 'real':
            return (0., 1.)
        else:
            raise ValueError(f"Unknown label: {label}")
        
    # 음성 파일을 배열로 변환하는 함수
    def speech_file_to_array_fn(self, path):
        audio, _ = librosa.load(path, sr=self.sampling_rate)
        inputs = self.feature_extractor(audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        return inputs.input_values.squeeze()
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        path = os.path.join("..", "dataset", row['path'])
        if not os.path.exists(path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        signal = self.speech_file_to_array_fn(path)
        
        if self.train_mode:     
            label = row['label']
            return signal, label
        else:
            return signal, -1
        
class Wav2Vec2DataModule(L.LightningDataModule):
    def __init__(self, train_csv, test_csv, config):
        super().__init__()
        self.config = config
        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)
    
    def setup(self, stage: str): # K-Fold 는 추후 추가..
        train_df, val_df, _, _ = model_selection.train_test_split(self.train_df, self.train_df['label'], test_size=0.05, random_state=self.config.seed)
        self.train_dset = Wav2Vec2Dataset(df=train_df, model_name=self.config.model_name, train_mode=True)
        self.val_dset = Wav2Vec2Dataset(df=val_df, model_name=self.config.model_name, train_mode=True)
        self.test_dset = Wav2Vec2Dataset(df=self.test_df, model_name=self.config.model_name, train_mode=False)
        
    # Collate 함수 정의
    def _collate_fn(self, batch):
        signals, bce_labels, center_labels = zip(*batch)
        
        max_length = max([signal.shape[0] for signal in signals])
        padded_signals = torch.zeros(len(signals), max_length)
        for i, signal in enumerate(signals):
            padded_signals[i, :signal.shape[0]] = signal
        bce_labels = torch.tensor(bce_labels)
        center_labels = torch.tensor(center_labels)
        return padded_signals, bce_labels, center_labels
    
    def _collate_fn_test(self, batch):
        signals = zip(*batch)
        
        max_length = max([signal.shape[0] for signal in signals])
        padded_signals = torch.zeros(len(signals), max_length)
        for i, signal in enumerate(signals):
            padded_signals[i, :signal.shape[0]] = signal
        return padded_signals
    
    def train_dataloader(self):
        return DataLoader(self.train_dset, shuffle=True, batch_size=self.config.batch_size, collate_fn=self._collate_fn, num_workers=24)
    
    def val_dataloader(self):
        return DataLoader(self.val_dset, shuffle=True, batch_size=self.config.batch_size, collate_fn=self._collate_fn, num_workers=24)
    
    def test_dataloader(self):
        return DataLoader(self.test_df, shuffle=False, batch_size=self.config.batch_size, collate_fn=self._collate_fn_test, num_workers=24)


# 데이터셋 클래스 정의
class Wav2Vec2Dataset(torch.utils.data.Dataset):
    def __init__(self, df, model_name, train_mode=True):
        self.df = df
        self.train_mode = train_mode
        
        if self.train_mode:
            self.real_indices = np.where(df['label'] == 'real')[0]
            self.fake_indices = np.where(df['label'] == 'fake')[0]
        
        model_name_or_path = model_name
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        self.sampling_rate = self.feature_extractor.sampling_rate
        
    def __len__(self):
        return self.df.shape[0]
        
    def _load_ogg_file_from_row(self, row):
        path = os.path.join("..", "dataset", row['path'])
        if not os.path.exists(path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        audio, _ = librosa.load(path, sr=self.sampling_rate)
        return audio
    
    def _mix_two_audio_files(self, audio1, audio2):
        # 두 오디오 파일 중 더 긴 길이를 계산
        max_length = max(len(audio1), len(audio2))

        # 두 오디오 파일의 길이를 맞춤
        if len(audio1) < max_length:
            audio1 = np.pad(audio1, (0, max_length - len(audio1)), mode='constant')
        if len(audio2) < max_length:
            audio2 = np.pad(audio2, (0, max_length - len(audio2)), mode='constant')

        # Overlay the audio files
        mixed_audio = audio1 + audio2

        # Normalize to prevent clipping
        mixed_audio = librosa.util.normalize(mixed_audio)

        return mixed_audio

    def _audio_augmentation(self, audio):
        # audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
        # audio = librosa.effects.pitch_shift(audio, sr=self.sampling_rate, n_steps=np.random.randint(-2, 2))
        # Add noise
        noise = np.random.randn(len(audio))
        noise_factor = np.random.uniform(0.0, 0.3)
        audio = audio + noise_factor * noise
        augmented_data = librosa.util.normalize(audio)
        
        return augmented_data
    
    
    def _pad(self, x, max_len=64600):
        # https://github.com/clovaai/aasist/blob/main/data_utils.py
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len) + 1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x


    def _pad_random(self, x: np.ndarray, max_len: int = 64600):
        # https://github.com/clovaai/aasist/blob/main/data_utils.py
        x_len = x.shape[0]
        # if duration is already long enough
        if x_len >= max_len:
            stt = np.random.randint(x_len - max_len)
            return x[stt:stt + max_len]

        # if too short
        num_repeats = int(max_len / x_len) + 1
        padded_x = np.tile(x, (num_repeats))[:max_len]
        return padded_x
    
    def get_mixed_audios_and_targets(self, idx):
        if np.random.rand() <= (1/6):
            # sample (X, X)
            audio = np.random.normal(0, 1.0, size=16000 * 4)
            # input_values = torch.tensor(input_values)
            targets = (0.0, 0.0), 1
        else:
            # sample (F, X) or (R, X)
            dice = np.random.rand()
            if dice < 0.4:
                # load two audio files
                row = self.df.iloc[idx]
                audio = self._load_ogg_file_from_row(row)
                
                if row['label'] == 'fake': targets = (1.0, 0.0), 1
                elif row['label'] == 'real': targets = (0.0, 1.0), 2
                
            elif 0.4 <= dice < 0.8:
                # sample same audio (e.g. (F, F) or (R, R))
                row1 = self.df.iloc[idx]
                if row1['label'] == 'fake': # F
                    sampled_index = random.choice(self.fake_indices) # F
                    row2 = self.df.iloc[sampled_index]
                else: # R
                    sampled_index = random.choice(self.real_indices) # R
                    row2 = self.df.iloc[sampled_index]
                
                audio1 = self._load_ogg_file_from_row(row1)
                audio2 = self._load_ogg_file_from_row(row2)
                
                audio = self._mix_two_audio_files(audio1=audio1, audio2=audio2)
                
                if row1['label'] == 'fake' and row2['label'] == 'fake': targets = (1.0, 0.0), 1
                elif row1['label'] == 'real' and row2['label'] == 'real': targets = (0.0, 1.0), 2
            else:
                # sample different ((F, R) or (R, F))
                row1 = self.df.iloc[idx]
                if row1['label'] == 'fake': # F
                    sampled_index = random.choice(self.real_indices) # R
                    row2 = self.df.iloc[sampled_index]
                else: # R
                    sampled_index = random.choice(self.fake_indices) # F
                    row2 = self.df.iloc[sampled_index]
                
                audio1 = self._load_ogg_file_from_row(row1)
                audio2 = self._load_ogg_file_from_row(row2)
                
                audio = self._mix_two_audio_files(audio1=audio1, audio2=audio2)
                
                if row1['label'] == 'fake' and row2['label'] == 'real': targets = (1.0, 1.0), 3
                if row1['label'] == 'real' and row2['label'] == 'fake': targets = (1.0, 1.0), 3
                
            if self.train_mode:
                audio = self._pad_random(x=audio, max_len=self.sampling_rate * 10)
            else:
                audio = self._pad(x=audio, max_len=self.sampling_rate * 10)
            
            audio = self._audio_augmentation(audio)
            
        return audio, targets
    
    def __getitem__(self, idx):
        if self.train_mode:
            audio, targets = self.get_mixed_audios_and_targets(idx)
            inputs = self.feature_extractor(audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
            input_values = inputs.input_values.squeeze()
            bce_targets, center_targets = targets
            return input_values, bce_targets, center_targets
        else:
            row = self.df.iloc[idx]
            audio = self._load_ogg_file_from_row(row)
            inputs = self.feature_extractor(audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
            input_values = inputs.input_values.squeeze()
            return input_values
        
class AASIST2DataModule(L.LightningDataModule):
    def __init__(self, train_csv, test_csv, config):
        super().__init__()
        self.config = config
        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)
    
    def setup(self, stage: str): # K-Fold 는 추후 추가..
        train_df, val_df, _, _ = model_selection.train_test_split(self.train_df, self.train_df['label'], test_size=0.05, random_state=self.config.seed)
        self.train_dset = AASISTDataset(df=train_df, model_name=self.config.model_name, train_mode=True)
        self.val_dset = AASISTDataset(df=val_df, model_name=self.config.model_name, train_mode=True)
        self.test_dset = AASISTDataset(df=self.test_df, model_name=self.config.model_name, train_mode=False)
        
    # Collate 함수 정의
    def _collate_fn(self, batch):
        signals, labels = zip(*batch)
        
        max_length = max([signal.shape[0] for signal in signals])
        padded_signals = torch.zeros(len(signals), max_length)
        for i, signal in enumerate(signals):
            padded_signals[i, :signal.shape[0]] = signal
        labels = torch.tensor(labels)
        return padded_signals, labels
    
    def train_dataloader(self):
        return DataLoader(self.train_dset, shuffle=True, batch_size=self.config.batch_size, collate_fn=self._collate_fn, num_workers=24)
    
    def val_dataloader(self):
        return DataLoader(self.val_dset, shuffle=True, batch_size=self.config.batch_size, collate_fn=self._collate_fn, num_workers=24)
    
    def test_dataloader(self):
        return DataLoader(self.test_df, shuffle=False, batch_size=self.config.batch_size, collate_fn=self._collate_fn, num_workers=24)


# 데이터셋 클래스 정의
class AASISTDataset(torch.utils.data.Dataset):
    def __init__(self, df, train_mode=True):
        self.df = df
        self.train_mode = train_mode
        
        if self.train_mode:
            self.real_indices = np.where(df['label'] == 'real')[0]
            self.fake_indices = np.where(df['label'] == 'fake')[0]
        
        self.sampling_rate = 16000
        
    def __len__(self):
        return self.df.shape[0]
        
    def _load_ogg_file_from_row(self, row):
        path = os.path.join("..", "dataset", row['path'])
        if not os.path.exists(path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        audio, _ = librosa.load(path, sr=self.sampling_rate)
        norm_audio = librosa.util.normalize(audio)
        return norm_audio
    
    def _mix_two_audio_files(self, audio1, audio2):
        # 두 오디오 파일 중 더 긴 길이를 계산
        max_length = max(len(audio1), len(audio2))

        # 두 오디오 파일의 길이를 맞춤
        if len(audio1) < max_length:
            audio1 = np.pad(audio1, (0, max_length - len(audio1)), mode='constant')
        if len(audio2) < max_length:
            audio2 = np.pad(audio2, (0, max_length - len(audio2)), mode='constant')

        # Overlay the audio files
        mixed_audio = audio1 + audio2

        # Normalize to prevent clipping
        mixed_audio = librosa.util.normalize(mixed_audio)

        return mixed_audio

    def _audio_augmentation(self, audio):
        # Add noise
        noise = np.random.randn(len(audio))
        noise_factor = np.random.uniform(0.0, 0.4)
        audio = audio + noise_factor * noise
        augmented_data = librosa.util.normalize(audio)
        
        return augmented_data
    
    def get_mixed_audios_and_targets(self, idx):
        if np.random.rand() <= (1/6):
            # sample (X, X)
            audio = np.random.normal(0, 1.0, size=16000 * 6)
            # input_values = torch.tensor(input_values)
            targets = (0.0, 0.0)
        else:
            # sample (F, X) or (R, X)
            dice = np.random.rand()
            if dice < 0.4:
                # load two audio files
                row = self.df.iloc[idx]
                audio = self._load_ogg_file_from_row(row)
                
                if row['label'] == 'fake': targets = (1.0, 0.0)
                elif row['label'] == 'real': targets = (0.0, 1.0)
                
            elif 0.4 <= dice < 0.8:
                # sample same audio (e.g. (F, F) or (R, R))
                row1 = self.df.iloc[idx]
                if row1['label'] == 'fake': # F
                    sampled_index = random.choice(self.fake_indices) # F
                    row2 = self.df.iloc[sampled_index]
                else: # R
                    sampled_index = random.choice(self.real_indices) # R
                    row2 = self.df.iloc[sampled_index]
                
                audio1 = self._load_ogg_file_from_row(row1)
                audio2 = self._load_ogg_file_from_row(row2)
                
                audio = self._mix_two_audio_files(audio1=audio1, audio2=audio2)
                
                if row1['label'] == 'fake' and row2['label'] == 'fake': targets = (1.0, 0.0)
                elif row1['label'] == 'real' and row2['label'] == 'real': targets = (0.0, 1.0)
            else:
                # sample different ((F, R) or (R, F))
                row1 = self.df.iloc[idx]
                if row1['label'] == 'fake': # F
                    sampled_index = random.choice(self.real_indices) # R
                    row2 = self.df.iloc[sampled_index]
                else: # R
                    sampled_index = random.choice(self.fake_indices) # F
                    row2 = self.df.iloc[sampled_index]
                
                audio1 = self._load_ogg_file_from_row(row1)
                audio2 = self._load_ogg_file_from_row(row2)
                
                audio = self._mix_two_audio_files(audio1=audio1, audio2=audio2)
                
                if row1['label'] == 'fake' and row2['label'] == 'real': targets = (1.0, 1.0)
                if row1['label'] == 'real' and row2['label'] == 'fake': targets = (1.0, 1.0)
                
        return audio, targets
    
    def _pad(self, x, max_len=16000 * 12):
        # https://github.com/clovaai/aasist/blob/main/data_utils.py
        x_len = x.shape[0]
        if x_len > max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len)
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x

    def _pad_random(self, x: np.ndarray, max_len: int = 16000 * 12):
        # https://github.com/clovaai/aasist/blob/main/data_utils.py
        x_len = x.shape[0]
        # if duration is already long enough
        if x_len > max_len:
            stt = np.random.randint(x_len - max_len)
            return x[stt:stt + max_len - 1]

        # if too short
        num_repeats = int(max_len / x_len)
        padded_x = np.tile(x, (num_repeats))[:max_len]
        return padded_x
    
    def __getitem__(self, idx):
        if self.train_mode:
            audio, targets = self.get_mixed_audios_and_targets(idx)
            audio = self._pad_random(audio)
            audio = torch.tensor(audio)
            return audio, targets
        else:
            row = self.df.iloc[idx]
            audio = self._load_ogg_file_from_row(row)
            audio = self._pad(audio)
            audio = torch.tensor(audio)
            return audio

class AASISTCenterLossDataModule(L.LightningDataModule):
    def __init__(self, train_csv, test_csv, config):
        super().__init__()
        self.config = config
        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)
    
    def setup(self, stage: str): # K-Fold 는 추후 추가..
        train_df, val_df, _, _ = model_selection.train_test_split(self.train_df, self.train_df['label'], test_size=0.2, random_state=self.config.seed)
        self.train_dset = AASISTCenterLossDataset(df=train_df, train_mode=True)
        self.val_dset = AASISTCenterLossDataset(df=val_df, train_mode=True)
        self.test_dset = AASISTCenterLossDataset(df=self.test_df, train_mode=False)
        
    # Collate 함수 정의
    def _collate_fn(self, batch):
        signals, bce_labels, center_labels = zip(*batch)
        
        max_length = max([signal.shape[0] for signal in signals])
        padded_signals = torch.zeros(len(signals), max_length)
        for i, signal in enumerate(signals):
            padded_signals[i, :signal.shape[0]] = signal
        bce_labels = torch.tensor(bce_labels)
        center_labels = torch.tensor(center_labels)
        return padded_signals, bce_labels, center_labels
    
    def _collate_fn_test(self, batch):
        signals = zip(*batch)
        
        max_length = max([signal.shape[0] for signal in signals])
        padded_signals = torch.zeros(len(signals), max_length)
        for i, signal in enumerate(signals):
            padded_signals[i, :signal.shape[0]] = signal
        return padded_signals
    
    def train_dataloader(self):
        return DataLoader(self.train_dset, shuffle=True, batch_size=self.config.batch_size, collate_fn=self._collate_fn, num_workers=24)
    
    def val_dataloader(self):
        return DataLoader(self.val_dset, shuffle=True, batch_size=self.config.batch_size, collate_fn=self._collate_fn, num_workers=24)
    
    def test_dataloader(self):
        return DataLoader(self.test_df, shuffle=False, batch_size=self.config.batch_size, collate_fn=self._collate_fn_test, num_workers=24)

# 데이터셋 클래스 정의
class AASISTCenterLossDataset(AASISTDataset):
    def __init__(self, df, train_mode=True):
        super().__init__(df, train_mode)
    
    def get_mixed_audios_and_targets(self, idx):
        if np.random.rand() <= (1/6):
            # sample (X, X)
            audio = np.random.normal(0, 1.0, size=16000 * 6)
            # input_values = torch.tensor(input_values)
            targets = (0.0, 0.0), 0
        else:
            # sample (F, X) or (R, X)
            dice = np.random.rand()
            if dice < 0.4:
                # load two audio files
                row = self.df.iloc[idx]
                audio = self._load_ogg_file_from_row(row)
                
                if row['label'] == 'fake': targets = (1.0, 0.0), 1
                elif row['label'] == 'real': targets = (0.0, 1.0), 2
                
            elif 0.4 <= dice < 0.8:
                # sample same audio (e.g. (F, F) or (R, R))
                row1 = self.df.iloc[idx]
                if row1['label'] == 'fake': # F
                    sampled_index = random.choice(self.fake_indices) # F
                    row2 = self.df.iloc[sampled_index]
                else: # R
                    sampled_index = random.choice(self.real_indices) # R
                    row2 = self.df.iloc[sampled_index]
                
                audio1 = self._load_ogg_file_from_row(row1)
                audio2 = self._load_ogg_file_from_row(row2)
                
                audio = self._mix_two_audio_files(audio1=audio1, audio2=audio2)
                
                if row1['label'] == 'fake' and row2['label'] == 'fake': targets = (1.0, 0.0), 1
                elif row1['label'] == 'real' and row2['label'] == 'real': targets = (0.0, 1.0), 2
            else:
                # sample different ((F, R) or (R, F))
                row1 = self.df.iloc[idx]
                if row1['label'] == 'fake': # F
                    sampled_index = random.choice(self.real_indices) # R
                    row2 = self.df.iloc[sampled_index]
                else: # R
                    sampled_index = random.choice(self.fake_indices) # F
                    row2 = self.df.iloc[sampled_index]
                
                audio1 = self._load_ogg_file_from_row(row1)
                audio2 = self._load_ogg_file_from_row(row2)
                
                audio = self._mix_two_audio_files(audio1=audio1, audio2=audio2)
                
                if row1['label'] == 'fake' and row2['label'] == 'real': targets = (1.0, 1.0), 3
                if row1['label'] == 'real' and row2['label'] == 'fake': targets = (1.0, 1.0), 3
                
            audio = self._audio_augmentation(audio)
            
        return audio, targets
    
    def __getitem__(self, idx):
        if self.train_mode:
            audio, targets = self.get_mixed_audios_and_targets(idx)
            audio = self._pad_random(audio)
            audio = torch.tensor(audio)
            bce_targets, center_targets = targets
            return audio, bce_targets, center_targets
        else:
            row = self.df.iloc[idx]
            audio = self._load_ogg_file_from_row(row)
            audio = self._pad(audio)
            audio = torch.tensor(audio)
            return audio



class FixMatchDataModule(L.LightningDataModule):
    def __init__(self, train_csv, test_csv, unlabeled_csv, config):
        super().__init__()
        self.config = config
        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)
        self.unlabeled_df = pd.read_csv(unlabeled_csv)
    
    def setup(self, stage: str): # K-Fold 는 추후 추가..
        train_df, val_df, _, _ = model_selection.train_test_split(self.train_df, self.train_df['label'], test_size=0.1, random_state=self.config.seed)
        self.train_dset = FixMatchDataset(df=train_df, unlabeled_df=self.unlabeled_df, train_mode=True, model_name=self.config.model_name)
        self.val_dset = FixMatchDataset(df=val_df, unlabeled_df=self.unlabeled_df, train_mode=True, model_name=self.config.model_name)
        self.test_dset = FixMatchDataset(df=self.test_df, train_mode=False, model_name=self.config.model_name)
        
    # Collate 함수 정의
    def _collate_fn(self, batch):
        labeled_inputs, weak_aug_unlabeled_inputs, strong_aug_unlabeled_inputs, targets = zip(*batch)
        
        max_length = max([labeled_input.shape[0] for labeled_input in labeled_inputs])
        padded_labeled_audios = torch.zeros(len(labeled_inputs), max_length)
        for i, labeled_input in enumerate(labeled_inputs):
            padded_labeled_audios[i, :labeled_input.shape[0]] = labeled_input
        
        max_length = max([weak_aug_unlabeled_input.shape[0] for weak_aug_unlabeled_input in weak_aug_unlabeled_inputs])
        padded_weak_aug_unlabeled_audios = torch.zeros(len(weak_aug_unlabeled_inputs), max_length)
        for i, weak_aug_unlabeled_input in enumerate(weak_aug_unlabeled_inputs):
            padded_weak_aug_unlabeled_audios[i, :weak_aug_unlabeled_input.shape[0]] = weak_aug_unlabeled_input
        
        max_length = max([strong_aug_unlabeled_input.shape[0] for strong_aug_unlabeled_input in strong_aug_unlabeled_inputs])
        padded_strong_aug_unlabeled_audios = torch.zeros(len(strong_aug_unlabeled_inputs), max_length)
        for i, strong_aug_unlabeled_input in enumerate(strong_aug_unlabeled_inputs):
            padded_strong_aug_unlabeled_audios[i, :strong_aug_unlabeled_input.shape[0]] = strong_aug_unlabeled_input
            
        targets = torch.tensor(targets)
        return padded_labeled_audios, padded_weak_aug_unlabeled_audios, padded_strong_aug_unlabeled_audios, targets
    
    def _collate_fn_test(self, batch):
        audios = zip(*batch)
        
        max_length = max([audio.audio[0] for audio in audios])
        padded_audios = torch.zeros(len(audios), max_length)
        for i, audio in enumerate(audios):
            padded_audios[i, :audio.shape[0]] = audio
        return padded_audios
    
    def train_dataloader(self):
        return DataLoader(self.train_dset, shuffle=True, batch_size=self.config.batch_size, collate_fn=self._collate_fn, num_workers=24)
    
    def val_dataloader(self):
        return DataLoader(self.val_dset, shuffle=True, batch_size=self.config.batch_size, collate_fn=self._collate_fn, num_workers=24)
    
    def test_dataloader(self):
        return DataLoader(self.test_df, shuffle=False, batch_size=self.config.batch_size, collate_fn=self._collate_fn_test, num_workers=24)

# 데이터셋 클래스 정의
class FixMatchDataset(torch.utils.data.Dataset):
    def __init__(self, df, model_name, train_mode=True, unlabeled_df=None):
        super().__init__()
        self.df = df
        self.train_mode = train_mode
        
        if not unlabeled_df is None:
            self.unlabeled_df = unlabeled_df
        if self.train_mode:
            self.real_indices = np.where(df['label'] == 'real')[0]
            self.fake_indices = np.where(df['label'] == 'fake')[0]
        
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.sampling_rate = self.feature_extractor.sampling_rate
        
        # fixmatch hparams
        self.fixmatch_mu = 7
        
    def __len__(self):
        return self.df.shape[0]
        
    def _load_ogg_file_from_row(self, row):
        path = os.path.join("..", "dataset", row['path'])
        if not os.path.exists(path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        audio, _ = librosa.load(path, sr=self.sampling_rate)
        audio = librosa.util.normalize(audio)
        return audio
    
    def _mix_two_audio_files(self, audio1, audio2):
        # 두 오디오 파일 중 더 긴 길이를 계산
        max_length = max(len(audio1), len(audio2))

        # 두 오디오 파일의 길이를 맞춤
        if len(audio1) < max_length:
            audio1 = np.pad(audio1, (0, max_length - len(audio1)), mode='constant')
        if len(audio2) < max_length:
            audio2 = np.pad(audio2, (0, max_length - len(audio2)), mode='constant')

        # Overlay the audio files
        mixed_audio = audio1 + audio2

        # Normalize to prevent clipping
        mixed_audio = librosa.util.normalize(mixed_audio)

        return mixed_audio

    def _strong_audio_augmentation(self, audio):
        # Add noise
        noise = np.random.randn(len(audio))
        noise_factor = np.random.uniform(0.0, 0.2)
        audio = audio + noise_factor * noise
        augmented_data = librosa.util.normalize(audio)
        
        return augmented_data

    def _weak_audio_augmentation(self, audio):
        # Add noise
        noise = np.random.randn(len(audio))
        noise_factor = np.random.uniform(0.0, 0.01)
        audio = audio + noise_factor * noise
        augmented_data = librosa.util.normalize(audio)
        
        return augmented_data
    
    def get_mixed_audios_and_targets(self, idx):
        if np.random.rand() <= (1/4):
            # sample (X, X)
            audio = np.random.normal(0, 1.0, size=16000 * np.random.randint(4, 8))
            # input_values = torch.tensor(input_values)
            targets = 0
        else:
            # sample (F, X) or (R, X)
            dice = np.random.rand()
            if dice < (1/3):
                # load two audio files
                row = self.df.iloc[idx]
                audio = self._load_ogg_file_from_row(row)
                
                if row['label'] == 'fake': targets = 1
                elif row['label'] == 'real': targets = 2
                
            elif (1/3) <= dice < (2/3):
                # sample same audio (e.g. (F, F) or (R, R))
                row1 = self.df.iloc[idx]
                if row1['label'] == 'fake': # F
                    sampled_index = random.choice(self.fake_indices) # F
                    row2 = self.df.iloc[sampled_index]
                else: # R
                    sampled_index = random.choice(self.real_indices) # R
                    row2 = self.df.iloc[sampled_index]
                
                audio1 = self._load_ogg_file_from_row(row1)
                audio2 = self._load_ogg_file_from_row(row2)
                
                audio = self._mix_two_audio_files(audio1=audio1, audio2=audio2)
                
                if row1['label'] == 'fake' and row2['label'] == 'fake': targets = 1
                elif row1['label'] == 'real' and row2['label'] == 'real': targets = 2
            else:
                # sample different ((F, R) or (R, F))
                row1 = self.df.iloc[idx]
                if row1['label'] == 'fake': # F
                    sampled_index = random.choice(self.real_indices) # R
                    row2 = self.df.iloc[sampled_index]
                else: # R
                    sampled_index = random.choice(self.fake_indices) # F
                    row2 = self.df.iloc[sampled_index]
                
                audio1 = self._load_ogg_file_from_row(row1)
                audio2 = self._load_ogg_file_from_row(row2)
                
                audio = self._mix_two_audio_files(audio1=audio1, audio2=audio2)
                
                if row1['label'] == 'fake' and row2['label'] == 'real': targets = 3
                if row1['label'] == 'real' and row2['label'] == 'fake': targets = 3
                
        return audio, targets
    
    def __getitem__(self, idx):
        if self.train_mode:
            labeled_audio, targets = self.get_mixed_audios_and_targets(idx)
            unlabeled_idx = np.random.random_integers(len(self.unlabeled_df)-1)
            unlabeled_audio = self._load_ogg_file_from_row(self.unlabeled_df.iloc[unlabeled_idx])
            
            labeled_audio = self._weak_audio_augmentation(labeled_audio)
            strong_aug_unlabeled_audio = self._strong_audio_augmentation(unlabeled_audio)
            weak_aug_unlabeled_audio = self._weak_audio_augmentation(unlabeled_audio)
            
            labeled_inputs = self.feature_extractor(labeled_audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
            labeled_inputs = labeled_inputs.input_values.squeeze()
            
            strong_aug_unlabeled_inputs = self.feature_extractor(strong_aug_unlabeled_audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
            strong_aug_unlabeled_inputs = strong_aug_unlabeled_inputs.input_values.squeeze()
            
            weak_aug_unlabeled_inputs = self.feature_extractor(weak_aug_unlabeled_audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
            weak_aug_unlabeled_inputs = weak_aug_unlabeled_inputs.input_values.squeeze()
            
            return labeled_inputs, weak_aug_unlabeled_inputs, strong_aug_unlabeled_inputs, targets
        else:
            row = self.df.iloc[idx]
            audio = self._load_ogg_file_from_row(row)
            inputs = self.feature_extractor(audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
            input_values = inputs.input_values.squeeze()
            return input_values
