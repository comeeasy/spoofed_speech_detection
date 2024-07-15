import os
import random
import librosa
import torch

import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def compute_mel_spectrogram(y, sr=32000, n_fft=2048, hop_length=512, n_mels=128):
    # Compute mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # Convert to log scale (dB)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

# Mixup 함수 정의 (필요한 경우)
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[:len(bin_weights)]
    prob_pred = prob_pred[:len(bin_weights)]
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece
    
def auc_brier_ece_df(answer_df, submission_df):
    # Check for missing values in submission_df
    if submission_df.isnull().values.any():
        raise ValueError("The submission dataframe contains missing values.")

    # Check if the number and names of columns are the same in both dataframes
    if len(answer_df.columns) != len(submission_df.columns) or not all(answer_df.columns == submission_df.columns):
        raise ValueError("The columns of the answer and submission dataframes do not match.")
        
    submission_df = submission_df[submission_df.iloc[:, 0].isin(answer_df.iloc[:, 0])]
    submission_df.index = range(submission_df.shape[0])
    
    # Calculate AUC for each class
    auc_scores = []
    for column in answer_df.columns[1:]:
        y_true = answer_df[column]
        y_scores = submission_df[column]
        auc = roc_auc_score(y_true, y_scores)
        auc_scores.append(auc)

    # Calculate mean AUC
    mean_auc = np.mean(auc_scores)

    brier_scores = []
    ece_scores = []
    
    # Calculate Brier Score and ECE for each class
    for column in answer_df.columns[1:]:
        y_true = answer_df[column].values
        y_prob = submission_df[column].values
        
        # Brier Score
        brier = mean_squared_error(y_true, y_prob)
        brier_scores.append(brier)
        
        # ECE
        ece = expected_calibration_error(y_true, y_prob)
        ece_scores.append(ece)
    
    # Calculate mean Brier Score and mean ECE
    mean_brier = np.mean(brier_scores)
    mean_ece = np.mean(ece_scores)
    
    print(f"mean_auc: {mean_auc}, mean_brier: {mean_brier}, mean_ece: {mean_ece}")
    
    # Calculate combined score
    combined_score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece
    
    return combined_score

def auc_brier_ece(y_true, y_prob):
    auc_scores = []
    brier_scores = []
    ece_scores = []
    
    _y_true = y_true.detach().cpu().numpy()
    
    # _y_prob = y_prob.detach().cpu().numpy()
    _y_prob = y_prob
    
    for i in range(2):
        # B x 2 -> B
        try:
            auc = roc_auc_score(_y_true[:, i], _y_prob[:, i]) 
            auc_scores.append(auc)
        except:
            pass
        
        if not auc_scores:
            mean_auc = 0.5
        else:
            mean_auc = np.mean(auc_scores)
    
    for i in range(2):
        brier = mean_squared_error(_y_true[:, i], _y_prob[:, i])
        ece = expected_calibration_error(_y_true[:, i], _y_prob[:, i])
        brier_scores.append(brier)
        ece_scores.append(ece)
    
    mean_brier = np.mean(brier_scores)
    mean_ece = np.mean(ece_scores)
    
    # Calculate combined score
    combined_score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece
    
    return combined_score, mean_auc, mean_brier, mean_ece