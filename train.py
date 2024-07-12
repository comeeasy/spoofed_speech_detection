import torch

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

import argparse
import wandb

from src.models import MLP, EModel, AASIST, Wav2Vec2Facebook
from src.utils import seed_everything
from src.datamodules import MFCCDataModule, EMDataModule, Wav2Vec2DataModule, AASIST2DataModule, AASISTCenterLossDataModule

import warnings
warnings.filterwarnings('ignore')


def main(args):
    # set logger
    wandb.login(relogin=True)
    wandb_logger = WandbLogger(project="DV_DV.Deep", log_model="all")


    train_csv, test_csv = "../dataset/train.csv", "../dataset/test.csv"

    # Model ===================================================================================================    
    # MFCC & MLP
    # datamodule = MFCCDataModule(train_csv="../dataset/train.csv", test_csv="../dataset/test.csv", config=args)
    # model = MLP(input_dim=args.n_mfcc, hidden_dim=128, output_dim=args.n_classes, config=args)
    
    # EModel
    # datamodule = EMDataModule(train_csv="../dataset/train.csv", test_csv="../dataset/test.csv", config=args)
    # model = EModel(config=args)
    
    # Wav2Vec2 with Lora
    # datamodule = Wav2Vec2DataModule(train_csv="../dataset/train.csv", test_csv="../dataset/test.csv", config=args)
    # model = Wav2Vec2Facebook(args)
    
    # AASIST
    datamodule = AASISTCenterLossDataModule(train_csv=train_csv, test_csv=test_csv, config=args)
    model = AASIST(args)
    #==========================================================================================================    
    
    
    early_stopping_callback = EarlyStopping (
        monitor='val_metric',   # Monitor validation loss
        patience=5,             # Number of epochs with no improvement after which training will be stopped
        verbose=True,           # Display information about early stopping
        mode='min'              # Minimize the monitored quantity
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_metric',       # Monitor validation metric
        filename='best-checkpoint_aug_oneshot', # Filename template for the checkpoints
        save_top_k=1,               # Save only the best model
        mode='min'                  # Minimize the monitored quantity
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = L.Trainer(
        accelerator="auto", 
        devices=[1], 
        logger=wandb_logger, 
        callbacks=[
            early_stopping_callback, 
            checkpoint_callback,
            lr_monitor
        ], 
        max_epochs=args.epochs
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--model_name", type=str, choices=['facebook/wav2vec2-base'])

    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--sr", type=int, default=32000)
    # parser.add_argument("--n_mfcc", type=int, default=13)
    parser.add_argument("--n_classes", type=int, default=2)
    args = parser.parse_args()

    print(args)

    seed_everything(args.seed) # Seed 고정
    main(args)