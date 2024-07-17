import torch
torch.set_float32_matmul_precision('medium')

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

import argparse
import wandb

from src.models import (
    MLP, EModel, AASIST, Wav2Vec2Facebook,
    Wav2Vec2FacebookFixMatch, Wav2Vec2_RawNet2
)
from src.utils import seed_everything
from src.datamodules import (
    MFCCDataModule, EMDataModule, Wav2Vec2DataModule, 
    AASIST2DataModule, AASISTCenterLossDataModule,
    FixMatchDataModule
)


import warnings
warnings.filterwarnings('ignore')


def main(args):
    # set logger
    wandb.login(relogin=True)
    wandb_logger = WandbLogger(project="DV_DV.Deep", log_model="all")


    train_csv, test_csv = "../dataset/train.csv", "../dataset/test.csv"
    unlabeled_csv = "../dataset/unlabeled_data.csv"

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
    # datamodule = AASISTCenterLossDataModule(train_csv=train_csv, test_csv=test_csv, config=args)
    # model = AASIST(args)
    
    # Wav2Vec2 + Lora + train-test-augmentation + FixMatch(SSL)
    datamodule = FixMatchDataModule(train_csv=train_csv, test_csv=test_csv, unlabeled_csv=unlabeled_csv, config=args)
    model = Wav2Vec2_RawNet2(args=args)
    
    #==========================================================================================================    
    
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',       # Monitor validation metric
        filename='best-checkpoint', # Filename template for the checkpoints
        save_top_k=1,               # Save only the best model
        mode='max'                  # Minimize the monitored quantity
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = L.Trainer(
        accelerator="auto", 
        devices=[1], 
        logger=wandb_logger, 
        callbacks=[
            checkpoint_callback,
            lr_monitor
        ], 
        max_epochs=args.epochs
    )
    
    trainer.fit(model, datamodule=datamodule)
    
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