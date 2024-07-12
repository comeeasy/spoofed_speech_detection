import torch
import torch.nn as nn

import lightning as L

from peft import LoraConfig
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HubertForSequenceClassification
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from transformers import AdamW
from tqdm import tqdm

from src.utils import mixup_data, mixup_criterion
from src.utils import auc_brier_ece
from src.lr_schedulers import LearningRateWarmUP, WarmupCosineAnnealingLR
from src.losses import CenterLoss
from src.AASIST import AASISTModule

from itertools import chain




class AASIST(L.LightningModule):
    def __init__(self, args):
        super(AASIST, self).__init__()
        
        self.config = args
        self.classifier = AASISTModule()
        
        # loss functions
        self.loss_fn = nn.BCELoss()
        self.alpha = 0.8
        self.center_loss_fn = CenterLoss(num_classes=4, feat_dim=160)
        self.automatic_optimization = False
    
    def forward(self, x):
        last_hidden, output = self.classifier(x)
        probs = torch.sigmoid(output)
        return probs, last_hidden

    def training_step(self, batch, batch_idx):
        inputs, targets, center_targets = batch
        
        probs, last_hidden = self(inputs)
        loss_bce = self.loss_fn(probs, targets)
        loss_centor = self.center_loss_fn(last_hidden, center_targets)
        
        loss = loss_bce + self.alpha * loss_centor
        
        metric, auc, brier, ece = auc_brier_ece(targets, probs)
        self.log("train_metric", metric)
        self.log("train_auc", auc)
        self.log("train_brier", brier)
        self.log("train_ece", ece)
        self.log("train_loss", loss)
        self.log("train_bce_loss", loss_bce)
        self.log("train_center", loss_centor)
        
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        
        opt.zero_grad()
        loss.backward()
        # multiple (1./alpha) in order to remove the effect of alpha on updating centers
        for param in self.loss_fn.parameters():
            param.grad.data *= (1./self.alpha)
        opt.step()
        scheduler.step()
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, center_targets = batch
        
        probs, output_center = self(inputs)
        loss_bce = self.loss_fn(probs, targets)
        loss_center = self.center_loss_fn(output_center, center_targets)
        loss = loss_bce + self.alpha * loss_center
        
        # calculate
        metric, auc, brier, ece = auc_brier_ece(targets, probs)
        self.log("val_metric", metric)
        self.log("val_auc", auc)
        self.log("val_brier", brier)
        self.log("val_ece", ece)
        self.log("val_loss", loss)
        self.log("val_loss_bce", loss_bce)
        self.log("val_loss_center", loss_center)
        
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch
        
        probs = self(inputs)
        print(f"{probs}")

    def inference(self, test_loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for features in tqdm(test_loader):
                probs = self(features)

                probs  = probs.cpu().detach().numpy()
                predictions += probs.tolist()
        return predictions

    def configure_optimizers(self):
        optimizer = AdamW(params=chain(self.classifier.parameters(), self.center_loss_fn.parameters()), lr=self.config.lr)
        scheduler = WarmupCosineAnnealingLR(optimizer=optimizer, warmup_steps=1000, total_steps=30000, min_lr=1e-8)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]



class Wav2Vec2Facebook(L.LightningModule):
    def __init__(self, args, train=True):
        super(Wav2Vec2Facebook, self).__init__()
        
        self.config = args
        
        # 모델 및 특징 추출기 설정
        model_name_or_path = "facebook/wav2vec2-base"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.model = AutoModelForAudioClassification.from_pretrained(model_name_or_path, num_labels=2)
        
        # define classifier
        self.model.classifier = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2, bias=True),
        )
        
        # freeze early layers
        # Add lora
        self._set_trainable_parameters()
        
        # initialize layers
        if train:
            self._initialize_parameters()
        
        # loss functions
        self.loss_fn = nn.BCELoss()
    
    def add_lora_to_qkv_projs(self, rank):
        lora_config = LoraConfig(
            r=rank,
            init_lora_weights=False,
            target_modules=['k_proj', 'q_proj', 'v_proj', 'out_proj']
            # transformer 의 attention 구하는 모듈들에만 lora 적용 (Stable diffusion)
        )
        self.model.add_adapter(lora_config)
        
    def _set_trainable_parameters(self):
        # 전체 모델의 파라미터를 고정
        for param in self.model.parameters():
            param.requires_grad = False

        self.add_lora_to_qkv_projs(rank=8)

        for param in self.model.classifier.parameters():
            param.requires_grad = True
        for param in self.model.projector.parameters():
            param.requires_grad = True
            
        trainable_layers = filter(lambda p: p.requires_grad, self.model.parameters())
        print(f"======== Trainable Lora layers ===========")
        for trainable_layer in trainable_layers:
            print(trainable_layer.shape)
        print(f"==========================================")    
            
    def _initialize_parameters(self):
        # Apply specific initializations to all layers
        for layer in self.model.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        for layer in self.model.projector.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        
    def forward(self, x):
        out = self.model(x).logits
        out = torch.sigmoid(out)
        return out

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.loss_fn(pred, y_a) + (1 - lam) * self.loss_fn(pred, y_b)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        '''
        if self.global_step % 4 == 0:
            x_batch, y_batch_a, y_batch_b, lam = mixup_data(inputs, targets)
            probs = self(x_batch)
            loss = mixup_criterion(self.loss_fn, probs, y_batch_a, y_batch_b, lam)
        else:
            probs = self(inputs)
            loss = self.loss_fn(probs, targets)
        '''
        
        print(f"inputs: {inputs.shape}, targets: {targets.shape}")
        
        probs = self(inputs)
        
        print(f"probs: {probs[:3]}, targets: {targets[:3]}")
        
        loss = self.loss_fn(probs, targets)
        
        print(f"loss: {loss}")
        
        metric, auc, brier, ece = auc_brier_ece(targets, probs)
        self.log("train_metric", metric)
        self.log("train_auc", auc)
        self.log("train_brier", brier)
        self.log("train_ece", ece)
        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        
        probs = self(inputs)
        loss = self.loss_fn(probs, targets)
        
        # calculate
        metric, auc, brier, ece = auc_brier_ece(targets, probs)
        self.log("val_metric", metric)
        self.log("val_auc", auc)
        self.log("val_brier", brier)
        self.log("val_ece", ece)
        self.log("val_loss", loss)
        
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch
        
        probs = self(inputs)
        print(f"{probs}")

    def inference(self, test_loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for features in tqdm(test_loader):
                probs = self(features)

                probs  = probs.cpu().detach().numpy()
                predictions += probs.tolist()
        return predictions

    def configure_optimizers(self):
        optimizer = AdamW(params = self.parameters(), lr = self.config.lr)
        scheduler = WarmupCosineAnnealingLR(optimizer=optimizer, warmup_steps=1000, total_steps=30000, min_lr=1e-8)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]



class EModel(L.LightningModule):
    def __init__(self, config):
        super(EModel, self).__init__()
        
        self.config = config
        
        # 모델 및 특징 추출기 설정
        model_name_or_path = 'facebook/hubert-large-ll60k'
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=2)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.model = HubertForSequenceClassification.from_pretrained(model_name_or_path, config=config)
        
        # freeze early layers
        self._freeze_model()
        
        # initialize layers
        self._initialize_parameters()
        
        # loss functions
        # self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.BCELoss()
    
    def _freeze_model(self):
        # 전체 모델의 파라미터를 고정
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True
        for param in self.model.projector.parameters():
            param.requires_grad = True
            
    def _initialize_parameters(self):
        # Apply specific initializations to all layers
        for layer in self.model.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Conv1d):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        for layer in self.model.projector.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Conv1d):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        
    def forward(self, x):
        out = self.model(x).logits
        out = torch.sigmoid(out)
        return out

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.loss_fn(pred, y_a) + (1 - lam) * self.loss_fn(pred, y_b)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        
        if self.global_step % 4 == 0:
            x_batch, y_batch_a, y_batch_b, lam = mixup_data(inputs, targets)
            probs = self(x_batch)
            loss = mixup_criterion(self.loss_fn, probs, y_batch_a, y_batch_b, lam)
        else:
            probs = self(inputs)
            loss = self.loss_fn(probs, targets)
        
        metric, auc, brier, ece = auc_brier_ece(targets, probs)
        self.log("train_metric", metric)
        self.log("train_auc", auc)
        self.log("train_brier", brier)
        self.log("train_ece", ece)
        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        
        probs = self(inputs)
        loss = self.loss_fn(probs, targets)
        
        # calculate
        metric, auc, brier, ece = auc_brier_ece(targets, probs)
        self.log("val_metric", metric)
        self.log("val_auc", auc)
        self.log("val_brier", brier)
        self.log("val_ece", ece)
        self.log("val_loss", loss)
        
        return loss

    def test_step(self, *args: torch.Any, **kwargs: torch.Any):
        pass

    def inference(self, test_loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for features in tqdm(iter(test_loader)):
                probs = self(features)

                probs  = probs.cpu().detach().numpy()
                predictions += probs.tolist()
        return predictions

    def configure_optimizers(self):
        optimizer = AdamW(params = self.parameters(), lr = self.config.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5, eta_min=1e-7)
        scheduler = WarmupCosineAnnealingLR(optimizer=optimizer, warmup_steps=5000, total_steps=20000, min_lr=1e-8)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
        

class MLP(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, config):
        super(MLP, self).__init__()
        
        self.config = config
        
        # Define model
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        # define Loss
        self.criterion = nn.BCELoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        features, targets = batch
        output = self(features)
        
        loss = self.criterion(output, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        output = self(features)
        
        loss = self.criterion(output, targets)
        return loss

    def test_step(self, *args: torch.Any, **kwargs: torch.Any):
        pass

    def inference(self, test_loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for features in tqdm(iter(test_loader)):
                features = features.float()
                
                probs = self(features)

                probs  = probs.cpu().detach().numpy()
                predictions += probs.tolist()
        return predictions

    def configure_optimizers(self):
        return torch.optim.Adam(params = self.parameters(), lr = self.config.lr)