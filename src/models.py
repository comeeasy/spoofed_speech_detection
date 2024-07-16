import numpy as np
import torch
import torch.nn as nn

import lightning as L
import torchmetrics

from peft import LoraConfig
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HubertForSequenceClassification
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Model
from transformers import AdamW
from transformers import Wav2Vec2Model

from tqdm import tqdm

from src.utils import mixup_data, mixup_criterion
from src.utils import auc_brier_ece
from src.lr_schedulers import LearningRateWarmUP, WarmupCosineAnnealingLR
from src.losses import CenterLoss
from src.AASIST import AASISTModule
from src.lora import LoRAWrapper
from src.RawNet2 import RawNet2

from itertools import chain



class Wav2Vec2_RawNet2(L.LightningModule):
    def __init__(self, args):
        super(Wav2Vec2_RawNet2, self).__init__()
        
        self.config = args
        
        # 모델 및 특징 추출기 설정
        self.sampling_rate = 16_000
        
        self.feature_extractor = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self", 
            torch_dtype=torch.float32, 
            attn_implementation="sdpa", 
            num_hidden_layers=12,
        )
        self.classifier = RawNet2()
        
        # freeze early layers
        self.FTL = 9
        self._set_trainable_parameters()
        
        # loss functions
        self.ce_sup_loss_fn = nn.CrossEntropyLoss()
        # self.ce_unsup_loss_fn = nn.CrossEntropyLoss(reduction='none')
        
        # center loss setting
        # self.center_loss_fn = CenterLoss(num_classes=4, feat_dim=4)
        # self.automatic_optimization = False
        # self.alpha = 0.8
        
        # FixMatch hparams
        self.fixmatch_threshold = 0.95
        
        # metric
        self.acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=4)
    
    def _set_trainable_parameters(self):
        # Feature extractor의 모든 파라미터를 고정
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        for i, transformer_layer in enumerate(self.feature_extractor.encoder.layers):
            if i >= self.FTL: # 우측 12 - FTL 개의 layers만 fine tunnig
                for param in transformer_layer.parameters():
                    param.requires_grad = True
            
        trainable_layers = filter(lambda p: p.requires_grad, self.parameters())
        print(f"======== Trainable Lora layers ===========")
        for trainable_layer in trainable_layers:
            print(trainable_layer.shape)
        print(f"==========================================")    
            
    def forward(self, x):
        output_fe = self.feature_extractor(x)
        feature = output_fe['extract_features']

        output = self.classifier(feature)

        return output

    def calculate_unsupervised_loss(self, weak_unlabeled_outputs, strong_aug_unlabeled_outputs):
        q_b = weak_unlabeled_outputs
        q_b_hat = torch.argmax(q_b, dim=1)
        
        conf, _ = torch.max(q_b, dim=1)
        mask = conf > self.fixmatch_threshold

        H = self.ce_unsup_loss_fn(strong_aug_unlabeled_outputs, q_b_hat)
        
        return torch.mean(mask * H)

    def training_step(self, batch, batch_idx):
        labeled_inputs, weak_aug_unlabeled_inputs, strong_aug_unlabeled_inputs, targets = batch
        
        labeled_outputs = self(labeled_inputs)
        weak_unlabeled_outputs = self(weak_aug_unlabeled_inputs)
        strong_aug_unlabeled_outputs = self(strong_aug_unlabeled_inputs)
        
        loss_supervised   = self.ce_sup_loss_fn(labeled_outputs, targets)
        # loss_center       = self.center_loss_fn(labeled_outputs, targets)
        # loss_unsupervised = self.calculate_unsupervised_loss(weak_unlabeled_outputs, strong_aug_unlabeled_outputs)
        
        loss = loss_supervised
        # loss = loss_supervised + loss_unsupervised + self.alpha * loss_center
        
        accuracy = self.acc_metric(labeled_outputs, targets)
        
        self.log("train_accuracy", accuracy)
        self.log("train_sup_loss", loss_supervised)
        # self.log("train_unsup_loss", loss_unsupervised)
        # self.log("train_center_loss", loss_center)
        self.log("train_loss", loss)
        
        ########### Center loss ====================
        # chech if `self.automatic_optimization = False`
        #
        # opt = self.optimizers()
        # scheduler = self.lr_schedulers()
        # opt.zero_grad()
        # loss.backward()
        # # multiple (1./alpha) in order to remove the effect of alpha on updating centers
        # for param in self.center_loss_fn.parameters():
        #     param.grad.data *= (1./self.alpha)
        # opt.step()
        # scheduler.step()
        #############################################
        
        return loss

    def validation_step(self, batch, batch_idx):
        labeled_inputs, weak_aug_unlabeled_inputs, strong_aug_unlabeled_inputs, targets = batch
        
        labeled_outputs = self(labeled_inputs)
        weak_unlabeled_outputs = self(weak_aug_unlabeled_inputs)
        strong_aug_unlabeled_outputs = self(strong_aug_unlabeled_inputs)
        
        loss_supervised   = self.ce_sup_loss_fn(labeled_outputs, targets)
        # loss_center       = self.center_loss_fn(labeled_outputs, targets)
        # loss_unsupervised = self.calculate_unsupervised_loss(weak_unlabeled_outputs, strong_aug_unlabeled_outputs)
        
        loss = loss_supervised
        # loss = loss_supervised + loss_unsupervised + self.alpha * loss_center
        
        accuracy = self.acc_metric(labeled_outputs, targets)
        
        self.log("val_accuracy", accuracy)
        self.log("val_sup_loss", loss_supervised)
        # self.log("val_unsup_loss", loss_unsupervised)
        # self.log("val_center_loss", loss_center)
        self.log("val_loss", loss)
        
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(params=self.parameters(), lr=self.config.lr, weight_decay=1e-2)
        scheduler = WarmupCosineAnnealingLR(optimizer=optimizer, warmup_steps=1000, total_steps=3000, min_lr=1e-8)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]




class Wav2Vec2FacebookFixMatchCenterLoss(L.LightningModule):
    def __init__(self, args, train=True):
        super(Wav2Vec2FacebookFixMatch, self).__init__()
        
        self.config = args
        
        # 모델 및 특징 추출기 설정
        model_name_or_path = "facebook/wav2vec2-base"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.model = AutoModelForAudioClassification.from_pretrained(model_name_or_path, num_labels=4)
        
        # define classifier
        self.model.classifier = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256),
            nn.SELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4, bias=True),
        )
        
        # freeze early layers
        # Add lora
        self._set_trainable_parameters()
        
        # loss functions
        self.ce_sup_loss_fn = nn.CrossEntropyLoss()
        self.ce_unsup_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.center_loss_fn = CenterLoss(num_classes=4, feat_dim=2)
        
        # FixMatch hparams
        self.fixmatch_threshold = 0.95
        
        # metric
        self.acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=4)
    
    def add_lora_to_qkv_projs(self, rank):
        lora_config = LoraConfig(
            r=rank,
            init_lora_weights=True,
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
            
    def forward(self, x):
        out = self.model(x).logits
        return out

    def calculate_unsupervised_loss(self, weak_unlabeled_outputs, strong_aug_unlabeled_outputs):
        q_b = weak_unlabeled_outputs
        q_b_hat = torch.argmax(q_b, dim=1)
        
        conf, _ = torch.max(q_b, dim=1)
        mask = conf > self.fixmatch_threshold

        H = self.ce_unsup_loss_fn(strong_aug_unlabeled_outputs, q_b_hat)
        
        return torch.mean(mask * H)

    def training_step(self, batch, batch_idx):
        labeled_inputs, weak_aug_unlabeled_inputs, strong_aug_unlabeled_inputs, targets = batch
        
        labeled_outputs = self(labeled_inputs)
        weak_unlabeled_outputs = self(weak_aug_unlabeled_inputs)
        strong_aug_unlabeled_outputs = self(strong_aug_unlabeled_inputs)
        
        loss_supervised = self.ce_sup_loss_fn(labeled_outputs, targets)
        loss_unsupervised = self.calculate_unsupervised_loss(weak_unlabeled_outputs, strong_aug_unlabeled_outputs)
        
        loss = loss_supervised + loss_unsupervised
        
        accuracy = self.acc_metric(labeled_outputs, targets)
        
        self.log("train_accuracy", accuracy)
        self.log("train_sup_loss", loss_supervised)
        self.log("train_unsup_loss", loss_unsupervised)
        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        labeled_inputs, weak_aug_unlabeled_inputs, strong_aug_unlabeled_inputs, targets = batch
        
        labeled_outputs = self(labeled_inputs)
        weak_unlabeled_outputs = self(weak_aug_unlabeled_inputs)
        strong_aug_unlabeled_outputs = self(strong_aug_unlabeled_inputs)
        
        loss_supervised = self.ce_sup_loss_fn(labeled_outputs, targets)
        loss_unsupervised = self.calculate_unsupervised_loss(weak_unlabeled_outputs, strong_aug_unlabeled_outputs)
        
        loss = loss_supervised + loss_unsupervised

        accuracy = self.acc_metric(labeled_outputs, targets)
        
        self.log("val_accuracy", accuracy)
        self.log("val_sup_loss", loss_supervised)
        self.log("val_unsup_loss", loss_unsupervised)
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
        optimizer = AdamW(params=self.parameters(), lr=self.config.lr, weight_decay=1e-2)
        scheduler = WarmupCosineAnnealingLR(optimizer=optimizer, warmup_steps=1000, total_steps=10000, min_lr=1e-9)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]





class Wav2Vec2FacebookFixMatch(L.LightningModule):
    def __init__(self, args, train=True):
        super(Wav2Vec2FacebookFixMatch, self).__init__()
        
        self.config = args
        
        # 모델 및 특징 추출기 설정
        model_name_or_path = "facebook/wav2vec2-base"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.model = AutoModelForAudioClassification.from_pretrained(model_name_or_path, num_labels=4)
        
        # define classifier
        self.model.classifier = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256),
            nn.SELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4, bias=True),
        )
        
        # freeze early layers
        # Add lora
        self._set_trainable_parameters()
        
        # loss functions
        self.ce_sup_loss_fn = nn.CrossEntropyLoss()
        self.ce_unsup_loss_fn = nn.CrossEntropyLoss(reduction='none')
        
        # FixMatch hparams
        self.fixmatch_threshold = 0.95
        
        # metric
        self.acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=4)
    
    def add_lora_to_qkv_projs(self, rank):
        lora_config = LoraConfig(
            r=rank,
            init_lora_weights=True,
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
            
    def forward(self, x):
        out = self.model(x).logits
        return out

    def calculate_unsupervised_loss(self, weak_unlabeled_outputs, strong_aug_unlabeled_outputs):
        q_b = weak_unlabeled_outputs
        q_b_hat = torch.argmax(q_b, dim=1)
        
        conf, _ = torch.max(q_b, dim=1)
        mask = conf > self.fixmatch_threshold

        H = self.ce_unsup_loss_fn(strong_aug_unlabeled_outputs, q_b_hat)
        
        return torch.mean(mask * H)

    def training_step(self, batch, batch_idx):
        labeled_inputs, weak_aug_unlabeled_inputs, strong_aug_unlabeled_inputs, targets = batch
        
        labeled_outputs = self(labeled_inputs)
        weak_unlabeled_outputs = self(weak_aug_unlabeled_inputs)
        strong_aug_unlabeled_outputs = self(strong_aug_unlabeled_inputs)
        
        loss_supervised = self.ce_sup_loss_fn(labeled_outputs, targets)
        loss_unsupervised = self.calculate_unsupervised_loss(weak_unlabeled_outputs, strong_aug_unlabeled_outputs)
        
        loss = loss_supervised + loss_unsupervised
        
        accuracy = self.acc_metric(labeled_outputs, targets)
        
        self.log("train_accuracy", accuracy)
        self.log("train_sup_loss", loss_supervised)
        self.log("train_unsup_loss", loss_unsupervised)
        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        labeled_inputs, weak_aug_unlabeled_inputs, strong_aug_unlabeled_inputs, targets = batch
        
        labeled_outputs = self(labeled_inputs)
        weak_unlabeled_outputs = self(weak_aug_unlabeled_inputs)
        strong_aug_unlabeled_outputs = self(strong_aug_unlabeled_inputs)
        
        loss_supervised = self.ce_sup_loss_fn(labeled_outputs, targets)
        loss_unsupervised = self.calculate_unsupervised_loss(weak_unlabeled_outputs, strong_aug_unlabeled_outputs)
        
        loss = loss_supervised + loss_unsupervised

        accuracy = self.acc_metric(labeled_outputs, targets)
        
        self.log("val_accuracy", accuracy)
        self.log("val_sup_loss", loss_supervised)
        self.log("val_unsup_loss", loss_unsupervised)
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
        scheduler = WarmupCosineAnnealingLR(optimizer=optimizer, warmup_steps=2000, total_steps=30000, min_lr=1e-9)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]










class AASIST(L.LightningModule):
    def __init__(self, args):
        super(AASIST, self).__init__()
        
        self.config = args
        
        # Load pre-trained AASIST module
        AASIST_weight_path = "/home/work/joono/joono/joono/src/AASIST_weight/AASIST.pth"
        classifier = AASISTModule()
        classifier.load_state_dict(torch.load(AASIST_weight_path))
        self._freeze_module(classifier.encoder)
        self._freeze_module(classifier.conv_time)
        self.classifier = classifier
        
        self.category_layer = nn.Sequential(
            nn.Linear(160, 160, bias=True),
            nn.LayerNorm(160),
            nn.SELU(inplace=True),
            nn.Dropout1d(p=0.2),
            nn.Linear(160, 4, bias=True),
        )
        self.probs_layer = nn.Sequential(
            nn.Linear(160, 160, bias=True),
            nn.LayerNorm(160),
            nn.SELU(inplace=True),
            nn.Dropout1d(p=0.2),
            nn.Linear(160, 2, bias=True),
        )
        
        # Add lora modules except encoder
        # self.classifier = LoRAWrapper(classifier, r=8)        
        
        # loss functions
        self.bce_loss_fn = nn.BCELoss()
        self.alpha = 0.5
        self.center_loss_fn = CenterLoss(num_classes=4, feat_dim=2)
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.automatic_optimization = False
    
    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
        
    
    def forward(self, x):
        last_hidden, output_logit = self.classifier(x)
        category_logit = self.category_layer(last_hidden)
        probs = torch.sigmoid(self.probs_layer(last_hidden))
        
        # return probs, output_logit, category_logit
        return probs, output_logit, category_logit

    def training_step(self, batch, batch_idx):
        inputs, targets, center_targets = batch
        
        # probs, output_logit, category_logit = self(inputs)
        probs, output_logit, category_logit = self(inputs)
        loss_bce = self.bce_loss_fn(probs, targets)
        loss_ce = self.ce_loss_fn(category_logit, center_targets)
        loss_center = self.center_loss_fn(output_logit, center_targets)
        
        loss = (loss_ce + loss_bce + self.alpha * loss_center) / 3
        
        metric, auc, brier, ece = auc_brier_ece(targets, probs)
        self.log("train_metric", metric)
        self.log("train_auc", auc)
        self.log("train_brier", brier)
        self.log("train_ece", ece)
        self.log("train_loss", loss)
        self.log("train_bce_loss", loss_bce)
        self.log("train_ce_loss", loss_ce)
        self.log("train_center", loss_center)
        
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        opt.zero_grad()
        loss.backward()
        # multiple (1./alpha) in order to remove the effect of alpha on updating centers
        for param in self.center_loss_fn.parameters():
            param.grad.data *= (1./self.alpha)
        opt.step()
        scheduler.step()
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, center_targets = batch
        
        probs, output_logit, category_logit = self(inputs)
        loss_bce = self.bce_loss_fn(probs, targets)
        loss_ce = self.ce_loss_fn(category_logit, center_targets)
        loss_center = self.center_loss_fn(output_logit, center_targets)
        
        loss = (loss_ce + loss_bce + self.alpha * loss_center) / 3
        # loss = (loss_ce + loss_bce) / 2
        
        # calculate
        metric, auc, brier, ece = auc_brier_ece(targets, probs)
        self.log("val_metric", metric)
        self.log("val_auc", auc)
        self.log("val_brier", brier)
        self.log("val_ece", ece)
        self.log("val_loss", loss)
        self.log("val_loss_bce", loss_bce)
        self.log("val_loss_ce", loss_ce)
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
        # optimizer = AdamW(params=chain(self.classifier.parameters(), self.center_loss_fn.parameters()), lr=self.config.lr)
        optimizer = AdamW(params=self.parameters(), lr=self.config.lr)
        scheduler = WarmupCosineAnnealingLR(optimizer=optimizer, warmup_steps=2000, total_steps=30000, min_lr=1e-9)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]



class Wav2Vec2Facebook(L.LightningModule):
    def __init__(self, args, train=True):
        super(Wav2Vec2Facebook, self).__init__()
        
        self.config = args
        
        # 모델 및 특징 추출기 설정
        model_name_or_path = "facebook/wav2vec2-base"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.model = AutoModelForAudioClassification.from_pretrained(model_name_or_path, num_labels=4)
        
        # define classifier
        self.model.classifier = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256),
            nn.SELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4, bias=True),
        )
        
        # freeze early layers
        # Add lora
        self._set_trainable_parameters()
        
        # loss functions
        self.loss_fn = nn.CrossEntropyLoss()
    
    def add_lora_to_qkv_projs(self, rank):
        lora_config = LoraConfig(
            r=rank,
            init_lora_weights=True,
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
            
    def forward(self, x):
        out = self.model(x).logits
        return out

    def training_step(self, batch, batch_idx):
        inputs, targets, class_target = batch
        
        outputs = self(inputs)
        
        loss = self.loss_fn(outputs, class_target)
        
        categories_np = outputs.detach().cpu().numpy()
        categories_softmax = np.exp(categories_np) / np.sum(np.exp(categories_np), axis=1, keepdims=True)
        category_probs = np.zeros((len(categories_softmax), 2))
        for row in range(len(category_probs)):
            cat_max_idx = np.argmax(categories_softmax[row])
            prob = categories_softmax[row, cat_max_idx]
            
            if      cat_max_idx == 0: category_probs[row] = np.array([1 - prob, 1 - prob])
            elif    cat_max_idx == 1: category_probs[row] = np.array([prob, 1 - prob])
            elif    cat_max_idx == 2: category_probs[row] = np.array([1 - prob, prob])
            elif    cat_max_idx == 3: category_probs[row] = np.array([prob, prob])
        
        metric, auc, brier, ece = auc_brier_ece(targets, category_probs)
        self.log("train_metric", metric)
        self.log("train_auc", auc)
        self.log("train_brier", brier)
        self.log("train_ece", ece)
        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, class_target = batch
        
        outputs = self(inputs)
        
        loss = self.loss_fn(outputs, class_target)
        
        categories_np = outputs.detach().cpu().numpy()
        categories_softmax = np.exp(categories_np) / np.sum(np.exp(categories_np), axis=1, keepdims=True)
        category_probs = np.zeros((len(categories_softmax), 2))
        for row in range(len(category_probs)):
            cat_max_idx = np.argmax(categories_softmax[row])
            prob = categories_softmax[row, cat_max_idx]
            
            if      cat_max_idx == 0: category_probs[row] = np.array([1 - prob, 1 - prob])
            elif    cat_max_idx == 1: category_probs[row] = np.array([prob, 1 - prob])
            elif    cat_max_idx == 2: category_probs[row] = np.array([1 - prob, prob])
            elif    cat_max_idx == 3: category_probs[row] = np.array([prob, prob])
        
        metric, auc, brier, ece = auc_brier_ece(targets, category_probs)
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
        scheduler = WarmupCosineAnnealingLR(optimizer=optimizer, warmup_steps=2000, total_steps=30000, min_lr=1e-9)
        
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