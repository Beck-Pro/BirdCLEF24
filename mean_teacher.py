import audiomentations as A
import os, time, librosa, random
from functools import partial
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from timm.models import resnet34d
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from tqdm import tqdm
from contextlib import nullcontext

# Global Vars
NO_LABEL = -1
NUM_CLASSES = 182
isTrain = True

class config:
    seed = 42
    device = "cuda:0"
    
    train_tp_csv = './train_metadata.csv'
    train_unlabeled = './train_unlabeled.csv'
    test_csv = './sample_submission.csv'
    save_path = './'
    
    encoder = resnet34d
    encoder_features = 512
    
    consistency_weight = 100.0
    consistency_rampup = 6
    ema_decay = 0.995
    positive_weight = 2.0
    
    lr = 1e-3
    epochs = 25
    batch_size = 32
    num_workers = 4
    train_5_folds = False
    
    period = 5 # 6 second clips
    step = 1
    model_params = {
        'sample_rate': 48000,
        'window_size': 2048,
        'hop_size': 512,
        'mel_bins': 384,
        'fmin': 20,
        'fmax': 48000 // 2,
        'classes_num': NUM_CLASSES
    }
    
    augmenter = A.Compose([
        A.AddGaussianNoise(p=0.33, max_amplitude=0.02),
        A.AddGaussianSNR(p=0.33),
        #A.SpecFrequencyMask(p=0.33),        
        A.TimeMask(min_band_part=0.01, max_band_part=0.25, p=0.33),
        A.Gain(p=0.33)
    ])


def get_n_fold_df(labeled_csv_path, unlabeled_csv_path, folds=5):
    train_csv = pd.read_csv(labeled_csv_path)
    df_unlabel = pd.read_csv(unlabeled_csv_path)
    train_csv['fold'] = -1

    X = train_csv["filename"].values
    y = train_csv["primary_label"].values
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.seed)
    for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
        train_csv.loc[valid_index, 'fold'] = int(fold)

    return train_csv, df_unlabel


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):
        try:
            self.y_true.extend(y_true.detach().cpu().numpy().tolist())
            self.y_pred.extend(torch.sigmoid(y_pred).cpu().detach().numpy().tolist())
        except:
            print("UPDATE FAILURE")

    def update_list(self, y_true, y_pred):
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)

    @property
    def avg(self):
        score_class, weight = lwlrap(np.array(self.y_true), np.array(self.y_pred))
        self.score = (score_class * weight).sum()

        return self.score
    

def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled

def _one_sample_positive_class_precisions(scores, truth):
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)

    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)

    retrieved_classes = np.argsort(scores)[::-1]

    class_rankings = np.zeros(num_classes, dtype=int)
    class_rankings[retrieved_classes] = range(num_classes)

    retrieved_class_true = np.zeros(num_classes, dtype=bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True

    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(float)))
    return pos_class_indices, precision_at_hits


def lwlrap(truth, scores):
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(scores[sample_num, :],
                                                                                     truth[sample_num, :])
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = precision_at_hits

    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))

    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    return per_class_lwlrap, weight_per_class


def pretty_print_metrics(fold, epoch, optimizer, train_loss_metrics, val_loss_metrics):
    print(f"""
    {time.ctime()} \n
    Fold:{fold}, Epoch:{epoch}, LR:{optimizer.param_groups[0]['lr']:.7}, Cons. Weight: {train_loss_metrics['consistency_weight']}\n
    --------------------------------------------------------
    Metric:              Train    |   Val
    --------------------------------------------------------
    Loss:                {train_loss_metrics['loss']:0.4f}   |   {val_loss_metrics['loss']:0.4f}\n
    LWLRAP:              {train_loss_metrics['lwlrap']:0.4f}   |   {val_loss_metrics['lwlrap']:0.4f}\n
    Class Loss:          {train_loss_metrics['class_loss']:0.4f}   |   {val_loss_metrics['class_loss']:0.4f}\n
    Consistency Loss:    {train_loss_metrics['consistency_loss']:0.4f}   |   {val_loss_metrics['consistency_loss']:0.4f}\n
    --------------------------------------------------------\n
    """)


class TestDataset(Dataset):
    def __init__(self, df, data_path, period=10, step=1):
        self.data_path = data_path
        self.period = period
        self.step = step
        self.recording_ids = list(df["filename"].unique())

    def __len__(self):
        return len(self.recording_ids)

    def __getitem__(self, idx):
        recording_id = self.recording_ids[idx]

        y, sr = sf.read(f"{self.data_path}/{recording_id}.flac")

        len_y = len(y)
        effective_length = sr * self.period
        effective_step = sr * self.step

        y_ = []
        i = 0
        while i + effective_length <= len_y:
            y__ = y[i:i + effective_length]

            y_.append(y__)
            i = i + effective_step

        y = np.stack(y_)

        label = np.zeros(NUM_CLASSES, dtype='f')

        return {
            "waveform": y,
            "target": torch.tensor(label, dtype=torch.float),
            "id": recording_id
        }


def predict_on_test(model, test_loader):
    model.eval()
    pred_list = []
    id_list = []
    with torch.no_grad():
        t = tqdm(test_loader)
        for i, sample in enumerate(t):
            input = sample["waveform"].to(config.device)
            bs, seq, w = input.shape
            input = input.reshape(bs * seq, w)
            id = sample["id"]
            output, _ = model(input)
            output = output.reshape(bs, seq, -1)
            output, _ = torch.max(output, dim=1)
            
            output = output.cpu().detach().numpy().tolist()
            pred_list.extend(output)
            id_list.extend(id)

    return pred_list, id_list

from timm import create_model
class AttentionHead(nn.Module):

    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.conv_attention = nn.Conv1d(in_channels=in_features, 
                                        out_channels=out_features,
                                        kernel_size=1, stride=1, 
                                        padding=0, bias=True)
        self.conv_classes = nn.Conv1d(in_channels=in_features, 
                                      out_channels=out_features,
                                      kernel_size=1, stride=1, 
                                      padding=0, bias=True)
        self.batch_norm_attention = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv_attention)
        init_layer(self.conv_classes)
        init_bn(self.batch_norm_attention)

    def forward(self, x):
        norm_att = torch.softmax(torch.tanh(self.conv_attention(x)), dim=-1)
        classes = self.conv_classes(x)
        x = torch.sum(norm_att * classes, dim=2)
        return x, norm_att, classes


class SEDAudioClassifier(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()
        self.interpolate_ratio = 32

        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size, window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)

        self.batch_norm = nn.BatchNorm2d(mel_bins)
        self.encoder = create_model('resnet34d', pretrained=True, in_chans=1)
        self.fc = nn.Linear(config.encoder_features, config.encoder_features, bias=True)
        self.att_head = AttentionHead(config.encoder_features, classes_num)
        self.avg_pool = nn.modules.pooling.AdaptiveAvgPool2d((1, 1))

        self.init_weight()

    def init_weight(self):
        init_bn(self.batch_norm)
        init_layer(self.fc)
        self.att_head.init_weights()

    def forward(self, input, spec_aug=False, mixup_lambda=None, return_encoding=False):
        x = self.spectrogram_extractor(input.float())
        x = self.logmel_extractor(x)
        
        x = x.transpose(1, 3)
        x = self.batch_norm(x)
        x = x.transpose(1, 3)

        x = self.encoder.forward_features(x)
        x = torch.mean(x, dim=3)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        clipwise_output, norm_att, segmentwise_output = self.att_head(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        
        # Ensure all outputs are used
        if return_encoding:
            return clipwise_output, framewise_output, x
        return clipwise_output, framewise_output


def sigmoid_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.sigmoid(input_logits)
    target_softmax = torch.sigmoid(target_logits)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False
                     ) / num_classes


class MeanTeacherLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.positive_weight = torch.ones(
            NUM_CLASSES).to(self.device) * config.positive_weight
        self.class_criterion = nn.BCEWithLogitsLoss(
            reduction='none', pos_weight=self.positive_weight)
        self.consistency_criterion = sigmoid_mse_loss

    def make_safe(self, pred):
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        return torch.where(torch.isinf(pred), torch.zeros_like(pred), pred)
        
    def get_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return config.consistency_weight * sigmoid_rampup(
            epoch, config.consistency_rampup)
    
    def forward(self, student_pred, teacher_pred, target, classif_weights, epoch):
        student_pred = student_pred.to(self.device)
        teacher_pred = teacher_pred.to(self.device)
        target = target.to(self.device)
        classif_weights = classif_weights.to(self.device)

        student_pred = self.make_safe(student_pred)
        teacher_pred = self.make_safe(teacher_pred).detach().data

        batch_size = len(target)
        labeled_batch_size = target.ne(NO_LABEL).all(axis=1).sum().item() + 1e-3

        student_classif, student_consistency = student_pred, student_pred
        student_class_loss = (self.class_criterion(
            student_classif, target) * classif_weights / labeled_batch_size).sum()

        consistency_weights = self.get_consistency_weight(epoch)
        consistency_loss = consistency_weights * self.consistency_criterion(
            student_consistency, teacher_pred) / batch_size
        loss = student_class_loss + consistency_loss
        return loss, student_class_loss, consistency_loss, consistency_weights


class MeanTeacherDataset(Dataset):
    
    def __init__(self, labeled_df, unlabeled_df, transforms, data_path="./train_audio", unlabel_path="./unlabeled_soundscapes", val=False, num_classes=182):
        self.labeled_df = labeled_df.reset_index(drop=True)
        self.unlabeled_df = unlabeled_df.reset_index(drop=True)
        self.transforms = transforms
        self.data_path = data_path
        self.unlabel_path = unlabel_path
        self.val = val
        self.num_classes = num_classes

        self.recording_ids = self.labeled_df["filename"].values
        self.primary_label = self.labeled_df["primary_label"].values
        self.unlabeled_recording_ids = self.unlabeled_df["filename"].values

        self.label_mapping = self.create_label_mapping()

    def __len__(self):
        return len(self.labeled_df) + len(self.unlabeled_df)

    def __getitem__(self, idx):
        if idx >= len(self.labeled_df):
            audio, label, rec_id, sr = self.get_unlabeled_item(idx - len(self.labeled_df))
            classif_weights = np.zeros(self.num_classes, dtype='f')
        else:
            audio, label, rec_id, sr = self.get_labeled_item(idx)
            classif_weights = np.ones(self.num_classes, dtype='f')

        audio_teacher = np.copy(audio)
        audio = self.transforms(samples=audio, sample_rate=sr)
        audio_teacher = self.transforms(samples=audio_teacher, sample_rate=sr)
        
        return {
            "waveform": torch.tensor(audio, dtype=torch.float32),
            "teacher_waveform": torch.tensor(audio_teacher, dtype=torch.float32),
            "target": torch.tensor(label, dtype=torch.float32),
            "classification_weights": torch.tensor(classif_weights, dtype=torch.float32),
            "id": rec_id
        }

    def get_labeled_item(self, idx):
        rec_id = self.recording_ids[idx]
        primary_label = self.primary_label[idx]
        file_name, _ = os.path.splitext(rec_id)
        file_path = f"{self.data_path}/{file_name}.ogg"
        #print(f"Loading labeled file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")

        rec, sr = librosa.load(file_path, sr=None)
        rec = rec.astype(np.float32)

        label_index = self.label_to_index(primary_label)
        label = one_hot_encode(label_index, self.num_classes)
        return rec, label, rec_id, sr
    
    def get_unlabeled_item(self, idx):
        rec_id = self.unlabeled_recording_ids[idx]
        file_name, _ = os.path.splitext(rec_id)
        file_path = f"{self.unlabel_path}/{file_name}.ogg"
        #print(f"Loading unlabeled file: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")

        rec, sr = librosa.load(file_path, sr=None)
        rec = rec.astype(np.float32)

        label = np.zeros(self.num_classes, dtype=np.float32)  # No label for unlabeled data
        
        return rec, label, rec_id, sr

    def create_label_mapping(self):
        unique_labels = set(self.primary_label)
        return {label: index for index, label in enumerate(unique_labels)}

    def label_to_index(self, label):
        return self.label_mapping[label]

# Function to convert label to one-hot encoding
def one_hot_encode(label, num_classes):
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[label] = 1.0
    return one_hot

def pad_collate_fn(batch):
    target_length = 48000 * 10  # Assuming 10 seconds and 48000 sample rate
    
    for sample in batch:
        waveform = sample['waveform']
        waveform_length = waveform.shape[0]
        
        if waveform_length < target_length:
            padding_length = target_length - waveform_length
            waveform = torch.nn.functional.pad(waveform, (0, padding_length))
        else:
            waveform = waveform[:target_length]
        
        sample['waveform'] = waveform
        
        # Do the same for teacher_waveform
        teacher_waveform = sample['teacher_waveform']
        teacher_waveform_length = teacher_waveform.shape[0]
        
        if teacher_waveform_length < target_length:
            teacher_padding_length = target_length - teacher_waveform_length
            teacher_waveform = torch.nn.functional.pad(teacher_waveform, (0, teacher_padding_length))
        else:
            teacher_waveform = teacher_waveform[:target_length]
        
        sample['teacher_waveform'] = teacher_waveform

    # Separate out non-tensor items (e.g., 'id')
    batch_tensors = {key: torch.stack([sample[key] for sample in batch]) for key in batch[0] if isinstance(batch[0][key], torch.Tensor)}
    batch_non_tensors = {key: [sample[key] for sample in batch] for key in batch[0] if not isinstance(batch[0][key], torch.Tensor)}

    # Merge dictionaries, prioritizing tensor stacks
    batch = {**batch_tensors, **batch_non_tensors}

    return batch

from torch.utils.data.distributed import DistributedSampler
def get_data_loader(labeled_df, unlabeled_df, rank, world_size, is_val=False, num_classes=182):
    dataset = MeanTeacherDataset(
        labeled_df=labeled_df,
        unlabeled_df=unlabeled_df,
        transforms=config.augmenter,
        num_classes=num_classes
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=not is_val)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size // world_size,
        sampler=sampler,
        drop_last=not is_val,
        num_workers=config.num_workers,
        collate_fn=pad_collate_fn,
        pin_memory=False    # Enable pinned memory for faster data transfer to GPU
    )


from contextlib import nullcontext
import torch
import gc

# Update teacher to be exponential moving average of student params.
def update_teacher_params(student, teacher, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train_one_epoch(student, mean_teacher, loader, 
                    criterion, optimizer, scheduler, epoch, is_val=False, rank=0):
    global_step = 0
    losses = AverageMeter()
    consistency_loss_avg = AverageMeter()
    class_loss_avg = AverageMeter()
    comp_metric = MetricMeter()
    
    accum_steps = 8  # Number of mini-batches to accumulate gradients

    if is_val:
        student.eval()
        mean_teacher.eval()
        context = torch.no_grad()
    else:
        student.train()
        mean_teacher.train()
        context = nullcontext()
    
    with context:
        t = tqdm(loader)
        optimizer.zero_grad()
        
        for i, sample in enumerate(t):
            student_input = sample['waveform'].to(rank, non_blocking=True)
            teacher_input = sample['teacher_waveform'].to(rank, non_blocking=True)
            target = sample['target'].to(rank, non_blocking=True)
            classif_weights = sample['classification_weights'].to(rank, non_blocking=True)
            batch_size = len(target)

            student_pred, _  = student(student_input)
            teacher_pred, _  = mean_teacher(teacher_input)

            loss, class_loss, consistency_loss, consistency_weight = criterion(
                student_pred, teacher_pred, target, classif_weights, epoch)
            
            # Ensure loss requires gradient
            if not is_val:
                loss.requires_grad_(True)
                loss.backward()
                if (i + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    update_teacher_params(student, mean_teacher, config.ema_decay, global_step)

                    scheduler.step()

            comp_metric.update(target, student_pred)
            losses.update(loss.item(), batch_size)
            consistency_loss_avg.update(consistency_loss.item(), batch_size)
            class_loss_avg.update(class_loss.item(), batch_size)
            global_step += 1

            del loss
            del sample
            del student_input
            del teacher_input
            del target
            del student_pred
            del teacher_pred
            # Clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()

            t.set_description(f"Epoch:{epoch} - Loss:{losses.avg:0.4f}")
        t.close()
    return {'lwlrap': comp_metric.avg, 
            'loss': losses.avg, 
            'consistency_loss': consistency_loss_avg.avg, 
            'class_loss': class_loss_avg.avg, 
            'consistency_weight': consistency_weight}


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def cleanup():
    dist.destroy_process_group()

import torch.utils.checkpoint as checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self, model):
        super(CheckpointedModel, self).__init__()
        self.model = model

    def forward(self, *inputs):
        return checkpoint.checkpoint(self.model, *inputs)

def get_model(is_mean_teacher=False):
    model = SEDAudioClassifier(**config.model_params)
    model = model.to(config.device)
    
    # Detach params for Exponential Moving Average Model (aka the Mean Teacher).
    # We'll manually update these params instead of using backprop.
    if is_mean_teacher:
        for param in model.parameters():
            param.detach_()
    else:
        for param in model.parameters():
            param.requires_grad = True  # Ensure gradients are required
            
        model = CheckpointedModel(model)
    return model

# DDP training function
def train_ddp(rank, world_size, df_labeled, df_unlabeled, fold):
    setup(rank, world_size)
    
    train_df = df_labeled[df_labeled.fold != fold]
    val_df = df_labeled[df_labeled.fold == fold]
    
    train_loader = get_data_loader(train_df, df_unlabeled, rank, world_size)
    val_loader = get_data_loader(val_df, df_unlabeled, rank, world_size, is_val=True)

    student_model = get_model().to(rank)
    teacher_model = get_model(is_mean_teacher=True).to(rank)
    
    student_model = DDP(student_model, device_ids=[rank], find_unused_parameters=True)
    teacher_model = teacher_model.to(rank)
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.lr)
    num_train_steps = int(len(train_loader) * config.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_steps)
    criterion = MeanTeacherLoss(rank)  # Pass the device to MeanTeacherLoss

    best_val_metric = -np.inf
    val_metrics = []
    train_metrics = []
    for epoch in range(config.epochs):
        train_loss_metrics = train_one_epoch(
            student_model, teacher_model, train_loader, 
            criterion, optimizer, scheduler, epoch, rank=rank)
        val_loss_metrics = train_one_epoch(
            student_model, teacher_model, val_loader, 
            criterion, optimizer, scheduler, epoch, is_val=True, rank=rank)

        train_metrics.append(train_loss_metrics)
        val_metrics.append(val_loss_metrics)
        pretty_print_metrics(fold, epoch, optimizer, train_loss_metrics, val_loss_metrics)
        
        if val_loss_metrics['lwlrap'] > best_val_metric:
            print(f"    LWLRAP Improved from {best_val_metric} --> {val_loss_metrics['lwlrap']}\n")
            torch.save(teacher_model.state_dict(), os.path.join(config.save_path, f'32_fold-{fold}.bin'))
            best_val_metric = val_loss_metrics['lwlrap']
    
    cleanup()

import torch.multiprocessing as mp

if __name__ == "__main__":
    import os
    import pandas as pd

    # Specify the directory containing the .ogg files
    directory = './unlabeled_soundscapes'

    # Specify the output CSV file
    output_csv = 'train_unlabeled.csv'

    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Filter the list to include only .ogg files
    ogg_files = [f for f in files if f.endswith('.ogg')]

    # Create a pandas DataFrame from the list of .ogg files
    df = pd.DataFrame(ogg_files, columns=['filename'])
    df['primary_label'] = 'a'
    # Write the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

    world_size = torch.cuda.device_count()
    df_labeled, df_unlabeled = get_n_fold_df(config.train_tp_csv, config.train_unlabeled)
    
    for fold in range(5 if config.train_5_folds else 1):
        mp.spawn(train_ddp,
                 args=(world_size, df_labeled, df_unlabeled, fold),
                 nprocs=world_size,
                 join=True)
