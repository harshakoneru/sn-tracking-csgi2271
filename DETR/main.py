import os
import torch
import torchvision
from transformers import DetrForObjectDetection, DetrImageProcessor
import random
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import cv2
from torch.utils.data import DataLoader
import numpy as np
import supervision as sv
from pytorch_lightning.callbacks import ModelCheckpoint


print(torch.__file__)
print(torch.__version__)

# How many GPUs are there?
print(torch.cuda.device_count())

# Get the name of the current GPU
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Is PyTorch using a GPU?
print(torch.cuda.is_available())


def collate_fn(batch):
    # DETR authors employ various image sizes during training, making it not possible
    # to directly batch together images. Hence they pad the images to the biggest
    # resolution in a given batch, and create a corresponding binary pixel_mask
    # which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

# Settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.8
image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
DATASET_ROOT_PATH = "/scratch/hk3820/ByteTrack/datasets/SoccerNet/"
TRAIN_DIRECTORY = os.path.join(DATASET_ROOT_PATH, "train")
VAL_DIRECTORY = os.path.join(DATASET_ROOT_PATH, "challenge")  # Assuming 'challenge' as validation
TEST_DIRECTORY = os.path.join(DATASET_ROOT_PATH, "test")
ANNOTATIONS_PATH = os.path.join(DATASET_ROOT_PATH, "annotations")

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        annotations_file: str,
        image_processor,
        train: bool = True
    ):
        annotation_file_path = os.path.join(ANNOTATIONS_PATH, annotations_file)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target

# Initialize datasets
TRAIN_DATASET = CocoDetection(
    image_directory_path=TRAIN_DIRECTORY,
    annotations_file="train.json",
    image_processor=image_processor,
    train=True)

TEST_DATASET = CocoDetection(
    image_directory_path=VAL_DIRECTORY,
    annotations_file="challenge2023.json",
    image_processor=image_processor,
    train=False)

VAL_DATASET = CocoDetection(
    image_directory_path=TEST_DIRECTORY,
    annotations_file="test.json",
    image_processor=image_processor,
    train=False)

# Print dataset sizes
print("Number of training examples:", len(TRAIN_DATASET))
print("Number of Challenge examples:", len(TEST_DATASET))
print("Number of Validation examples:", len(VAL_DATASET))

# select random image
image_ids = TRAIN_DATASET.coco.getImgIds()
image_id = random.choice(image_ids)
print('Image #{}'.format(image_id))

# load image and annotatons 
image = TRAIN_DATASET.coco.loadImgs(image_id)[0]
annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
image_path = os.path.join(TRAIN_DATASET.root, image['file_name'])
image = cv2.imread(image_path)

# annotate
detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)

# we will use id2label function for training
categories = TRAIN_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}

labels = [
    f"{id2label[class_id]}" 
    for _, _, class_id, _ 
    in detections
]

box_annotator = sv.BoxAnnotator()
TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4,num_workers =4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4, num_workers=4)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4, num_workers=4)

class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT, 
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss,batch_size=4)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation/loss", loss,batch_size=4,sync_dist=True)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
            
        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here: 
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
# settings
MAX_EPOCHS = 50
checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=1, monitor="val_loss")

trainer = Trainer(devices=torch.cuda.device_count(), accelerator="gpu",callbacks=[checkpoint_callback], strategy="ddp", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

trainer.fit(model)

