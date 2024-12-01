import cv2
import numpy as np
from torch.utils.data import DataLoader
from base_work import data
import segmentation_models_pytorch as smp
import torch
import pytorch_lightning as pl
import os
import  matplotlib.pyplot as plt
import utils
from watermark.jaws import SystemJASW

current_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0001
IMG_TEST_PATH = "../dataset/Test/test3.jfif"
IMG_TRAINING_SIZE = 384
epochs = 10

class SegmentationWaterModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes,

            **kwargs
        )
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.train_outputs = []
        self.valid_outputs = []

    def forward(self, image):
        image = image.clone().detach().float()
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]
        assert mask.ndim == 4

        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, "train")
        self.train_outputs.append(output)
        return output

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.train_outputs, "train")
        self.train_outputs.clear()

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, "valid")
        self.valid_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.valid_outputs, "valid")
        self.valid_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=lr)

def train_model(arch):
    MODEL_PATH = f"../models/model_{arch}_epoch_{epochs}_lr_{lr}.pth"
    dataset = data.DatasetSegWater('train', 0.8, 384, shuffle=False, DATASET_PATH='../dataset/Water Bodies Dataset')
    model = SegmentationWaterModel(arch, "resnet34", in_channels=3, out_classes=1, )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True,  )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True,  )

    trainer = pl.Trainer(
        max_epochs=epochs,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    torch.save(model.state_dict(), MODEL_PATH)

def test_model(arch):
    MODEL_PATH = f"../models/model_{arch}_epoch_10_lr_0.0001.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegmentationWaterModel(arch, "resnet34", in_channels=3, out_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    dataset = data.DataTestModel(384, DATASET_PATH="../dataset/Test", apply_contrast=False)
    TEST_NUM = len(dataset)

    with torch.no_grad():
        tensor_input = [dataset[i] for i in range(TEST_NUM)]
        tensor_input = torch.tensor(np.array(tensor_input), dtype=torch.float32).to(device)
        pred = model.forward(tensor_input)
        pred = torch.nn.functional.sigmoid(pred)
        pred = (pred > 0.5).float()

    pred = pred.cpu().numpy()
    for i in range(TEST_NUM):
        utils.show_img(dataset[i].transpose(1, 2, 0))
        utils.show_img(pred[i][0])
    plt.show()

def do_watermark(img, model_path, arch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = img/255.
    model = SegmentationWaterModel(arch, "resnet34", in_channels=3, out_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        model_input = img.transpose((2,0,1))
        model_input = torch.tensor(np.array([model_input]), dtype=torch.float32).to(device)
        pred = model.forward(model_input)
        pred = torch.nn.functional.sigmoid(pred)
        pred = (pred > 0.5).float()

    pred = pred.cpu().numpy()
    pred = pred[0][0]
    hidden_info = 34

    sys_insert_watermark = SystemJASW(K=100, M=256, alpha=0.02)
    CW = sys_insert_watermark.insert_watermark(img[:,:,0], pred, hidden_info)
    extracted_info = sys_insert_watermark.extract_watermark(CW)

    utils.show_images(
        [
            (img, "Исходное изображение"),
            (pred, "Предсказанная маска"),
            (img[:,:,0], "Исходный канал RED"),
            (CW, "Заполненый контейнер")
        ],
        2,
        f"Hidden b: {hidden_info}\n"
        f"Extracted info: {extracted_info}\n"
        f"PSNR = {sys_insert_watermark.calculate_psnr(img[:,:,0]*255., CW*255.)}"
    )

def calculate_loss(arch):
    MODEL_PATH = f"../models/model_{arch}_epoch_{epochs}_lr_{lr}.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    total_loss = 0.0

    model = SegmentationWaterModel(arch, "resnet34", in_channels=3, out_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    loss_fn = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
    dataset = data.DatasetSegWater('test', 0.2, 384, shuffle=False, DATASET_PATH='../dataset/Water Bodies Dataset')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            pred = model(images)
            loss = loss_fn(pred, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Average Dice Loss over batch size {batch_size}: {avg_loss:.4f}")

# UNET   0.1045
# FPN    0.1036
# UNET++ 0.0809

if __name__ == "__main__":
    #train_model('UnetPlusPlus')
    #train_model('FPN')
    # test_model('UnetPlusPlus')
    # test_model('Unet')
    #test_model('FPN')

    arch = 'UnetPlusPlus'
    img = cv2.imread(IMG_TEST_PATH, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_TRAINING_SIZE, IMG_TRAINING_SIZE))
    do_watermark(img, model_path=f"../models/model_{arch}_epoch_10_lr_0.0001.pth",
                arch=arch)

    calculate_loss('UnetPlusPlus')