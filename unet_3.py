import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import PIL
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt

# … (clases DoubleConv, Down, Up, OutConv, UNet idénticas a las tuyas) …
class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class Down(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x =  self.conv(x)

        return x


class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}\n")

if __name__ == '__main__':
    # --- 1) Leer y concatenar imágenes y máscaras ---
    images = sorted(os.listdir('./flood/data/Image/'))
    image_list = []
    mask_list  = []
    for image in images:
        # 1a) Leer imagen
        dd = PIL.Image.open(f'./flood/data/Image/{image}').convert('RGB')
        tt = TF.pil_to_tensor(dd)                    # [3, H, W], uint8
        tt = TF.resize(tt, (100, 100))               # [3, 100,100]
        tt = tt.float() / 255.                        # [3, 100,100], float
        tt = tt.unsqueeze(0)                          # [1, 3, 100,100]
        if tt.shape != (1, 3, 100, 100):
            continue
        image_list .append(tt)

        # 1b) Leer máscara correspondiente
        mask_name = image.replace('.jpg', '.png')
        ddm = PIL.Image.open(f'./flood/data/Mask/{mask_name}').convert('L')
        mm = TF.pil_to_tensor(ddm)                   # [1, H, W], uint8
        mm = TF.resize(mm, (100, 100))               # [1, 100,100]
        # Convertir a etiquetas {0,1}
        mm = (mm > 0).long().squeeze(0)               # [100, 100], valores 0 ó 1
        mask_list.append(mm.unsqueeze(0))             # ahora [1, 100,100]

    # Concatenar todos en un solo tensor
    image_tensor = torch.cat(image_list, dim=0)       # [N, 3, 100,100]
    masks_tensor = torch.cat(mask_list, dim=0)        # [N, 1, 100,100] ó [N, 100,100] si haces squeeze

    N = image_tensor.size(0)
    # --- 2) Split aleatorio (80% train, 20% test) ---
    train_frac = 0.8
    N_train = int(train_frac * N)
    N_test  = N - N_train
    perm = torch.randperm(N)
    idx_train, idx_test = perm[:N_train], perm[N_train:]
    train_images = image_tensor[idx_train]            # [N_train, 3,100,100]
    train_masks  = masks_tensor[idx_train].squeeze(1) # [N_train,100,100]
    test_images  = image_tensor[idx_test]             # [N_test, 3,100,100]
    test_masks   = masks_tensor[idx_test].squeeze(1)  # [N_test, 100,100]
    print(f"N={N}  |  Train={train_images.size(0)}  |  Test={test_images.size(0)}")

    # --- 3) DataLoaders con TensorDataset ---
    batch_size = 16
    train_ds = TensorDataset(train_images, train_masks)
    test_ds  = TensorDataset(test_images,  test_masks)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- 4) Modelo + Optimizador + Scheduler + Crítico de pérdida ---
    unet = UNet(n_channels=3, n_classes=2).to(device)
    optim = Adam(unet.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = StepLR(optim, step_size=5, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    # Listas para almacenar métricas por época
    loss_list_train    = []
    iou_list_train     = []
    loss_list_test     = []
    iou_list_test      = []

    # --- 5) Bucle principal de entrenamiento + validación ---
    num_epochs = 30
    for epoch in range(num_epochs):
        # -------- ENTRENAMIENTO --------
        unet.train()
        running_loss_train = 0.0
        iou_accum_train    = []

        for imgs, masks in train_loader:
            imgs  = imgs.to(device)               # [B, 3,100,100]
            masks = masks.to(device)              # [B, 100,100] con valores 0 ó 1

            optim.zero_grad()
            preds = unet(imgs)                    # [B, 2,100,100]
            loss  = criterion(preds, masks)       # CrossEntropy espera [B,2,H,W] vs [B,H,W]
            loss.backward()
            optim.step()
            running_loss_train += loss.item() * imgs.size(0)

            # Cálculo de IoU (para la clase “1”) en este batch
            with torch.no_grad():
                pred_labels = torch.argmax(preds, dim=1)   # [B, 100,100]
                # Máscara de predicción y verdad donde hay “1” (inundación)
                pred_fg   = (pred_labels == 1)
                target_fg = (masks == 1)
                intersection = (pred_fg & target_fg).sum(dim=(1,2)).float()
                union        = (pred_fg | target_fg).sum(dim=(1,2)).float()
                # Evitar división por cero
                iou_batch = torch.where(union == 0,
                                        torch.ones_like(intersection),
                                        intersection / union)
                iou_accum_train.append(iou_batch.mean().item())

        epoch_loss_train = running_loss_train / len(train_ds)
        epoch_iou_train  = sum(iou_accum_train) / len(iou_accum_train)
        loss_list_train.append(epoch_loss_train)
        iou_list_train.append(epoch_iou_train)

        # -------- VALIDACIÓN / TEST --------
        unet.eval()
        running_loss_test = 0.0
        iou_accum_test    = []

        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs  = imgs.to(device)
                masks = masks.to(device)

                preds = unet(imgs)
                loss  = criterion(preds, masks)
                running_loss_test += loss.item() * imgs.size(0)

                pred_labels = torch.argmax(preds, dim=1)
                pred_fg   = (pred_labels == 1)
                target_fg = (masks == 1)
                intersection = (pred_fg & target_fg).sum(dim=(1,2)).float()
                union        = (pred_fg | target_fg).sum(dim=(1,2)).float()
                iou_batch    = torch.where(union == 0,
                                           torch.ones_like(intersection),
                                           intersection / union)
                iou_accum_test.append(iou_batch.mean().item())

        epoch_loss_test = running_loss_test / len(test_ds)
        epoch_iou_test  = sum(iou_accum_test) / len(iou_accum_test)
        loss_list_test.append(epoch_loss_test)
        iou_list_test.append(epoch_iou_test)

        # Ajustar LR si estás usando scheduler
        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}]  "
            f"Loss Train: {epoch_loss_train:.4f}  |  IoU Train: {epoch_iou_train:.4f}  ||  "
            f"Loss Test:  {epoch_loss_test:.4f}  |  IoU Test:  {epoch_iou_test:.4f}"
        )

    # --- 6) Graficar métricas (IoU vs Loss) ---
    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, iou_list_train, label='IoU Train')
    plt.plot(epochs, iou_list_test,  label='IoU Test')
    plt.title("IoU (Jaccard) Clase 1")
    plt.xlabel("Época")
    plt.ylabel("IoU")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss_list_train, label='Loss Train')
    plt.plot(epochs, loss_list_test,  label='Loss Test')
    plt.title("CrossEntropy Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("Metricas.jpg", format='jpg')
    plt.show()
