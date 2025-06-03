import os
import torch
import torchvision
import torch.nn.functional as F
import PIL.Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, random_split, DataLoader
from matplotlib import pyplot as plt

# -------------------
# 1) DEFINICIÓN DEL UNET
# -------------------
class DoubleConv(torch.nn.Module):
    """(conv => [BN] => ReLU) * 2"""
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

class Down(torch.nn.Module):
    """Downscaling con MaxPool luego DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(torch.nn.Module):
    """Upscaling luego DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ajustar padding si hace falta
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenar
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(torch.nn.Module):
    """Última capa 1×1"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear

        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor     = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1   = Up(1024, 512 // factor, bilinear)
        self.up2   = Up(512, 256 // factor, bilinear)
        self.up3   = Up(256, 128 // factor, bilinear)
        self.up4   = Up(128, 64, bilinear)
        self.outc  = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x, x3)
        x  = self.up3(x, x2)
        x  = self.up4(x, x1)
        return self.outc(x)

# -------------------
# 2) PARÁMETROS Y DEVICE
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}\n")

images_dir = './flood/data/Image/'
masks_dir  = './flood/data/Mask/'

# Hiperparámetros sugeridos (puedes ajustar)
batch_size   = 32
num_epochs   = 20
learning_rate= 1e-3
step_size    = 5    # cada 5 épocas reduce lr
gamma        = 0.5  # factor de reducción

# -------------------
# 3) CARGA DE IMÁGENES Y MÁSCARAS A MEMORIA
# -------------------
# → Leemos todas las imágenes, las resizeamos a 100×100, normalizamos y apilamos en un solo Tensor
# → Lo mismo con las máscaras, pero generando one-hot (2 clases: fondo=0 e inundación=1)

image_list = []
mask_list  = []

all_images = sorted(os.listdir(images_dir))
print(f"Total de imágenes en carpeta: {len(all_images)}")

for filename in all_images:
    # 1) Cargar imagen
    img_path = os.path.join(images_dir, filename)
    img_pil  = PIL.Image.open(img_path).convert("RGB")
    img_tens = torchvision.transforms.functional.pil_to_tensor(img_pil)  # [3,H,W], uint8
    img_tens = torchvision.transforms.functional.resize(img_tens, (100, 100))
    img_tens = img_tens.float() / 255.0  # [3,100,100], float

    # Añadimos la dimensión de batch: [1,3,100,100]
    img_tens = img_tens.unsqueeze(0)
    image_list.append(img_tens)

    # 2) Cargar máscara emparejada
    # Asumimos que si la imagen se llama "0001.jpg", la máscara es "0001.png"
    base, _   = os.path.splitext(filename)
    mask_name = base + '.png'
    mask_path = os.path.join(masks_dir, mask_name)
    if not os.path.isfile(mask_path):
        # Si no existe la máscara, la saltamos
        continue

    mask_pil  = PIL.Image.open(mask_path).convert("L")  # en escala de grises
    mask_tens = torchvision.transforms.functional.pil_to_tensor(mask_pil)  # [1,H,W], uint8
    mask_tens = torchvision.transforms.functional.resize(mask_tens, (100, 100))
    mask_tens = (mask_tens > 0).long().squeeze(0)  # [100,100], valores 0 o 1

    # One-hot para 2 clases: resultado [2,100,100]
    mask_onehot = F.one_hot(mask_tens, num_classes=2)      # [100,100,2]
    mask_onehot = mask_onehot.permute(2, 0, 1).float()      # [2,100,100]
    mask_onehot = mask_onehot.unsqueeze(0)                  # [1,2,100,100]
    mask_list.append(mask_onehot)

# Concatenamos todos en un solo tensor:
#  → image_tensor: [N, 3, 100, 100]
#  → masks_tensor: [N, 2, 100, 100]
image_tensor = torch.cat(image_list, dim=0)
masks_tensor = torch.cat(mask_list, dim=0)

print(f"Tensor de imágenes:  {image_tensor.shape}")
print(f"Tensor de máscaras:   {masks_tensor.shape}\n")

# -------------------
# 4) CREAR DATASET Y SPLIT (train/test)
# -------------------
# Construimos un TensorDataset para mantener sincronizados image y mask
dataset = TensorDataset(image_tensor, masks_tensor)

N_total   = len(dataset)
N_train   = int(0.8 * N_total)       # 80% → train
N_test    = N_total - N_train        # 20% → test

# Para reproducibilidad:
g = torch.Generator().manual_seed(42)

train_ds, test_ds = random_split(dataset, [N_train, N_test], generator=g)
print(f"Tamaño total dataset: {N_total}")
print(f"  → Train: {len(train_ds)}  |  Test: {len(test_ds)}\n")

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# -------------------
# 5) INSTANCIAR UNET, OPTIMIZADOR, CRITERIO, SCHEDULER
# -------------------
unet       = UNet(n_channels=3, n_classes=2, bilinear=False).to(device)
optimizer  = torch.optim.Adam(unet.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion  = torch.nn.CrossEntropyLoss()
scheduler  = StepLR(optimizer, step_size=step_size, gamma=gamma)

# Listas para almacenar métricas por época
loss_train_list = []
loss_test_list  = []
iou_train_list  = []
iou_test_list   = []

# -------------------
# 6) BUCLE DE ENTRENAMIENTO Y VALIDACIÓN
# -------------------
for epoch in range(num_epochs):
    unet.train()
    running_loss_train = 0.0
    iou_accum_train    = []

    # → Entrenamiento
    for batch_imgs, batch_masks_onehot in train_loader:
        batch_imgs = batch_imgs.to(device)                   # [B,3,100,100]
        batch_masks_onehot = batch_masks_onehot.to(device)   # [B,2,100,100]
        # CrossEntropyLoss espera etiquetas como [B,H,W] con clases (0/1), no one-hot
        batch_masks_labels = torch.argmax(batch_masks_onehot, dim=1)  # [B,100,100]

        optimizer.zero_grad()
        preds = unet(batch_imgs)                              # [B,2,100,100]
        loss  = criterion(preds, batch_masks_labels)
        loss.backward()
        optimizer.step()

        running_loss_train += loss.item() * batch_imgs.size(0)

        # Cálculo de IoU (Jaccard) en train para este batch
        with torch.no_grad():
            _, preds_labels = torch.max(preds, dim=1)                 # [B,100,100]
            # IoU para la clase "1" (inundación)
            preds_fg   = (preds_labels == 1)
            target_fg  = (batch_masks_labels == 1)
            intersection = torch.logical_and(preds_fg, target_fg).sum(dim=(1,2)).float()
            union        = torch.logical_or(preds_fg, target_fg).sum(dim=(1,2)).float()
            # Para evitar división por cero:
            iou_batch = torch.where(union == 0, torch.ones_like(intersection), intersection / union)
            iou_accum_train.append(iou_batch.mean().item())

    epoch_loss_train = running_loss_train / len(train_loader.dataset)
    epoch_iou_train  = float(torch.tensor(iou_accum_train).mean())
    loss_train_list.append(epoch_loss_train)
    iou_train_list.append(epoch_iou_train)

    # → Validación / Test
    unet.eval()
    running_loss_test = 0.0
    iou_accum_test    = []

    with torch.no_grad():
        for batch_imgs, batch_masks_onehot in test_loader:
            batch_imgs = batch_imgs.to(device)
            batch_masks_onehot = batch_masks_onehot.to(device)
            batch_masks_labels = torch.argmax(batch_masks_onehot, dim=1)

            preds = unet(batch_imgs)
            loss  = criterion(preds, batch_masks_labels)
            running_loss_test += loss.item() * batch_imgs.size(0)

            # IoU en test para este batch
            _, preds_labels = torch.max(preds, dim=1)
            preds_fg  = (preds_labels == 1)
            target_fg = (batch_masks_labels == 1)
            intersection = torch.logical_and(preds_fg, target_fg).sum(dim=(1,2)).float()
            union        = torch.logical_or(preds_fg, target_fg).sum(dim=(1,2)).float()
            iou_batch    = torch.where(union == 0, torch.ones_like(intersection), intersection / union)
            iou_accum_test.append(iou_batch.mean().item())

    epoch_loss_test = running_loss_test / len(test_loader.dataset)
    epoch_iou_test  = float(torch.tensor(iou_accum_test).mean())
    loss_test_list.append(epoch_loss_test)
    iou_test_list.append(epoch_iou_test)

    # Reducir lr si aplica
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}]  "
          f"Loss Train: {epoch_loss_train:.4f}  |  IoU Train: {epoch_iou_train:.4f}  ||  "
          f"Loss Test: {epoch_loss_test:.4f}  |  IoU Test: {epoch_iou_test:.4f}")

# -------------------
# 7) GRAFICAR PÉRDIDA E IOU (TRAIN vs TEST)
# -------------------
epochs = list(range(1, num_epochs + 1))

plt.figure(figsize=(10,4))
# — Pérdida
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_train_list, label="Loss Train")
plt.plot(epochs, loss_test_list,  label="Loss Test")
plt.xlabel("Época")
plt.ylabel("CrossEntropy Loss")
plt.title("Pérdida Train vs Test")
plt.legend()

# — IoU
plt.subplot(1, 2, 2)
plt.plot(epochs, iou_train_list, label="IoU Train")
plt.plot(epochs, iou_test_list,  label="IoU Test")
plt.xlabel("Época")
plt.ylabel("IoU (Jaccard) Clase 1")
plt.title("IoU Train vs Test")
plt.legend()

plt.tight_layout()
plt.savefig("metrics_comparativas.jpg", dpi=200)
plt.show()
