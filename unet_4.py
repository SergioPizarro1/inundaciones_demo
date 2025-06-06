import torch
import torchvision
import torch.nn.functional as F
import PIL
import os

from matplotlib import pyplot
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

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


class FloodDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}\n")

if __name__ == '__main__':
    images = os.listdir('./flood/data/Image/')
    masks = os.listdir('./flood/data/Mask/')

    print(len(images), len(masks))

    image_tensor = []
    masks_tensor = []
    for image in images:
        # Imagen
        dd = PIL.Image.open(f'./flood/data/Image/{image}')
        tt = torchvision.transforms.functional.pil_to_tensor(dd)
        tt = torchvision.transforms.functional.resize(tt, (100, 100))
        tt = tt[None, :, :, :]
        tt = torch.tensor(tt, dtype=torch.float) / 255.
        if tt.shape != (1, 3, 100, 100):
            continue
        image_tensor.append(tt)

        # Máscara
        mask = image.replace('.jpg', '.png')
        dd = PIL.Image.open(f'./flood/data/Mask/{mask}')
        mm = torchvision.transforms.functional.pil_to_tensor(dd)
        mm = torchvision.transforms.functional.resize(mm, (100, 100))
        mm = mm.squeeze(0)  # [100, 100]
        mm = (mm > 0).long()  # Binariza
        masks_tensor.append(mm.unsqueeze(0))

    image_tensor = torch.cat(image_tensor)
    masks_tensor = torch.cat(masks_tensor)

    print("Imagenes:", image_tensor.shape)
    print("Máscaras:", masks_tensor.shape)

    N = image_tensor.size(0)
    train_frac = 0.8
    N_train = int(train_frac * N)
    N_test = N - N_train

    perm = torch.randperm(N)
    idx_train = perm[:N_train]
    idx_test = perm[N_train:]

    train_images = image_tensor[idx_train]
    train_masks = masks_tensor[idx_train].squeeze(1)
    test_images = image_tensor[idx_test]
    test_masks = masks_tensor[idx_test].squeeze(1)

    # Dataset y Dataloaders
    train_dataset = FloodDataset(train_images, train_masks)
    test_dataset = FloodDataset(test_images, test_masks)

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

    # Modelo y entrenamiento
    unet = UNet(n_channels=3, n_classes=2).to(device)
    optim = torch.optim.Adam(unet.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_list_train, jaccard_list_train = [], []
    loss_list_test, jaccard_list_test = [], []

    for epoch in range(20):
        print(f"Epoch {epoch} running")

        unet.train()
        running_loss = 0.
        jaccard_epoch_train = []

        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)

            optim.zero_grad()
            pred = unet(img)
            loss = loss_fn(pred, mask)
            running_loss += loss.item()
            loss.backward()
            optim.step()

            pred_classes = pred.argmax(dim=1)
            intersection = (pred_classes == mask).sum(dim=(1, 2)) / (100 * 100)
            jaccard_epoch_train.append(torch.mean(intersection).cpu().item())

        loss_list_train.append((running_loss) / len(train_loader))
        jaccard_list_train.append(sum(jaccard_epoch_train) / len(jaccard_epoch_train))

        unet.eval()
        running_loss = 0. 
        jaccard_epoch_test = []
        with torch.no_grad():
            for img, mask in test_loader:
                img, mask = img.to(device), mask.to(device)
                pred = unet(img)
                loss = loss_fn(pred, mask)
                running_loss += loss.item()

                pred_classes = pred.argmax(dim=1)
                intersection = (pred_classes == mask).sum(dim=(1, 2)) / (100 * 100)
                jaccard_epoch_test.append(torch.mean(intersection).cpu().item())

        loss_list_test.append((running_loss) / len(test_loader))
        jaccard_list_test.append(sum(jaccard_epoch_test) / len(jaccard_epoch_test))

    # Plots
    pyplot.clf()
    pyplot.subplot(1, 2, 1)
    pyplot.plot(jaccard_list_train, label='TRAIN')
    pyplot.plot(jaccard_list_test, label='TEST')
    pyplot.title("IOU")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("IOU")
    pyplot.legend()

    pyplot.subplot(1, 2, 2)
    pyplot.plot(loss_list_train, label='TRAIN')
    pyplot.plot(loss_list_test, label='TEST')
    pyplot.title("Loss")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.legend()
    pyplot.savefig("Metricas.jpg", format='jpg')
    pyplot.show()

    import random
    import torchvision.transforms.functional as TF

    def plot_random_sample_prediction(model, dataset, device, title_prefix, filename):
        model.eval()
        idx = random.randint(0, len(dataset) - 1)
        image, mask = dataset[idx]  # elemento aleatorio

        image_input = image.unsqueeze(0).to(device)  # [1, 3, H, W]

        with torch.no_grad():
            pred = model(image_input)
            pred_class = pred.argmax(dim=1).squeeze(0).cpu()  # [H, W]

        # Preparar para visualización
        image_np = image.permute(1, 2, 0).cpu().numpy()        # [H, W, 3]
        mask_np = mask.cpu().numpy()                          # [H, W]
        pred_np = pred_class.numpy()                          # [H, W]

        # Dibujar y guardar figura
        fig, axs = pyplot.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image_np)
        axs[0].set_title(f'{title_prefix} Image')
        axs[0].axis('off')

        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title(f'{title_prefix} Mask')
        axs[1].axis('off')

        axs[2].imshow(pred_np, cmap='gray')
        axs[2].set_title(f'{title_prefix} Prediction')
        axs[2].axis('off')

        pyplot.tight_layout()
        pyplot.savefig(filename, format='jpg')
        pyplot.close()

    # 🔹 Imagen aleatoria de entrenamiento
    plot_random_sample_prediction(unet, train_dataset, device, "Train", "train_prediction.jpg")

    # 🔸 Imagen aleatoria de test
    plot_random_sample_prediction(unet, test_dataset, device, "Test", "test_prediction.jpg")

