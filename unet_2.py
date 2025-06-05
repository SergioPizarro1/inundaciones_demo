import torch
import torchvision
import torch.nn.functional as F
import PIL
import os

from matplotlib import pyplot
from torch.optim.lr_scheduler import StepLR

#C1
#JJJJ

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
    images = os.listdir('./flood/data/Image/')
    masks = os.listdir('./flood/data/Mask/')

    print(len(images), len(masks))

    image_tensor = list()
    masks_tensor = list()
    for image in images:
        dd = PIL.Image.open(f'./flood/data/Image/{image}')
        tt = torchvision.transforms.functional.pil_to_tensor(dd)
        tt = torchvision.transforms.functional.resize(tt, (100, 100))

        tt = tt[None, :, :, :]
        tt = torch.tensor(tt, dtype=torch.float) / 255.

        if tt.shape != (1, 3, 100, 100):
            continue

        image_tensor.append(tt)

        mask = image.replace('.jpg', '.png')
        dd = PIL.Image.open(f'./flood/data/Mask/{mask}')
        mm = torchvision.transforms.functional.pil_to_tensor(dd)
        mm = mm.repeat(3, 1, 1)
        mm = torchvision.transforms.functional.resize(mm, (100, 100))
        mm = mm[:1, :, :]

        mm = torch.tensor((mm > 0.).detach().numpy(), dtype=torch.long)
        mm = torch.nn.functional.one_hot(mm)
        mm = torch.permute(mm, (0, 3, 1, 2))
        mm = torch.tensor(mm, dtype=torch.float)

        masks_tensor.append(mm)

    image_tensor = torch.cat(image_tensor)
    print(image_tensor.shape)

    masks_tensor = torch.cat(masks_tensor)
    print(masks_tensor.shape)

    N = image_tensor.size(0)

    # --- (2) Decidir tamaño del split ---
    train_frac = 0.8
    N_train = int(train_frac * N)
    N_test  = N - N_train

    # --- (3) Generar índices aleatorios y dividirlos ---
    perm = torch.randperm(N)             # permuta las posiciones 0,1,...,N-1
    idx_train = perm[:N_train]
    idx_test  = perm[N_train:]

    # --- (4) Indexar image_tensor y masks_tensor con esos índices ---
    train_images = image_tensor[idx_train]   # shape: [N_train, 3, 100, 100]
    train_masks  = masks_tensor[idx_train]   # shape: [N_train, 2, 100, 100]
    test_images  = image_tensor[idx_test]    # shape: [N_test,  3, 100, 100]
    test_masks   = masks_tensor[idx_test]    # shape: [N_test,  2, 100, 100]

    print("N = ",image_tensor.size(0),"\n N_train = ",train_images.size(0),"\n N_test = ",test_images.size(0))



    unet = UNet(n_channels=3, n_classes=2).to(device)

    dataloader_train_image = torch.utils.data.DataLoader(train_images, batch_size=16, shuffle=False)
    dataloader_train_target = torch.utils.data.DataLoader(train_masks, batch_size=16, shuffle=False)

    dataloader_test_image = torch.utils.data.DataLoader(test_images, batch_size=16, shuffle=False)
    dataloader_test_target = torch.utils.data.DataLoader(test_masks, batch_size=16, shuffle=False)

    optim = torch.optim.Adam(unet.parameters(), lr=1e-5)
    cross_entropy = torch.nn.CrossEntropyLoss()

    loss_list_train = list()
    jaccard_list_train = list()

    loss_list_test = list()
    jaccard_list_test = list()

    for epoch in range(30):
        print("Epoch {} running".format(epoch))

        unet.train()
        running_loss = 0.
        jaccard_epoch_train = list()
        for image, target in zip(dataloader_train_image, dataloader_train_target):
            image=image.to(device)
            target=target.to(device)
            
            optim.zero_grad()
            pred = unet(image)

            loss = cross_entropy(pred, target)
            running_loss += loss.item()

            loss.backward()
            optim.step()

        for image, target in zip(dataloader_train_image, dataloader_train_target):
            image=image.to(device)
            target=target.to(device)

            pred = unet(image)

            _, pred_unflatten = torch.max(pred, dim=1)
            _, target_unflatten = torch.max(target, dim=1)

            intersection = torch.sum(pred_unflatten == target_unflatten, dim=(1, 2))/10000.

            jaccard_epoch_train.append(torch.mean(intersection).detach())

        jaccard = sum(jaccard_epoch_train)/len(jaccard_epoch_train)
        jaccard_list_train.append(jaccard.cpu().item())

        loss_list_train.append(running_loss)

        unet.eval()
        running_loss = 0.
        jaccard_epoch_test = list()
        with torch.no_grad():
            for image, target in zip(dataloader_test_image, dataloader_test_target):
                image=image.to(device)
                target=target.to(device)

                pred=unet(image)
                
                loss = cross_entropy(pred, target)
                running_loss += loss.item()

            for image, target in zip(dataloader_test_image, dataloader_test_target):
                image=image.to(device)
                target=target.to(device)

                pred=unet(image)

                _, pred_unflatten = torch.max(pred, dim=1)
                _, target_unflatten = torch.max(target, dim=1)

                intersection = torch.sum(pred_unflatten == target_unflatten, dim=(1, 2))/10000.

                jaccard_epoch_test.append(torch.mean(intersection).detach())
        
        jaccard = sum(jaccard_epoch_test)/len(jaccard_epoch_test)
        jaccard_list_test.append(jaccard.cpu().item())
        loss_list_test.append(running_loss)

    pyplot.clf()
    pyplot.subplot(1, 2, 1)
    pyplot.plot(jaccard_list_train, label='TRAIN')
    pyplot.plot(jaccard_list_test, label='TEST')
    pyplot.title("IOU")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("IOU")
    pyplot.legend()
    
    #pyplot.clf()
    pyplot.subplot(1, 2, 2)
    pyplot.plot(loss_list_train, label='TRAIN')
    pyplot.plot(loss_list_test, label='TEST')
    pyplot.title("Loss")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.legend()
    pyplot.savefig("Metricas.jpg", format='jpg')
    pyplot.show()