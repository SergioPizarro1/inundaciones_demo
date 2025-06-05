import torch
import torchvision
import torch.nn.functional as F
import PIL
import os

from matplotlib import pyplot
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# ... (módulos del modelo UNet definidos igual que antes) ...


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

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Modelo y entrenamiento
    unet = UNet(n_channels=3, n_classes=2).to(device)
    optim = torch.optim.Adam(unet.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_list_train, jaccard_list_train = [], []
    loss_list_test, jaccard_list_test = [], []

    for epoch in range(30):
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

        loss_list_train.append(running_loss)
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

        loss_list_test.append(running_loss)
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
