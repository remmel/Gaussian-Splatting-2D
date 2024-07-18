import os
import shutil

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from tqdm import tqdm

from viewer import ImageViewerPlt


# Simpler version than run.py without pruning / cloning

class GS2D:
    def __init__(self, img_size=(256, 256, 3), num_samples=1000, device="cuda"):
        self.img_size = img_size
        self.num_samples = num_samples
        self.device = device

        h, w = self.img_size[:2]
        x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        self.x = x.to(self.device)
        self.y = y.to(self.device)

        self.viewer = ImageViewerPlt('Training Progress')

    def draw_gaussian(self, sigma, rho, mean, color, alpha):
        r = rho.view(-1, 1, 1)
        sx = sigma[:, :1, None]
        sy = sigma[:, -1:, None]
        dx = self.x.unsqueeze(0) - mean[:, 0].view(-1, 1, 1)
        dy = self.y.unsqueeze(0) - mean[:, 1].view(-1, 1, 1)
        v = torch.exp(-0.5 * (((sx * dx) ** 2 + (sy * dy) ** 2) - 2 * dx * dy * r * sy * sx) / (sy ** 2 * sx ** 2 * (1 - r ** 2) + 1e-8))
        img = torch.sum(v.unsqueeze(1) * color.view(-1, 3, 1, 1) * alpha.view(-1, 1, 1, 1), dim=0)
        assert img.shape == (3, 256, 256)
        return torch.clamp(img, 0, 1)


    def random_init_param(self):
        sigma = torch.rand(size = (self.num_samples, 2)) - 3 #[-3,-2]
        rho = torch.rand(size = (self.num_samples, 1)) * 2 #[0,2]
        mean = torch.atanh(torch.rand(size = (self.num_samples, 2))*2 - 1) #[-1,1]->[-inf,inf]
        color = torch.atanh(torch.rand(size = (self.num_samples, 3))) #[0,inf]
        alpha = torch.zeros(size = (self.num_samples, 1))-0.01
        w = torch.cat([sigma, rho, mean, color, alpha], dim =1).to(self.device)
        return nn.Parameter(w)

    def parse_param(self, w):
        size = torch.tensor(self.img_size[:2][::-1]).to(self.device)
        sigma = torch.sigmoid(w[:, :2]) * size * 0.25
        rho = torch.tanh(w[:, 2:3])
        mean = (0.5 * torch.tanh(w[:, 3:5]) + 0.5) * size #using sigmoid : center ok, corner bof
        color = 0.5 * torch.tanh(w[:, 5:8]) + 0.5 #using sigmoid: slighly better
        alpha = 0.5 * torch.tanh(w[:, 8:9]) + 0.5
        return sigma, rho, mean, color, alpha

    def train(self, target, num_epochs=10, lr=0.005):
        # w = nn.Parameter(torch.randn(self.num_samples, 9).to(self.device)) # does not work if that init used
        w = self.random_init_param()
        optimizer = torch.optim.Adam([w], lr=lr)

        for epoch in range(num_epochs):
            for _ in tqdm(range(30), desc=f"Epoch {epoch}"):
                optimizer.zero_grad()
                predicted = self.draw_gaussian(*self.parse_param(w))
                loss = nn.functional.l1_loss(predicted, target)
                loss.backward()
                optimizer.step()

            torchvision.utils.save_image(torch.stack([predicted, target]), f"images/epoch_{epoch}.jpg")
            if self.viewer.show_training_progress(predicted, target, epoch):
                print("Training stopped by user.")
                break

        return w


def main():
    shutil.rmtree("images", ignore_errors=True)
    os.makedirs("images")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])(Image.open("a.png").convert("RGB")).to(device)

    assert img.shape == (3, 256, 256)
    h, w = img.size()[1:]
    gs = GS2D(img_size=(h,w,3), device=device)
    gs.train(img)


if __name__ == "__main__":
    main()