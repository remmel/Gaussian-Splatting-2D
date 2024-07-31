import os
import shutil

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from tqdm import tqdm

from viewer import ImageViewerPlt


# Another version using ellipse equation instead of gaussian
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

    def draw(self, scale, angle, mean, color, alpha):
        sx = scale[:, :1, None]
        sy = scale[:, -1:, None]

        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        dx = self.x.unsqueeze(0) - mean[:, 0].view(-1, 1, 1) #mean[:, 0, None, None]
        dy = self.y.unsqueeze(0) - mean[:, 1].view(-1, 1, 1)

        x_rot = dx * cos_angle.view(-1, 1, 1) + dy * sin_angle.view(-1, 1, 1)
        y_rot = -dx * sin_angle.view(-1, 1, 1) + dy * cos_angle.view(-1, 1, 1)

        ellipse_eq = ((x_rot / sx.view(-1, 1, 1)) ** 2 + (y_rot / sy.view(-1, 1, 1)) ** 2)
        v = 1.0 - torch.sigmoid(ellipse_eq)
        # v = torch.exp(-ellipse_eq / 0.1)
        # v = vv <= 1
        img = torch.sum(v.unsqueeze(1).float() * color.view(-1, 3, 1, 1) * alpha.view(-1, 1, 1, 1), dim=0)
        assert img.shape == (3, 256, 256)
        return torch.clamp(img, 0, 1)


    def random_init_param(self):
        scale = torch.rand(size=(self.num_samples, 2))-3
        angle = torch.rand(size=(self.num_samples, 1)) * 2 -1  # Rotation angle
        mean = torch.atanh(torch.rand(size=(self.num_samples, 2)) * 2 - 1)  # Center
        color = torch.atanh(torch.rand(size=(self.num_samples, 3)))  # Color
        alpha = torch.zeros(size=(self.num_samples, 1)) - 0.01  # Alpha
        w = torch.cat([scale, angle, mean, color, alpha], dim=1).to(self.device)
        return nn.Parameter(w)

    def parse_param(self, w):
        size = torch.tensor(self.img_size[:2]).to(self.device)
        scale = torch.sigmoid(w[:, :2]) * size * 0.25
        angle = torch.tanh(w[:, 2:3])
        mean = (0.5 * torch.tanh(w[:, 3:5]) + 0.5) * size
        color = 0.5 * torch.tanh(w[:, 5:8]) + 0.5
        alpha = 0.5 * torch.tanh(w[:, 8:9]) + 0.5
        return scale, angle, mean, color, alpha

    def train(self, target, num_epochs=10, lr=0.005):
        # w = nn.Parameter(torch.randn(self.num_samples, 9).to(self.device)) # does not work if that init used
        w = self.random_init_param()
        optimizer = torch.optim.Adam([w], lr=lr)

        for epoch in range(num_epochs):
            for _ in tqdm(range(30), desc=f"Epoch {epoch}"):
                optimizer.zero_grad()
                predicted = self.draw(*self.parse_param(w))
                loss = nn.functional.l1_loss(predicted, target)
                loss.backward()
                optimizer.step()

            torchvision.utils.save_image(torch.stack([predicted, target]), f"images/epoch_{epoch}.jpg")

            if self.viewer.show_training_progress(predicted, target, epoch):
                print("Training stopped by user.")
                return w

        return w


def main_train():
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

def main_demo():
    device = 'cuda'
    gs = GS2D(img_size=(256, 256, 3), device=device)

    w = torch.tensor([
        # sx,  sy, rho, m_x, m_y, c_r, c_g, c_b,   a
        [-3., -0., -1, 0.0, 0.0, 1.0, 0.0, 0.0, .1],  # Red circle
        [-1., -3., .0, 0.0, 0.0, 0.0, 1.0, 0.0, .1],  # Green ellipse
        # [-3., -3., .75, 1., 1., 0.0, 0.0, 1.0, 1.0],  # Blue tilted ellipse
    ], device=device)

    scale, angle, mean, color, alpha = gs.parse_param(w)

    # Render the ellipses
    image = gs.draw(scale, angle, mean, color, alpha)
    gs.viewer.show_training_progress(image, image, 0, True)


if __name__ == "__main__":
    # main_train()
    main_demo()