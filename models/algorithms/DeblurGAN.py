"""DeblurGAN (Kupyn et al., CVPR 2018) cho bộ GoPro.

Cài đặt theo paper: generator ResNet 9 block + global skip, critic PatchGAN,
loss = WGAN-GP + 100 * perceptual loss (VGG19 conv3_3).

Chạy từ thư mục gốc project:
    python models/algorithms/DeblurGAN.py train --epochs 300
    python models/algorithms/DeblurGAN.py eval  --ckpt models/saved_models/deblurgan/G_latest.pth
    python models/algorithms/DeblurGAN.py predict --ckpt <file.pth> --input anh_mo.png --output anh_net.png
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
GOPRO = ROOT / "data" / "raw" / "GOPRO_Large"
SAVE_DIR = ROOT / "models" / "saved_models" / "deblurgan"
SEED = 42


# ----------------------------------------------------------------------------
# Dữ liệu: cặp (blur, sharp) của GoPro, ảnh chuẩn hoá về [-1, 1]
# ----------------------------------------------------------------------------
def list_pairs(scene_dirs, blur_dir="blur"):
    pairs = []
    for scene in scene_dirs:
        for sharp in sorted((scene / "sharp").glob("*.png")):
            blur = scene / blur_dir / sharp.name
            if blur.exists():
                pairs.append((blur, sharp))
    return pairs


def split_scenes(n_val_scenes=2):
    """Tách validation từ train theo scene để không rò rỉ khung hình liền nhau."""
    train_scenes = sorted(p for p in (GOPRO / "train").iterdir() if p.is_dir())
    test_scenes = sorted(p for p in (GOPRO / "test").iterdir() if p.is_dir())
    shuffled = train_scenes[:]
    random.Random(SEED).shuffle(shuffled)
    val_scenes = sorted(shuffled[:n_val_scenes])
    train_scenes = sorted(shuffled[n_val_scenes:])
    return train_scenes, val_scenes, test_scenes


def to_tensor(img):
    return TF.to_tensor(img) * 2 - 1  # [0,1] -> [-1,1]


def to_uint8(t):
    arr = (t.clamp(-1, 1) + 1) / 2 * 255
    return arr.permute(1, 2, 0).cpu().numpy().round().astype(np.uint8)


class GoProDataset(Dataset):
    def __init__(self, pairs, crop=256, scale=1.0, train=True):
        self.pairs, self.crop, self.scale, self.train = pairs, crop, scale, train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        blur = Image.open(self.pairs[i][0]).convert("RGB")
        sharp = Image.open(self.pairs[i][1]).convert("RGB")
        if self.scale != 1.0:  # paper (WILD) thu nhỏ ảnh GoPro 2 lần trước khi crop
            size = (round(blur.width * self.scale), round(blur.height * self.scale))
            blur = blur.resize(size, Image.BICUBIC)
            sharp = sharp.resize(size, Image.BICUBIC)
        if self.train:
            top = random.randint(0, blur.height - self.crop)
            left = random.randint(0, blur.width - self.crop)
            blur = TF.crop(blur, top, left, self.crop, self.crop)
            sharp = TF.crop(sharp, top, left, self.crop, self.crop)
            if random.random() < 0.5:
                blur, sharp = TF.hflip(blur), TF.hflip(sharp)
        return to_tensor(blur), to_tensor(sharp)


# ----------------------------------------------------------------------------
# Generator: conv 7x7 -> 2 conv stride 2 -> 9 ResBlock -> 2 transposed conv -> conv 7x7
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, ngf=64, n_blocks=9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        ch = ngf
        for _ in range(2):  # downsample
            layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
            ch *= 2
        layers += [ResBlock(ch) for _ in range(n_blocks)]
        for _ in range(2):  # upsample
            layers += [nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.clamp(x + self.net(x), -1, 1)  # global skip: I_S = I_B + I_R


# ----------------------------------------------------------------------------
# Critic: PatchGAN, không sigmoid (WGAN), không điều kiện theo ảnh mờ
# ----------------------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(3, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ch = ndf
        for i in range(1, n_layers + 1):
            out = ndf * min(2 ** i, 8)
            stride = 2 if i < n_layers else 1
            layers += [nn.Conv2d(ch, out, 4, stride=stride, padding=1), nn.InstanceNorm2d(out), nn.LeakyReLU(0.2, True)]
            ch = out
        layers += [nn.Conv2d(ch, 1, 4, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    """MSE giữa feature map VGG19 conv3_3 (features[:15]) của ảnh khử mờ và ảnh nét."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:15]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, fake, real):
        norm = lambda x: ((x + 1) / 2 - self.mean) / self.std
        return F.mse_loss(self.vgg(norm(fake)), self.vgg(norm(real)))


def gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    grad = torch.autograd.grad(critic(mixed).sum(), mixed, create_graph=True)[0]
    return ((grad.view(grad.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()


# ----------------------------------------------------------------------------
# Train / Eval / Predict
# ----------------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    random.seed(SEED)
    train_scenes, val_scenes, _ = split_scenes(args.val_scenes)
    train_pairs = list_pairs(train_scenes, args.blur_dir)
    if args.max_images:
        train_pairs = random.Random(SEED).sample(train_pairs, min(args.max_images, len(train_pairs)))
    print(f"Train: {len(train_pairs)} cặp | Val scenes: {[s.name for s in val_scenes]} | device: {device}")

    loader = DataLoader(GoProDataset(train_pairs, args.crop, args.scale), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, drop_last=True)
    G, D = Generator().to(device), Critic().to(device)
    content = PerceptualLoss().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # lr giữ nguyên nửa đầu, giảm tuyến tính về 0 ở nửa sau
    half = args.epochs // 2
    decay = lambda e: 1.0 if e < half else max(0.0, 1 - (e - half) / max(1, args.epochs - half))
    sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, decay)
    sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, decay)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        G.train(); D.train()
        t0 = time.time()
        for step, (blur, sharp) in enumerate(loader):
            blur, sharp = blur.to(device), sharp.to(device)
            fake = G(blur)

            # 1) Cập nhật critic n_critic lần (WGAN-GP)
            for _ in range(args.n_critic):
                opt_D.zero_grad()
                loss_D = (D(fake.detach()).mean() - D(sharp).mean()
                          + args.gp_weight * gradient_penalty(D, sharp, fake.detach(), device))
                loss_D.backward()
                opt_D.step()

            # 2) Cập nhật generator: -D(G(x)) + lambda * perceptual
            opt_G.zero_grad()
            loss_adv = -D(fake).mean()
            loss_content = content(fake, sharp)
            loss_G = loss_adv + args.content_weight * loss_content
            loss_G.backward()
            opt_G.step()

            if step % 100 == 0:
                print(f"epoch {epoch + 1}/{args.epochs} step {step}/{len(loader)} "
                      f"loss_D {loss_D.item():.4f} loss_adv {loss_adv.item():.4f} "
                      f"loss_content {loss_content.item():.4f}")
        sch_G.step(); sch_D.step()
        torch.save(G.state_dict(), SAVE_DIR / "G_latest.pth")
        torch.save(D.state_dict(), SAVE_DIR / "D_latest.pth")
        if (epoch + 1) % args.val_every == 0:
            psnr, ssim, _ = evaluate(G, list_pairs(val_scenes, args.blur_dir)[:args.val_images], device)
            print(f"[val] epoch {epoch + 1}: PSNR {psnr:.2f} dB | SSIM {ssim:.4f} | {time.time() - t0:.0f}s")


def load_generator(ckpt, device):
    G = Generator().to(device)
    G.load_state_dict(torch.load(ckpt, map_location=device))
    G.eval()
    for m in G.modules():  # paper giữ dropout bật khi test (theo pix2pix)
        if isinstance(m, nn.Dropout):
            m.train()
    return G


@torch.no_grad()
def deblur(G, img, device):
    """img: PIL RGB -> PIL RGB. Pad về bội số của 4 rồi cắt lại kích thước gốc."""
    x = to_tensor(img).unsqueeze(0).to(device)
    h, w = x.shape[2:]
    ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    y = G(x)[:, :, :h, :w]
    return Image.fromarray(to_uint8(y[0]))


def evaluate(G, pairs, device):
    psnrs, ssims, times = [], [], []
    for blur_path, sharp_path in pairs:
        blur = Image.open(blur_path).convert("RGB")
        sharp = np.array(Image.open(sharp_path).convert("RGB"))
        t0 = time.time()
        out = np.array(deblur(G, blur, device))
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        psnrs.append(peak_signal_noise_ratio(sharp, out, data_range=255))
        ssims.append(structural_similarity(sharp, out, channel_axis=2, data_range=255))
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(times))


def run_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = load_generator(args.ckpt, device)
    _, _, test_scenes = split_scenes(args.val_scenes)
    pairs = list_pairs(test_scenes, args.blur_dir)
    if args.max_images:
        pairs = pairs[:args.max_images]
    psnr, ssim, t = evaluate(G, pairs, device)
    print(f"GoPro test ({len(pairs)} ảnh): PSNR {psnr:.2f} dB | SSIM {ssim:.4f} | {t:.3f} s/ảnh")


def run_predict(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = load_generator(args.ckpt, device)
    deblur(G, Image.open(args.input).convert("RGB"), device).save(args.output)
    print(f"Đã lưu: {args.output}")


def main():
    p = argparse.ArgumentParser(description="DeblurGAN trên GoPro")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval", "predict"):
        s = sub.add_parser(name)
        s.add_argument("--blur-dir", default="blur", choices=["blur", "blur_gamma"])
        s.add_argument("--val-scenes", type=int, default=2)
        if name != "train":
            s.add_argument("--ckpt", required=True)
        if name in ("train", "eval"):
            s.add_argument("--max-images", type=int, default=0, help="0 = dùng hết")
    t = sub.choices["train"]
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--batch-size", type=int, default=1)
    t.add_argument("--crop", type=int, default=256)
    t.add_argument("--scale", type=float, default=1.0, help="0.5 = thu nhỏ 2 lần như DeblurGAN-WILD")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--n-critic", type=int, default=5)
    t.add_argument("--gp-weight", type=float, default=10.0)
    t.add_argument("--content-weight", type=float, default=100.0)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--val-every", type=int, default=5)
    t.add_argument("--val-images", type=int, default=50)
    pr = sub.choices["predict"]
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", required=True)

    args = p.parse_args()
    {"train": train, "eval": run_eval, "predict": run_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
