"""Giải nén data/raw/archive.zip và tạo data/processed/{train,val,test}/{input,target}.

input  = ảnh mờ (low_res), target = ảnh rõ (high_res), trùng tên file.
Chạy từ thư mục gốc project:  python scripts/prepare_data.py
"""
import random, re, zipfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
SEED = 42


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def extract():
    dst = RAW / "dataset"
    if not dst.exists():
        print("Đang giải nén archive.zip ...")
        with zipfile.ZipFile(RAW / "archive.zip") as z:
            z.extractall(dst)
    return dst


def find_split(base, split):
    for hr in base.rglob("high_res"):
        if hr.parent.name == split:
            return hr.parent / "low_res", hr
    raise FileNotFoundError(f"Không thấy thư mục '{split}/high_res' trong {base}")


def list_pairs(lr, hr):
    names = sorted((p.name for p in hr.glob("*.png")), key=natural_key)
    return [(lr / n, hr / n) for n in names if (lr / n).exists()]


def save(split, pairs):
    (OUT / split / "input").mkdir(parents=True, exist_ok=True)
    (OUT / split / "target").mkdir(parents=True, exist_ok=True)
    for i, (lr, hr) in enumerate(pairs):
        Image.open(lr).convert("RGB").save(OUT / split / "input" / f"{i:04d}.png")
        Image.open(hr).convert("RGB").save(OUT / split / "target" / f"{i:04d}.png")
    print(f"{split}: {len(pairs)} cặp")


def main():
    base = extract()
    train = list_pairs(*find_split(base, "train"))
    val_all = list_pairs(*find_split(base, "val"))   # bỏ qua 'Raw Data' vì trùng train+val
    random.Random(SEED).shuffle(val_all)
    half = len(val_all) // 2
    save("train", train)
    save("val", val_all[:half])
    save("test", val_all[half:])


if __name__ == "__main__":
    main()