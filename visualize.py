
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import IMAGE_OUTPUT_PATH, OUTPUT_PATH
from src.report import generate_multipage_report


csv_path = Path("outputs/evaluation_results.csv")
output_dir = IMAGE_OUTPUT_PATH
output_dir.mkdir(parents=True, exist_ok=True)

if not csv_path.exists():
    raise FileNotFoundError("Khong tim thay evaluation_results.csv")

df = pd.read_csv(csv_path)

print("KET QUA DANH GIA")
print(df)

plt.figure(figsize=(9, 5))
plt.bar(df["Method"], df["PSNR"])
plt.title("PSNR Comparison")
plt.xlabel("Method")
plt.ylabel("PSNR (dB)")
plt.tight_layout()
psnr_path = output_dir / "psnr_comparison.png"
plt.savefig(psnr_path, dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(df["Method"], df["SSIM"])
plt.title("SSIM Comparison")
plt.xlabel("Method")
plt.ylabel("SSIM")
plt.ylim(0, 1)
plt.tight_layout()
ssim_path = output_dir / "ssim_comparison.png"
plt.savefig(ssim_path, dpi=300)
plt.close()

report_path = generate_multipage_report(csv_path)
print(f"Da tao bieu do: {psnr_path.name}, {ssim_path.name}")
print(f"Da tao bao cao HTML: {report_path.name}")
