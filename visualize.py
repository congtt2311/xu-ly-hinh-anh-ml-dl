
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("outputs/evaluation_results.csv")
output_dir = Path("outputs")
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
plt.savefig(output_dir / "psnr_comparison.png", dpi=300)
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(df["Method"], df["SSIM"])
plt.title("SSIM Comparison")
plt.xlabel("Method")
plt.ylabel("SSIM")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(output_dir / "ssim_comparison.png", dpi=300)
plt.show()

print("Da tao xong 2 bieu do trong outputs!")
