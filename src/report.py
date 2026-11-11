"""Generate a small, linked HTML report website for the deblurring project."""

from __future__ import annotations

import csv
import html
import math
from datetime import date
from pathlib import Path

from config import (
    DEBLURGAN_CHECKPOINT_PATH,
    MANIFEST_PATH,
    NAFNET_CHECKPOINT_PATH,
    OUTPUT_PATH,
    REPORT_PATH,
)


ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    ("index.html", "Tổng quan"),
    ("01_data_report.html", "Dữ liệu"),
    ("02_feature_pipeline_report.html", "Pipeline"),
    ("03_model_training_report.html", "Huấn luyện"),
    ("04_model_evaluation_report.html", "Đánh giá"),
)

STYLES = """
:root{color-scheme:light;--ink:#172554;--muted:#64748b;--blue:#2563eb;--line:#dbe4f0;--bg:#f3f6fb;--green:#059669;--amber:#b45309}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:#1e293b;font:15px/1.65 "Segoe UI",Arial,sans-serif}
.layout{max-width:1120px;margin:26px auto;padding:0 20px}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-bottom:18px}
.brand{color:var(--ink);font-weight:800;text-decoration:none}
nav{display:flex;flex-wrap:wrap;gap:7px}
nav a{padding:7px 10px;border-radius:8px;color:#475569;text-decoration:none;font-size:13px}
nav a:hover,nav a.active{background:#dbeafe;color:#1d4ed8}
.hero{padding:36px;border-radius:18px;color:white;background:linear-gradient(120deg,#102a56,#2563eb)}
.eyebrow{color:#bfdbfe;font-size:11px;font-weight:800;letter-spacing:.14em}
h1{margin:8px 0;font-size:clamp(28px,4vw,40px);line-height:1.15}
.hero p{max-width:780px;margin:0;color:#dbeafe}
.pills{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.pill{padding:5px 11px;border:1px solid #ffffff55;border-radius:99px;font-size:13px}
section,.panel{margin-top:18px;padding:22px;border:1px solid var(--line);border-radius:14px;background:white;box-shadow:0 8px 26px #0f172a0a}
h2{margin:0 0 12px;color:var(--ink);font-size:21px}
h3{margin:0 0 8px;color:var(--ink);font-size:17px}
p{margin:8px 0}
.muted,.caption{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.stat,.link-card{padding:17px;border:1px solid var(--line);border-radius:12px;background:#f8fafc}
.stat strong{display:block;color:var(--ink);font-size:25px}
.stat span{color:var(--muted);font-size:13px}
.link-card{display:block;color:inherit;text-decoration:none}
.link-card:hover{border-color:#93c5fd;background:#eff6ff}
.link-card b{display:block;margin-bottom:4px;color:#1d4ed8}
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;text-align:left}
th,td{padding:11px 13px;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:#f8fafc;color:#334155}
.note{margin-top:14px;padding:13px 15px;border-left:4px solid #f59e0b;background:#fffbeb}
.ok{border-left-color:var(--green);background:#ecfdf5}
.steps{display:grid;gap:10px}
.step{padding:14px 16px;border:1px solid var(--line);border-radius:11px;background:#f8fafc}
.step b{color:#1d4ed8}
code,pre{font-family:Consolas,"Courier New",monospace}
pre{overflow-x:auto;padding:14px;border-radius:10px;background:#0f172a;color:#e2e8f0}
.chart{display:block;width:100%;max-width:760px}
.chart-label{fill:#334155;font-size:14px}
.chart-value{fill:#0f172a;font-size:13px;font-weight:600}
.bar-track{fill:#e8eef8}
.bar{fill:var(--blue)}
.footer{padding:20px 4px;color:var(--muted);font-size:12px}
@media(max-width:700px){.layout{margin:14px auto;padding:0 12px}.topbar{align-items:flex-start;flex-direction:column}.hero{padding:25px 21px}section,.panel{padding:17px}}
@media print{body{background:white}.layout{margin:0 auto}section,.panel{break-inside:avoid;box-shadow:none}}
"""


def _read_results(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required_fields = {"Method", "PSNR", "SSIM"}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(
                f"Results CSV must contain these columns: "
                f"{', '.join(sorted(required_fields))}"
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                psnr = float(row["PSNR"])
                ssim = float(row["SSIM"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid PSNR or SSIM on CSV line {line_number}"
                ) from exc
            if not row["Method"] or not math.isfinite(psnr) or not math.isfinite(ssim):
                raise ValueError(f"Invalid evaluation result on CSV line {line_number}")
            if not 0 <= ssim <= 1:
                raise ValueError(f"SSIM must be between 0 and 1 on line {line_number}")
            rows.append(
                {
                    "Method": row["Method"],
                    "Images": row.get("Images", ""),
                    "PSNR": f"{psnr:.4f}",
                    "SSIM": f"{ssim:.4f}",
                }
            )

    if not rows:
        raise ValueError(f"No evaluation results found in {csv_path}")
    return rows


def _read_dataset_summary(manifest_path: Path) -> tuple[dict[str, int], int, str | None]:
    if not manifest_path.is_file():
        return {}, 0, f"Không tìm thấy manifest: {manifest_path}"

    counts: dict[str, int] = {}
    sequences: set[str] = set()
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_fields = {"split", "sequence", "blur", "sharp"}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected GoPro manifest columns in {manifest_path}")
        for line_number, row in enumerate(reader, start=2):
            split = row["split"]
            if split not in {"train", "valid", "test"}:
                raise ValueError(f"Invalid dataset split on manifest line {line_number}")
            if not row["blur"] or not row["sharp"]:
                raise ValueError(f"Empty image path on manifest line {line_number}")
            counts[split] = counts.get(split, 0) + 1
            sequences.add(f"{split}:{row['sequence']}")
    return counts, len(sequences), None


def _chart(rows: list[dict[str, str]], metric: str) -> str:
    width = 720
    label_width = 170
    bar_width = 420
    row_height = 46
    height = 28 + row_height * len(rows)
    values = [float(row[metric]) for row in rows]
    scale_max = max(values) * 1.12 if metric == "PSNR" else 1.0
    scale_max = max(scale_max, 1e-9)
    elements = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{html.escape(metric)} comparison chart">'
    ]
    for index, row in enumerate(rows):
        y = 12 + index * row_height
        value = float(row[metric])
        bar_length = max(2, bar_width * value / scale_max)
        label = html.escape(row["Method"])
        value_label = f"{value:.2f} dB" if metric == "PSNR" else f"{value:.3f}"
        elements.append(
            f'<text x="0" y="{y + 21}" class="chart-label">{label}</text>'
            f'<rect x="{label_width}" y="{y}" width="{bar_width}" height="28" '
            'rx="7" class="bar-track"/>'
            f'<rect x="{label_width}" y="{y}" width="{bar_length:.1f}" '
            f'height="28" rx="7" class="bar"/>'
            f'<text x="{label_width + bar_width + 12}" y="{y + 20}" '
            f'class="chart-value">{value_label}</text>'
        )
    elements.append("</svg>")
    return "\n".join(elements)


def _shell(title: str, active_page: str, body: str) -> str:
    navigation = "\n".join(
        f'<a href="{filename}" class="{"active" if filename == active_page else ""}">'
        f"{html.escape(label)}</a>"
        for filename, label in PAGES
    )
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} | Hệ thống khử mờ ảnh</title>
  <style>{STYLES}</style>
</head>
<body>
<main class="layout">
  <div class="topbar"><a class="brand" href="index.html">GOPRO · DEBLUR REPORT</a>
    <nav aria-label="Điều hướng báo cáo">{navigation}</nav></div>
{body.strip()}
  <footer class="footer">Báo cáo dự án xử lý ảnh · Tạo ngày {date.today().strftime("%d/%m/%Y")}</footer>
</main>
</body>
</html>
"""


def _stat(value: str, label: str) -> str:
    return (
        f'<div class="stat"><strong>{html.escape(value)}</strong>'
        f"<span>{html.escape(label)}</span></div>"
    )


def generate_multipage_report(
    results_csv: str | Path = OUTPUT_PATH / "evaluation_results.csv",
    report_directory: str | Path = REPORT_PATH,
    manifest_path: str | Path = MANIFEST_PATH,
    checkpoint_path: str | Path = NAFNET_CHECKPOINT_PATH,
) -> Path:
    """Generate an index and four linked reports; return the index path."""
    results_csv = Path(results_csv)
    report_directory = Path(report_directory)
    manifest_path = Path(manifest_path)
    checkpoint_path = Path(checkpoint_path)
    rows = _read_results(results_csv)
    split_counts, sequence_count, manifest_warning = _read_dataset_summary(manifest_path)
    total_pairs = sum(split_counts.values())
    nafnet_checkpoint_exists = checkpoint_path.is_file()
    deblurgan_checkpoint_exists = DEBLURGAN_CHECKPOINT_PATH.is_file()
    image_counts_present = all(row["Images"] for row in rows)

    report_directory.mkdir(parents=True, exist_ok=True)

    overview_body = f"""
  <header class="hero"><span class="eyebrow">MÔ HÌNH CHÍNH · DEEP LEARNING</span>
    <h1>NAFNet — khôi phục ảnh mờ GoPro</h1>
    <p>NAFNet là mô hình trọng tâm của đồ án. Wiener Filter và DeblurGAN được sử dụng làm hai mốc so sánh.</p>
    <div class="pills"><span class="pill">GoPro Large</span>
      <span class="pill">NAFNet · mô hình chính</span>
      <span class="pill">Wiener + DeblurGAN · baseline</span>
      <span class="pill">PSNR · SSIM</span></div>
  </header>
  <section><h2>Vai trò mô hình</h2>
    <div class="grid">
      <div class="stat"><strong>NAFNet</strong><span>mô hình chính cần huấn luyện và đánh giá</span></div>
      <div class="stat"><strong>Wiener</strong><span>baseline xử lý ảnh cổ điển</span></div>
      <div class="stat"><strong>DeblurGAN</strong><span>baseline học sâu để so sánh</span></div>
    </div>
    <p class="note {"ok" if nafnet_checkpoint_exists else ""}">
      {"Đã tìm thấy checkpoint NAFNet." if nafnet_checkpoint_exists else "Chưa có checkpoint NAFNet; báo cáo không xem kết quả của CNN + Wiener là kết quả NAFNet."}
    </p>
  </section>
  <section><h2>Dữ liệu dự án</h2>
    <div class="grid">
      {_stat(str(total_pairs) if total_pairs else "—", "cặp ảnh trong manifest")}
      {_stat(str(split_counts.get("train", "—")), "cặp train")}
      {_stat(str(split_counts.get("valid", "—")), "cặp validation")}
      {_stat(str(split_counts.get("test", "—")), "cặp test")}
    </div>
  </section>
  <section><h2>Các phần trình bày</h2><div class="grid">
    <a class="link-card" href="01_data_report.html"><b>01 · Dữ liệu</b><span>GoPro, ghép cặp ảnh và chia tập.</span></a>
    <a class="link-card" href="02_feature_pipeline_report.html"><b>02 · NAFNet pipeline</b><span>Luồng xử lý của mô hình chính.</span></a>
    <a class="link-card" href="03_model_training_report.html"><b>03 · Huấn luyện NAFNet</b><span>Kiến trúc, cấu hình và checkpoint.</span></a>
    <a class="link-card" href="04_model_evaluation_report.html"><b>04 · Đánh giá</b><span>NAFNet so với Wiener và DeblurGAN.</span></a>
  </div></section>
"""

    split_rows = "\n".join(
        f"<tr><td>{label}</td><td>{split_counts.get(key, 0):,}</td></tr>"
        for key, label in (("train", "Train"), ("valid", "Validation"), ("test", "Test"))
    )
    data_warning = (
        f'<p class="note">{html.escape(manifest_warning)}</p>'
        if manifest_warning
        else f'<p class="note ok">Manifest có {sequence_count} sequence (tính riêng theo từng split); '
        "các ảnh blur/sharp được ghép theo cùng tên tệp.</p>"
    )
    data_body = f"""
  <header class="hero"><span class="eyebrow">01 · DATA REPORT</span>
    <h1>Dữ liệu GoPro Large</h1>
    <p>Dữ liệu GoPro dùng để huấn luyện và đánh giá NAFNet; hai thuật toán còn lại là các mốc tham chiếu.</p>
  </header>
  <section><h2>Phân chia dữ liệu</h2>
    <div class="table-wrap"><table><thead><tr><th>Tập dữ liệu</th><th>Số cặp ảnh</th></tr></thead>
      <tbody>{split_rows}<tr><th>Tổng</th><th>{total_pairs:,}</th></tr></tbody>
    </table></div>{data_warning}
    <p class="muted">Nguồn manifest: <code>{html.escape(str(manifest_path))}</code>.</p>
  </section>
  <section><h2>Cấu trúc mỗi cặp</h2><div class="steps">
    <div class="step"><b>Blur</b> — ảnh đầu vào bị nhòe chuyển động.</div>
    <div class="step"><b>Sharp</b> — ảnh mục tiêu tương ứng, dùng làm ground truth.</div>
    <div class="step"><b>Blur gamma</b> — biến thể ảnh mờ được lưu trong dataset; pipeline hiện tại dùng thư mục blur.</div>
  </div></section>
  <section><h2>Quy tắc sử dụng</h2>
    <p>Train và validation được tách theo sequence trong manifest; tập test giữ riêng để đánh giá NAFNet và các baseline.</p>
    <p class="muted">Các số lượng trên được đọc trực tiếp từ manifest khi tạo báo cáo.</p>
  </section>
"""

    pipeline_body = """
  <header class="hero"><span class="eyebrow">02 · NAFNET PIPELINE</span>
    <h1>Pipeline mô hình chính: NAFNet</h1>
    <p>NAFNet học ánh xạ trực tiếp từ ảnh blur sang ảnh sharp. Wiener Filter và DeblurGAN được đánh giá riêng làm baseline.</p>
  </header>
  <section><h2>Luồng dữ liệu</h2><div class="steps">
    <div class="step"><b>1. Đọc cặp ảnh</b> — ảnh RGB blur/sharp được chuẩn hóa về [0, 1].</div>
    <div class="step"><b>2. Crop train</b> — crop ngẫu nhiên 256×256 đồng bộ cho ảnh blur/sharp; validation dùng toàn ảnh.</div>
    <div class="step"><b>3. NAFNet</b> — encoder-decoder có NAFBlock, SimpleGate và simplified channel attention dự đoán ảnh khôi phục.</div>
    <div class="step"><b>4. Padding/crop</b> — kích thước được đệm chia hết cho 8 khi cần; đầu ra được cắt về kích thước ban đầu.</div>
    <div class="step"><b>5. Đánh giá</b> — so ảnh NAFNet với sharp bằng L1, PSNR và các metric trong benchmark.</div>
  </div></section>
  <section><h2>Đầu vào và đầu ra mô hình</h2>
    <pre>restored = clamp(NAFNet(blur), 0, 1)</pre>
    <p>Wiener Filter và DeblurGAN không nằm trong đường truyền NAFNet; mỗi baseline xử lý ảnh blur độc lập để so sánh công bằng trên cùng tập test.</p>
    <p class="muted">Cài đặt NAFNet: <code>scripts/train_nafnet.py</code>; đánh giá so sánh: <code>scripts/evaluate_deblurring.py</code>.</p>
  </section>
"""

    checkpoint_text = (
        f"Đã tìm thấy checkpoint NAFNet: {checkpoint_path} ({checkpoint_path.stat().st_size:,} bytes)."
        if nafnet_checkpoint_exists
        else f"Chưa có checkpoint NAFNet tại {checkpoint_path}; mô hình chính chưa được xác nhận đã huấn luyện."
    )
    training_body = f"""
  <header class="hero"><span class="eyebrow">03 · MODEL TRAINING REPORT</span>
    <h1>Huấn luyện mô hình chính: NAFNet</h1>
    <p>Cấu hình dưới đây lấy từ lệnh huấn luyện NAFNet của dự án; Wiener và DeblurGAN là mô hình so sánh, không phải mô hình chính.</p>
  </header>
  <section><h2>Kiến trúc NAFNet</h2>
    <div class="grid">
      {_stat("Width 32", "số kênh nền")}
      {_stat("12", "middle NAF blocks")}
      {_stat("2, 2, 4, 8", "encoder blocks")}
      {_stat("2, 2, 2, 2", "decoder blocks")}
      {_stat("SimpleGate", "gating")}
      {_stat("SCA", "channel attention")}
    </div>
  </section>
  <section><h2>Cấu hình huấn luyện trong mã nguồn</h2>
    <div class="grid">
      {_stat("100", "epochs mặc định")}
      {_stat("1", "batch size mặc định")}
      {_stat("256 × 256", "crop train")}
      {_stat("0.001", "learning rate")}
      {_stat("AdamW", "optimizer")}
      {_stat("L1", "loss")}
    </div>
    <p>Seed mặc định: 42. Checkpoint tốt nhất được chọn theo validation L1.</p>
  </section>
  <section><h2>Checkpoint</h2>
    <p class="note {"ok" if nafnet_checkpoint_exists else ""}">{html.escape(checkpoint_text)}</p>
    <p>Đường dẫn mặc định: <code>weights/NAFNet-GoPro-width32.pth</code>.</p>
  </section>
  <section><h2>Lệnh chạy</h2>
    <pre>python scripts/train_nafnet.py --epochs 100 --device auto
python scripts/evaluate_deblurring.py --method both</pre>
    <p class="muted">Đây là lệnh hướng dẫn; báo cáo không khẳng định huấn luyện đã chạy nếu không tìm thấy checkpoint.</p>
  </section>
"""

    def metric_cells(row: dict[str, str] | None) -> str:
        if row is None:
            return "<td>—</td><td>Chưa đánh giá</td><td>Chưa đánh giá</td>"
        image_count = html.escape(row["Images"]) if row["Images"] else "—"
        return (
            f"<td>{image_count}</td><td>{row['PSNR']}</td><td>{row['SSIM']}</td>"
        )

    nafnet_row = next(
        (row for row in rows if "nafnet" in row["Method"].casefold()),
        None,
    )
    wiener_row = next(
        (row for row in rows if "wiener" in row["Method"].casefold()),
        None,
    )
    deblurgan_row = next(
        (row for row in rows if "deblurgan" in row["Method"].casefold()),
        None,
    )
    nafnet_status = (
        "Có checkpoint · chưa đánh giá"
        if nafnet_checkpoint_exists and nafnet_row is None
        else "Đã đánh giá"
        if nafnet_row is not None
        else "Chưa có checkpoint"
    )
    deblurgan_status = (
        "Có checkpoint · chưa đánh giá"
        if deblurgan_checkpoint_exists and deblurgan_row is None
        else "Đã đánh giá"
        if deblurgan_row is not None
        else "Chưa có checkpoint"
    )
    model_rows = "\n".join(
        [
            "<tr><th>NAFNet (mô hình chính)</th>"
            f"<td>{nafnet_status}</td>{metric_cells(nafnet_row)}</tr>",
            "<tr><td>Wiener Filter (baseline cổ điển)</td>"
            f"<td>{'Baseline' if wiener_row else 'Chưa có số liệu'}</td>"
            f"{metric_cells(wiener_row)}</tr>",
            "<tr><td>DeblurGAN (baseline học sâu)</td>"
            f"<td>{deblurgan_status}</td>{metric_cells(deblurgan_row)}</tr>",
        ]
    )

    result_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['Method'])}</td>"
        f"<td>{html.escape(row['Images']) if row['Images'] else '—'}</td>"
        f"<td>{row['PSNR']}</td><td>{row['SSIM']}</td></tr>"
        for row in rows
    )
    evaluation_body = f"""
  <header class="hero"><span class="eyebrow">04 · MODEL EVALUATION REPORT</span>
    <h1>Đánh giá NAFNet và hai mốc so sánh</h1>
    <p>NAFNet là mô hình chính; Wiener Filter và DeblurGAN là hai baseline. Chỉ số chỉ được hiển thị khi có kết quả đúng với phương pháp đó.</p>
  </header>
  <section><h2>So sánh các mô hình của đề tài</h2>
    <div class="table-wrap"><table><thead><tr><th>Mô hình / vai trò</th><th>Trạng thái</th><th>Số ảnh</th><th>PSNR (dB) ↑</th><th>SSIM ↑</th></tr></thead>
      <tbody>{model_rows}</tbody>
    </table></div>
    <p class="note">CSV hiện chưa có kết quả NAFNet hoặc DeblurGAN. Các ô được ghi “Chưa đánh giá”, không thay bằng số liệu của mô hình khác.</p>
  </section>
  <section><h2>Số liệu CSV hiện có</h2>
    <p class="muted">Bảng dưới đây giữ nguyên kết quả nguồn. Dòng CNN + Wiener là một thí nghiệm riêng, không phải NAFNet và không được dùng làm kết quả mô hình chính.</p>
    <div class="table-wrap"><table><thead><tr><th>Phương pháp</th><th>Số ảnh</th><th>PSNR (dB) ↑</th><th>SSIM ↑</th></tr></thead>
      <tbody>{result_rows}</tbody>
    </table></div>
    <p class="note">Nguồn: <code>{html.escape(results_csv.name)}</code>. {"CSV hiện không có cột số lượng ảnh, vì vậy số lượng được ghi là chưa cung cấp." if not image_counts_present else "Số ảnh được đọc từ CSV."}</p>
  </section>
  <section><h2>Biểu đồ số liệu CSV hiện có — PSNR</h2>{_chart(rows, "PSNR")}</section>
  <section><h2>Biểu đồ số liệu CSV hiện có — SSIM</h2>{_chart(rows, "SSIM")}</section>
  <section><h2>Diễn giải</h2>
    <p>NAFNet cần được huấn luyện và đánh giá trên tập GoPro test. Wiener Filter cung cấp baseline cổ điển; DeblurGAN cung cấp baseline học sâu.</p>
    <p class="muted">Biểu đồ CSV có thể chứa kết quả CNN + Wiener từ thí nghiệm cũ. Không diễn giải dòng này thành NAFNet hoặc DeblurGAN.</p>
  </section>
"""

    contents = {
        "index.html": ("Tổng quan", overview_body),
        "01_data_report.html": ("Báo cáo dữ liệu", data_body),
        "02_feature_pipeline_report.html": ("Báo cáo feature pipeline", pipeline_body),
        "03_model_training_report.html": ("Báo cáo huấn luyện mô hình", training_body),
        "04_model_evaluation_report.html": ("Báo cáo đánh giá mô hình", evaluation_body),
    }
    for filename, (title, body) in contents.items():
        (report_directory / filename).write_text(
            _shell(title, filename, body),
            encoding="utf-8",
        )
    return report_directory / "index.html"
