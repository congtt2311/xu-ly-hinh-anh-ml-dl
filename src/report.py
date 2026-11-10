"""Generate a small, linked HTML report website for the deblurring project."""

from __future__ import annotations

import csv
import html
import math
from datetime import date
from pathlib import Path

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MANIFEST_PATH,
    NUM_WORKERS,
    OUTPUT_PATH,
    REPORT_PATH,
    WIENER_SIZE,
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
    checkpoint_path: str | Path = CHECKPOINT_PATH / "best_model.pth",
) -> Path:
    """Generate an index and four linked reports; return the index path."""
    results_csv = Path(results_csv)
    report_directory = Path(report_directory)
    manifest_path = Path(manifest_path)
    checkpoint_path = Path(checkpoint_path)
    rows = _read_results(results_csv)
    split_counts, sequence_count, manifest_warning = _read_dataset_summary(manifest_path)
    total_pairs = sum(split_counts.values())
    checkpoint_exists = checkpoint_path.is_file()
    image_counts_present = all(row["Images"] for row in rows)

    report_directory.mkdir(parents=True, exist_ok=True)

    overview_body = f"""
  <header class="hero"><span class="eyebrow">PROJECT PRESENTATION · GOPRO</span>
    <h1>Hệ thống khử mờ và phục hồi ảnh</h1>
    <p>Báo cáo tổng quan dữ liệu, pipeline xử lý, cấu hình huấn luyện và kết quả đánh giá.</p>
    <div class="pills"><span class="pill">GoPro Large</span>
      <span class="pill">{len(rows)} phương pháp trong CSV</span>
      <span class="pill">PSNR · SSIM</span></div>
  </header>
  <section><h2>Tổng quan dự án</h2>
    <div class="grid">
      {_stat(str(total_pairs) if total_pairs else "—", "cặp ảnh trong manifest")}
      {_stat(str(split_counts.get("train", "—")), "cặp train")}
      {_stat(str(split_counts.get("valid", "—")), "cặp validation")}
      {_stat(str(split_counts.get("test", "—")), "cặp test")}
    </div>
    <p class="note {"ok" if checkpoint_exists else ""}">
      {"Đã tìm thấy checkpoint CNN." if checkpoint_exists else "Chưa tìm thấy checkpoint CNN; kết quả CSV có thể là thí nghiệm đã lưu trước đó."}
      Báo cáo không tự suy diễn trạng thái huấn luyện từ số liệu đánh giá.
    </p>
  </section>
  <section><h2>Các phần trình bày</h2><div class="grid">
    <a class="link-card" href="01_data_report.html"><b>01 · Dữ liệu</b><span>GoPro, ghép cặp ảnh và chia tập.</span></a>
    <a class="link-card" href="02_feature_pipeline_report.html"><b>02 · Feature pipeline</b><span>Từ ảnh mờ đến ảnh phục hồi.</span></a>
    <a class="link-card" href="03_model_training_report.html"><b>03 · Huấn luyện</b><span>Mô hình, loss, optimizer và checkpoint.</span></a>
    <a class="link-card" href="04_model_evaluation_report.html"><b>04 · Đánh giá</b><span>PSNR, SSIM và so sánh phương pháp.</span></a>
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
    <p>Dữ liệu ảnh mờ chuyển động và ảnh sharp tham chiếu dùng cho bài toán image deblurring.</p>
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
    <p>Train và validation được tách theo sequence trong manifest; tập test giữ riêng để đánh giá.</p>
    <p class="muted">Các số lượng trên được đọc trực tiếp từ manifest khi tạo báo cáo.</p>
  </section>
"""

    pipeline_body = """
  <header class="hero"><span class="eyebrow">02 · FEATURE PIPELINE REPORT</span>
    <h1>Pipeline xử lý ảnh</h1>
    <p>Luồng xử lý kết hợp bộ lọc Wiener làm đầu vào tham khảo và CNN dự đoán phần hiệu chỉnh.</p>
  </header>
  <section><h2>Luồng dữ liệu</h2><div class="steps">
    <div class="step"><b>1. Đọc cặp ảnh</b> — ảnh RGB blur và sharp được chuẩn hóa về [0, 1].</div>
    <div class="step"><b>2. Crop train</b> — lấy crop ngẫu nhiên đồng bộ trên ảnh blur/sharp; validation và test dùng center crop hoặc toàn ảnh theo cấu hình.</div>
    <div class="step"><b>3. Wiener Filter</b> — lọc từng kênh để tạo ảnh tham khảo bổ sung.</div>
    <div class="step"><b>4. CNN residual</b> — CNN nhận ảnh blur và Wiener, dự đoán phần hiệu chỉnh cộng vào ảnh blur.</div>
    <div class="step"><b>5. Đánh giá/lưu</b> — đầu ra được giới hạn trong [0, 1], tính PSNR và SSIM so với sharp.</div>
  </div></section>
  <section><h2>Đầu vào và đầu ra mô hình</h2>
    <p><code>input = concat(blur, wiener)</code> có 6 kênh; CNN dự đoán correction 3 kênh.</p>
    <pre>restored = clamp(blur + CNN(concat(blur, wiener)), 0, 1)</pre>
    <p class="muted">Cài đặt tương ứng nằm trong <code>src/dataset.py</code>, <code>src/wiener.py</code> và <code>src/model.py</code>.</p>
  </section>
"""

    checkpoint_text = (
        f"Checkpoint hiện có: {checkpoint_path} ({checkpoint_path.stat().st_size:,} bytes)."
        if checkpoint_exists
        else f"Chưa có checkpoint tại {checkpoint_path}; chưa thể xác nhận mô hình đã được huấn luyện trong workspace này."
    )
    training_body = f"""
  <header class="hero"><span class="eyebrow">03 · MODEL TRAINING REPORT</span>
    <h1>Huấn luyện mô hình</h1>
    <p>Cấu hình mặc định trong mã nguồn và trạng thái checkpoint tại thời điểm tạo báo cáo.</p>
  </header>
  <section><h2>Cấu hình mặc định</h2>
    <div class="grid">
      {_stat(str(EPOCHS), "epochs")}
      {_stat(str(BATCH_SIZE), "batch size")}
      {_stat(str(IMAGE_SIZE) + " × " + str(IMAGE_SIZE), "crop train")}
      {_stat(f"{LEARNING_RATE:g}", "learning rate")}
      {_stat("Adam", "optimizer")}
      {_stat(f"{WIENER_SIZE} × {WIENER_SIZE}", "Wiener window")}
    </div>
    <p>Loss huấn luyện: L1. Scheduler: ReduceLROnPlateau theo validation loss. Seed mặc định của lệnh train: 42.</p>
  </section>
  <section><h2>Checkpoint</h2>
    <p class="note {"ok" if checkpoint_exists else ""}">{html.escape(checkpoint_text)}</p>
    <p>Checkpoint dự kiến được lưu tại <code>checkpoints/best_model.pth</code> và <code>checkpoints/last_model.pth</code>.</p>
  </section>
  <section><h2>Lệnh chạy</h2>
    <pre>python train.py --epochs 5 --batch-size 1 --patch-size 128
python evaluate.py --limit 0</pre>
    <p class="muted">Huấn luyện toàn bộ dataset có thể cần thời gian và tài nguyên đáng kể; lệnh mẫu trên là cấu hình demo, không phải kết quả đã chạy.</p>
  </section>
"""

    result_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['Method'])}</td>"
        f"<td>{html.escape(row['Images']) if row['Images'] else '—'}</td>"
        f"<td>{row['PSNR']}</td><td>{row['SSIM']}</td></tr>"
        for row in rows
    )
    evaluation_body = f"""
  <header class="hero"><span class="eyebrow">04 · MODEL EVALUATION REPORT</span>
    <h1>Kết quả đánh giá</h1>
    <p>So sánh chất lượng ảnh theo PSNR và SSIM. Giá trị cao hơn thường cho thấy kết quả tốt hơn.</p>
  </header>
  <section><h2>Bảng kết quả</h2>
    <div class="table-wrap"><table><thead><tr><th>Phương pháp</th><th>Số ảnh</th><th>PSNR (dB) ↑</th><th>SSIM ↑</th></tr></thead>
      <tbody>{result_rows}</tbody>
    </table></div>
    <p class="note">Nguồn: <code>{html.escape(results_csv.name)}</code>. {"CSV hiện không có cột số lượng ảnh, vì vậy số lượng được ghi là chưa cung cấp." if not image_counts_present else "Số ảnh được đọc từ CSV."}</p>
  </section>
  <section><h2>PSNR</h2>{_chart(rows, "PSNR")}</section>
  <section><h2>SSIM</h2>{_chart(rows, "SSIM")}</section>
  <section><h2>Diễn giải</h2>
    <p>Blur là mốc ảnh đầu vào; Wiener là baseline cổ điển; CNN + Wiener là phương pháp học sâu kết hợp trong bảng CSV.</p>
    <p class="muted">Báo cáo trình bày số liệu hiện có, không tự chạy lại benchmark hay giả định checkpoint đã tồn tại.</p>
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
