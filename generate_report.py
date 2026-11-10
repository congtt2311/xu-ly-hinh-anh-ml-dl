import argparse

from config import (
    MANIFEST_PATH,
    NAFNET_CHECKPOINT_PATH,
    OUTPUT_PATH,
    REPORT_PATH,
    ROOT,
)
from src.report import generate_multipage_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate a multi-page HTML project report."
    )
    parser.add_argument(
        "--input",
        default=str(OUTPUT_PATH / "evaluation_results.csv"),
        help="Evaluation CSV file.",
    )
    parser.add_argument(
        "--output",
        default=str(REPORT_PATH),
        help="Directory for generated report pages.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="GoPro manifest; defaults to data/processed/gopro_manifest.csv.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="NAFNet checkpoint; defaults to the project GoPro NAFNet checkpoint.",
    )
    args = parser.parse_args()
    report_path = generate_multipage_report(
        args.input,
        args.output,
        manifest_path=args.manifest or MANIFEST_PATH,
        checkpoint_path=args.checkpoint or NAFNET_CHECKPOINT_PATH,
    )
    print(f"HTML report saved to: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
