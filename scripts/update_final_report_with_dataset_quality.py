import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_file(path: str) -> str:
    if not os.path.exists(path):
        logging.warning(f"File not found: {path}")
        return ""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to read {path}: {e}")
        return ""


def append_quality_to_report(quality_md: str, quality_csv: str, out_report: str):
    os.makedirs(os.path.dirname(out_report), exist_ok=True)

    # Build the block to append
    block_lines = []
    block_lines.append("\n\n## Dataset Quality Assessment\n\n")

    md_content = read_file(quality_md)
    if md_content:
        block_lines.append(md_content)
        if not md_content.endswith("\n"):
            block_lines.append("\n")
    else:
        block_lines.append("(quality markdown not found)\n")

    # Optionally include a short CSV reference
    if os.path.exists(quality_csv):
        block_lines.append("\nReference CSV: ")
        block_lines.append(f"`{quality_csv}`\n")
    else:
        block_lines.append("\nQuality CSV not found.\n")

    try:
        with open(out_report, 'a', encoding='utf-8') as f:
            f.writelines(block_lines)
        logging.info(f"Appended dataset-quality section to {out_report}")
    except Exception as e:
        logging.error(f"Failed to append to {out_report}: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Append dataset-quality summary into the final report")
    p.add_argument('--quality_md', required=True, help='Path to dataset_quality_assessment.md')
    p.add_argument('--quality_csv', required=True, help='Path to dataset_quality_assessment.csv')
    p.add_argument('--out_report', required=True, help='Path to final_summary_report.md')
    return p.parse_args()


def main():
    args = parse_args()
    append_quality_to_report(args.quality_md, args.quality_csv, args.out_report)


if __name__ == '__main__':
    main()