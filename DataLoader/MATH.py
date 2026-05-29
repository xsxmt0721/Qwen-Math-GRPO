import argparse
import json
import os
import re
from datasets import load_dataset


def extract_final_answer(answer_text: str) -> str:
    markers = ["\\boxed{", "\\fbox{"]
    spans = []
    for marker in markers:
        start = 0
        while True:
            idx = answer_text.find(marker, start)
            if idx == -1:
                break
            i = idx + len(marker)
            depth = 1
            while i < len(answer_text) and depth > 0:
                if answer_text[i] == "{":
                    depth += 1
                elif answer_text[i] == "}":
                    depth -= 1
                i += 1
            if depth == 0:
                spans.append((idx, i))
            start = idx + 1
    if not spans:
        return ""
    spans.sort(key=lambda x: x[0])
    start, end = spans[-1]
    inner = answer_text[start:end]
    left = inner.find("{")
    right = inner.rfind("}")
    if left == -1 or right == -1 or right <= left:
        return ""
    return inner[left + 1 : right].strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save MATH dataset")
    parser.add_argument(
        "--save-dir",
        default="~/MathRL/Data/MATH",
        help="Save directory on host, e.g. ~/MathRL/Data/MATH",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing directory contents",
    )
    args = parser.parse_args()
    save_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(args.save_dir)))

    dataset = load_dataset("HuggingFaceH4/MATH")

    os.makedirs(save_dir, exist_ok=True)
    if os.path.isdir(save_dir) and os.listdir(save_dir):
        if not args.force:
            raise SystemExit("Target directory is not empty. Use --force to overwrite.")
        for root, dirs, files in os.walk(save_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    dataset.save_to_disk(save_dir)

    processed_dir = os.path.join(save_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    for split_name, split_data in dataset.items():
        output_path = os.path.join(processed_dir, f"{split_name}.json")
        rows = []
        for item in split_data:
            answer_text = item["solution"]
            rows.append(
                {
                    "question": item["problem"],
                    "answer": [answer_text],
                    "final_answer": extract_final_answer(answer_text),
                }
            )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
