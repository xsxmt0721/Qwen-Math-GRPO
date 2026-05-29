import argparse
import json
import os
from datasets import load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save GSM8K dataset")
    parser.add_argument(
        "--save-dir",
        default="~/MathRL/Data/GSM8K",
        help="Save directory on host, e.g. ~/MathRL/Data/GSM8K",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing directory contents",
    )
    args = parser.parse_args()
    save_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(args.save_dir)))

    dataset = load_dataset("openai/gsm8k", "main")

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

    def extract_final_answer(answer_text: str) -> str:
        marker = "####"
        if marker in answer_text:
            return answer_text.split(marker, 1)[1].strip()
        return answer_text.strip()

    def normalize_answer(answer_text: str, final_answer: str) -> str:
        marker = "####"
        if marker in answer_text:
            prefix = answer_text.split(marker, 1)[0].rstrip()
            return f"{prefix}\nSo the answer is \\box{{{final_answer}}}"
        return f"{answer_text.rstrip()}\nSo the answer is \\box{{{final_answer}}}"

    for split_name, split_data in dataset.items():
        output_path = os.path.join(processed_dir, f"{split_name}.json")
        rows = []
        for item in split_data:
            final_answer = extract_final_answer(item["answer"])
            rows.append(
                {
                    "question": item["question"],
                    "answer": [normalize_answer(item["answer"], final_answer)],
                    "final_answer": final_answer,
                }
            )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
