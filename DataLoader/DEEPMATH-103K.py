import argparse
import json
import os
from datasets import load_dataset


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Download and save DeepMath-103K dataset")
	parser.add_argument(
		"--save-dir",
		default="~/MathRL/Data/DeepMath-103K",
		help="Save directory on host, e.g. ~/MathRL/Data/DeepMath-103K",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Overwrite existing directory contents",
	)
	args = parser.parse_args()
	save_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(args.save_dir)))

	dataset = load_dataset("zwhe99/DeepMath-103K")

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
			answers = [item.get("r1_solution_1"), item.get("r1_solution_2"), item.get("r1_solution_3")]
			answers = [text.strip() for text in answers if isinstance(text, str) and text.strip()]
			answers = sorted(set(answers), key=len)
			rows.append(
				{
					"question": item["question"],
					"answer": answers,
					"final_answer": item["final_answer"],
				}
			)
		with open(output_path, "w", encoding="utf-8") as f:
			json.dump(rows, f, ensure_ascii=False, indent=2)
