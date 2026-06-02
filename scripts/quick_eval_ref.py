from ref.eval import extract_boxed_anwser
import json

data_dir = "/workspace/logs/ref_logs/test/predictions.jsonl"

if __name__ == "__main__":
    correct = 0
    total = 0
    with open(data_dir, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred = item["prediction"]
            ans = extract_boxed_anwser(pred)
            true_ans = item["gold_answer"]
            if ans == true_ans:
                correct += 1
            total += 1
    print(f"Accuracy: {correct/total:.4f} ({correct}/{total})")