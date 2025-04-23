import os

def parse_trueeval_scores(log_path):
    scores = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "[TrueEval]" in line and "Eval Reward" in line:
                    try:
                        score = float(line.strip().split("Eval Reward:")[1].split()[0])
                        scores.append(score)
                    except:
                        continue
    except FileNotFoundError:
        pass
    return scores

def compute_scores(scores):
    if not scores:
        return None, None

    first_score = scores[0]
    avg = sum(scores) / len(scores)

    # Task 1: 正常公式
    task1_score = min(avg, 480) / 480 * 15

    # Task 2: 如果第一筆為負
    if first_score < 0:
        task2_score = (min(avg, 19) + 21) / 40 * 20
    else:
        task2_score = None

    return task1_score, task2_score

def find_best_task1_and_task2(root_path="a"):
    best_task1_score = -1
    best_task1_folder = ""
    best_task2_score = -100000000
    best_task2_folder = ""

    for folder in os.listdir(root_path):
        if folder == "latest-run" or folder.endswith(".log"):
            continue
        log_path = os.path.join(root_path, folder, "file", "output.log")
        scores = parse_trueeval_scores(log_path)

        task1_score, task2_score = compute_scores(scores)

        if task1_score is not None and task1_score > best_task1_score:
            best_task1_score = task1_score
            best_task1_folder = folder

        if task2_score is not None and task2_score > best_task2_score:
            best_task2_score = task2_score
            best_task2_folder = folder

        print(f"[{folder}] → Task1: {task1_score}, Task2: {task2_score}")

    print("\nfinal result")
    print(f"Task1 highest score:{best_task1_score:.2f}% in folder [{best_task1_folder}]")
    print(f"Task2 highest score:{best_task2_score:.2f}% in folder[{best_task2_folder}]")

if __name__ == "__main__":
    find_best_task1_and_task2("wandb")
