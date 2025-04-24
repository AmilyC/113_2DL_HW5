# run_dqn_experiments.py
import subprocess

episodes_list = [200000, 400000, 600000, 800000, 1000000, 1100000]

for ep in episodes_list:
    run_name = f"cartpole-ep{ep}"
    dir_name = f"./cartpole-ep{ep}"
    cmd = [
        "python", "dqn-2.py",  # <-- 把這裡換成你的主程式檔名
        "--wandb-run-name", run_name,
        "--save-dir",dir_name,
        "--episode", str(ep),
        "--batch-size", "32",
        "--memory-size", "100000",
        "--lr", "0.0001",
        "--discount-factor", "0.99",
        "--epsilon-start", "1.0",
        "--epsilon-decay", "0.999999",
        "--epsilon-min", "0.05",
        "--target-update-frequency", "1000",
        "--replay-start-size", "10000",
        "--max-episode-steps", "10000",
        "--train-per-step", "1",
        "--episode", str(ep)
    ]
    print(f"Running experiment: {run_name}")
    subprocess.run(cmd)