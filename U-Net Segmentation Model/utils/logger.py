import os
import json
import matplotlib.pyplot as plt


def save_json_log(data: dict, filename: str, save_dir="outputs/logs"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def plot_metric_curve(values, metric_name="mIoU", save_dir="outputs/logs"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(values, marker='o')
    plt.title(f"{metric_name} per trial")
    plt.xlabel("Trial")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{metric_name}_per_trial.png"))
    plt.close()
