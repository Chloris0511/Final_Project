import csv
from pathlib import Path

def log_val_result(
    csv_path,
    experiment,
    ablation_mode,
    hidden_dim,
    lr,
    epoch,
    val_acc,
    val_macro_f1
):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "experiment",
                "ablation_mode",
                "hidden_dim",
                "lr",
                "epoch",
                "val_acc",
                "val_macro_f1"
            ])

        writer.writerow([
            experiment,
            ablation_mode,
            hidden_dim,
            lr,
            epoch,
            f"{val_acc:.4f}",
            f"{val_macro_f1:.4f}"
        ])
