import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


METHODS = [
    ("model_greedy", "Model Greedy", "tab:blue"),
    ("model_best_k", "Model Best-K", "tab:green"),
    ("heuristic_greedy", "Heuristic Greedy", "tab:orange"),
    ("concorde", "Concorde", "tab:red"),
]


def load_hist_data(hist_dir):
    files = []
    for fname in sorted(os.listdir(hist_dir)):
        if fname.endswith(".npz"):
            files.append(os.path.join(hist_dir, fname))
    return files


def normalize_str(val):
    if isinstance(val, np.ndarray) and val.shape == ():
        return str(val.item())
    return str(val)


def extract_meta(payload):
    dataset = normalize_str(payload.get("dataset", ""))
    checkpoint = normalize_str(payload.get("checkpoint", ""))
    selection_mode = normalize_str(payload.get("selection_mode", ""))
    metric = normalize_str(payload.get("metric", "euclidean"))
    best_k = int(payload.get("best_k", 0))
    max_instances = int(payload.get("max_instances", -1))
    checkpoint_dir = os.path.basename(os.path.dirname(checkpoint)) if checkpoint else ""
    return {
        "dataset": dataset,
        "dataset_base": os.path.splitext(os.path.basename(dataset))[0],
        "checkpoint": checkpoint,
        "checkpoint_base": os.path.splitext(os.path.basename(checkpoint))[0],
        "checkpoint_dir": checkpoint_dir,
        "selection_mode": selection_mode,
        "metric": metric,
        "best_k": best_k,
        "max_instances": max_instances,
    }


def parse_dataset_label(dataset_base):
    base = dataset_base
    if base.startswith("tsp_"):
        base = base[4:]
    parts = base.split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        instances = int(parts[-1])
        nodes = int(parts[-2])
        mode = "_".join(parts[:-2]) or "uniform"
        return mode, nodes, instances
    return base, None, None


def format_context_title(dataset_base, metric, selection_mode=None, max_instances=None):
    mode, nodes, instances = parse_dataset_label(dataset_base)
    mode_label = "" if mode in ("", "uniform") else mode.replace("_", " ")
    parts = []
    if mode_label:
        parts.append(mode_label)
    if nodes is not None and instances is not None:
        if max_instances is not None and max_instances > 0:
            parts.append(f"N={nodes} M={max_instances}")
        else:
            parts.append(f"N={nodes} M={instances}")
    parts.append(metric)
    if selection_mode:
        parts.append(f"selected={selection_mode}")
    return " | ".join(parts)


def plot_overlay_hist(out_path, series, title, xlabel="Tour length"):
    plt.figure(figsize=(7, 5))
    all_vals = [
        np.asarray(values)
        for _, values, _ in series
        if values is not None and len(values) > 0
    ]
    bins = None
    if all_vals:
        bins = np.histogram_bin_edges(np.concatenate(all_vals), bins=30)
    for label, values, color in series:
        if values is None or len(values) == 0:
            continue
        plt.hist(values, bins=bins, density=True, alpha=0.5, label=label, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_avg_length_bars(payloads, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    grouped = {}
    for payload in payloads:
        meta = extract_meta(payload)
        key = (meta["dataset_base"], meta["metric"], meta["checkpoint_dir"])
        if key not in grouped:
            grouped[key] = {"greedy": None, "bestk": None, "heuristic": [], "concorde": [], "max_instances": []}
        if meta["selection_mode"] == "greedy":
            grouped[key]["greedy"] = payload.get("model_greedy")
        elif meta["selection_mode"] == "bestk":
            grouped[key]["bestk"] = payload.get("model_best_k")

        heuristic = payload.get("heuristic_greedy")
        if heuristic is not None:
            grouped[key]["heuristic"].append(heuristic)

        concorde = payload.get("concorde")
        if concorde is not None and np.asarray(concorde).size > 0:
            grouped[key]["concorde"].append(concorde)
        grouped[key]["max_instances"].append(meta["max_instances"])

    for (dataset_base, metric, checkpoint_dir), data in grouped.items():
        labels = []
        means = []
        colors = []
        if data["greedy"] is not None:
            labels.append("Model Greedy")
            means.append(float(np.mean(np.asarray(data["greedy"]))))
            colors.append("tab:blue")
        if data["bestk"] is not None:
            labels.append("Model Best-K")
            means.append(float(np.mean(np.asarray(data["bestk"]))))
            colors.append("tab:green")
        if data["heuristic"]:
            vals = np.concatenate([np.asarray(v) for v in data["heuristic"]])
            labels.append("Heuristic Greedy")
            means.append(float(np.mean(vals)))
            colors.append("tab:orange")
        if data["concorde"]:
            vals = np.concatenate([np.asarray(v) for v in data["concorde"]])
            labels.append("Concorde")
            means.append(float(np.mean(vals)))
            colors.append("tab:red")
        if not labels:
            continue
        max_instances = None
        if data["max_instances"]:
            vals = [v for v in data["max_instances"] if v > 0]
            max_instances = vals[0] if vals else None
        context = format_context_title(dataset_base, metric, max_instances=max_instances)
        title = f"Average Lengths\n{context}"
        out_path = os.path.join(out_dir, f"avg_{dataset_base}_{checkpoint_dir}_{metric}.png")
        plt.figure(figsize=(7, 5))
        x = np.arange(len(labels))
        plt.bar(x, means, color=colors)
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.ylabel("Average tour length")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved average bar chart: {out_path}")


def plot_pairwise_histograms(payloads, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for payload in payloads:
        meta = extract_meta(payload)
        dataset_base = meta["dataset_base"]
        metric = meta["metric"]
        ckpt_base = meta["checkpoint_base"]
        ckpt_dir = meta["checkpoint_dir"]
        sel = meta["selection_mode"]

        model_greedy = np.asarray(payload.get("model_greedy", []))
        model_best = np.asarray(payload.get("model_best_k", []))
        heuristic = np.asarray(payload.get("heuristic_greedy", []))
        concorde = np.asarray(payload.get("concorde", []))

        pairs = []
        if sel == "greedy":
            pairs.append(("heuristic", "Heuristic Greedy", heuristic, "Model Greedy", model_greedy))
            pairs.append(("concorde", "Concorde", concorde, "Model Greedy", model_greedy))
        elif sel == "bestk":
            pairs.append(("heuristic", "Heuristic Greedy", heuristic, "Model Best-K", model_best))
            pairs.append(("concorde", "Concorde", concorde, "Model Best-K", model_best))
            pairs.append(("model", "Model Greedy", model_greedy, "Model Best-K", model_best))

        for tag, label_a, data_a, label_b, data_b in pairs:
            if data_a is None or len(data_a) == 0 or data_b is None or len(data_b) == 0:
                continue
            series = [
                (label_a, data_a, "tab:orange" if "Heuristic" in label_a else "tab:red" if "Concorde" in label_a else "tab:blue"),
                (label_b, data_b, "tab:green" if "Best" in label_b else "tab:blue"),
            ]
            context = format_context_title(dataset_base, metric, selection_mode=sel, max_instances=meta["max_instances"])
            title = f"{label_a} vs {label_b}\n{context}"
            out_path = os.path.join(
                out_dir,
                f"compare_{tag}_{label_a.replace(' ', '_')}_{label_b.replace(' ', '_')}_{dataset_base}_{ckpt_dir}_{ckpt_base}_{sel}_{metric}.png",
            )
            plot_overlay_hist(out_path, series, title)
            print(f"Saved comparison: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist_dir", default="hist_data")
    parser.add_argument("--avg_out_dir", default="hist_plots/avg")
    parser.add_argument("--compare_out_dir", default="hist_plots/compare")
    args = parser.parse_args()

    npz_files = load_hist_data(args.hist_dir)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {args.hist_dir}")

    payloads = []
    for path in npz_files:
        payloads.append(np.load(path, allow_pickle=True))

    plot_avg_length_bars(payloads, args.avg_out_dir)
    plot_pairwise_histograms(payloads, args.compare_out_dir)


if __name__ == "__main__":
    main()
