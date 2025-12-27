# travelling-salesman-machine-learning

Simple training and evaluation pipeline for TSP attention model with optional Concorde baseline.

## Folder layout
- `data/`: training datasets (`.h5`) (not tracked; you must generate)
- `test_data/`: held-out datasets for evaluation (not tracked; you must generate)
- `checkpoints*/`: saved models (ignored by git)
- `hist_data/`: saved evaluation arrays (ignored by git)
- `hist_plots/`: plots built from `hist_data/` (ignored by git)
- `visualizations/`: tour images (ignored by git)

## Generate datasets
Datasets are not stored in git. You need to generate them locally.

Edit `tsp_instance_generator.py` and set:
- `MODE` (e.g., `uniform`, `two_islands`, `three_columns`)
- `num_points`, `num_instances`
- `test` to write into `test_data/` instead of `data/`

Then run:
```bash
python tsp_instance_generator.py
```

## Train
```bash
python train.py --data_file data/tsp_20_10000.h5 --save_dir checkpoints
```

Manhattan training:
```bash
python train.py --data_file data/tsp_20_10000.h5 --distance_metric manhattan --save_dir checkpoints_manhattan
```

Resume from a checkpoint:
```bash
python train.py --data_file data/tsp_two_islands_50_10000.h5 --save_dir checkpoints_two_islands_50 \
  --resume checkpoints_two_islands_50/ckpt_ep50.pt --epochs 100
```

## Visualize tours
```bash
python visualize.py --model checkpoints/tsp_attention_model.pt \
  --data_file test_data/tsp_20_10000.h5 --indices 0,1,2,3,4
```

## Evaluate and save histogram data
This picks the best checkpoint for greedy and for best-of-K (K=50) and saves arrays to `hist_data/`.
```bash
python evaluate_histograms.py --test_data_dir test_data \
  --checkpoints_dirs checkpoints,checkpoints_manhattan,checkpoints_two_islands_50,checkpoints_three_columns_50 \
  --best_k 50 --max_instances 500 --strict_dataset_match
```

## Build plots from hist_data
```bash
python plot_hist_data.py --hist_dir hist_data
```

## Notes on Concorde
Concorde is optional. If available, it is used as a baseline and runs in a temporary directory to avoid leaving `.res` files.
