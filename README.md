# Conditional Flow Matching and MeanFlow on Morpho-MNIST

This repository contains implementations of conditional generative models trained on the Morpho-MNIST dataset. It supports two methods:
1. **Conditional Flow Matching (CFM)**: A simulation-free continuous normalizing flow method.
2. **MeanFlow**: A method for efficient one-step generation by learning the average velocity field.

The models incorporate two forms of conditioning: discrete digit identity and a continuous slant descriptor extracted from the morphological annotations.

### Inference Results

### Inference Results

Comparison between **Conditional Flow Matching (CFM)** (ODE integration), **MeanFlow** (1-step), and **Rectified MeanFlow** (1-step after distillation).

![Inference Comparison](inference_comparison.png)

## 1. Repository Layout

- `dataset/`: Data loading utilities, including transforms for PIL↔Tensor conversion and a `MorphoMNISTDataset` class that reads the gzipped IDX files and morphological descriptors.
- `model/`: Core model components such as the time-position embedding, transformer backbone.
- `script/train.py`: Training entry point with mixed-precision support and Weights & Biases logging.
- `script/inference.py`: Sampling script demonstrating conditional generation under digit-only, slant-only, and joint conditioning regimes.

## 2. Data Preparation

Place the Morpho-MNIST assets (IDX images/labels and `*-morpho.csv` files) under `dataset/Morpho-MNIST/`. The provided `dataset.dataset.MorphoMNISTDataset` expects filenames following the canonical `train-*`/`t10k-*` naming used by the original release.


## 3. Training

```bash
pip install torch torchvision pandas matplotlib wandb
export WANDB_API_KEY=...         # optional, required for online logging
# Train Conditional Flow Matching (CFM) - Default
python script/train.py --method cfm

# Train MeanFlow (One-step generation)
python script/train.py --method mean_flow
```

Some notes:
- Automatic mixed precision is enabled on CUDA devices; disable by forcing `DEVICE=cpu`.
- Weights & Biases logging can be deactivated by setting `WANDB_MODE=offline` or `WANDB_DISABLED=true`.

## 4. Inference and Evaluation

```bash
# Inference with CFM model
python script/inference.py --method cfm

# Inference with MeanFlow model
python script/inference.py --method mean_flow
```

The script loads `model.pt`, samples random noise vectors, and performs flow matching generation while conditioning on:

1. All ten digit classes with zero slant.
2. A fixed digit with a sweep across slant values.
3. A two-dimensional grid spanning digits and slants.

## 5. Acknowledgement
The original paper for the Morpho-MNIST dataset: https://arxiv.org/pdf/1809.10780

## 6. Rectified Flow Distillation (Improving MeanFlow)

To improve the resolution of MeanFlow (1-step generation) while maintaining its speed, we can use **Rectified Flow** (or Reflow) to distill the high-quality CFM model into current MeanFlow model.

1.  **Generate Rectified Dataset**: Use the trained CFM model to generate (noise, data) pairs connected by ODE trajectories.
    ```bash
    python script/distill.py --generate --teacher_path model_cfm.pt --save_path rectified_data.pt
    ```

2.  **Train Student MeanFlow**: Train the MeanFlow model on the rectified paths.
    ```bash
    python script/distill.py --train --save_path rectified_data.pt --student_path model_mean_flow_rectified.pt
    ```

3.  **Inference**:
    You can then run inference with the new model directly:
    ```bash
    python script/inference.py --method mean_flow_rectified
    ```