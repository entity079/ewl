# System requirements

## Operating System
nnU-Net has been tested on Linux (Ubuntu 18.04, 20.04, 22.04; centOS, RHEL), Windows and MacOS! It should work out of the box!

## Hardware requirements
We support GPU (recommended), CPU and Apple M1/M2 as devices (currently Apple mps does not implement 3D 
convolutions, so you might have to use the CPU on those devices).

### Hardware requirements for Training
We recommend you use a GPU for training as this will take a really long time on CPU or MPS (Apple M1/M2). 
For training a GPU with at least 10 GB (popular non-datacenter options are the RTX 2080ti, RTX 3080/3090 or RTX 4080/4090) is 
required. We also recommend a strong CPU to go along with the GPU. 6 cores (12 threads) 
are the bare minimum! CPU requirements are mostly related to data augmentation and scale with the number of 
input channels and target structures. Plus, the faster the GPU, the better the CPU should be!

### Hardware Requirements for inference
Again we recommend a GPU to make predictions as this will be substantially faster than the other options. However, 
inference times are typically still manageable on CPU and MPS (Apple M1/M2). If using a GPU, it should have at least 
4 GB of available (unused) VRAM.

### Example hardware configurations
Example workstation configurations for training:
- CPU: Ryzen 5800X - 5900X or 7900X would be even better! We have not yet tested Intel Alder/Raptor lake but they will likely work as well.
- GPU: RTX 3090 or RTX 4090
- RAM: 64GB
- Storage: SSD (M.2 PCIe Gen 3 or better!)

Example Server configuration for training:
- CPU: 2x AMD EPYC7763 for a total of 128C/256T. 16C/GPU are highly recommended for fast GPUs such as the A100!
- GPU: 8xA100 PCIe (price/performance superior to SXM variant + they use less power)
- RAM: 1 TB
- Storage: local SSD storage (PCIe Gen 3 or better) or ultra fast network storage

(nnU-net by default uses one GPU per training. The server configuration can run up to 8 model trainings simultaneously)

### Setting the correct number of Workers for data augmentation (training only)
Note that you will need to manually set the number of processes nnU-Net uses for data augmentation according to your 
CPU/GPU ratio. For the server above (256 threads for 8 GPUs), a good value would be 24-30. You can do this by 
setting the `nnUNet_n_proc_DA` environment variable (`export nnUNet_n_proc_DA=XX`). 
Recommended values (assuming a recent CPU with good IPC) are 10-12 for RTX 2080 ti, 12 for a RTX 3090, 16-18 for 
RTX 4090, 28-32 for A100. Optimal values may vary depending on the number of input channels/modalities and number of classes.

# Installation instructions
We strongly recommend that you install nnU-Net in a virtual environment! Pip or anaconda are both fine. If you choose to 
compile PyTorch from source (see below), you will need to use conda instead of pip. 

Use a recent version of Python! 3.9 or newer is guaranteed to work!

**nnU-Net v2 can coexist with nnU-Net v1! Both can be installed at the same time.**

1) Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website (conda/pip). Please 
install the latest version with support for your hardware (cuda, mps, cpu).
**DO NOT JUST `pip install nnunetv2` WITHOUT PROPERLY INSTALLING PYTORCH FIRST**. For maximum speed, consider 
[compiling pytorch yourself](https://github.com/pytorch/pytorch#from-source) (experienced users only!). 
2) Install nnU-Net depending on your use case:
    1) For use as **standardized baseline**, **out-of-the-box segmentation algorithm** or for running 
     **inference with pretrained models**:

       ```pip install nnunetv2```

    2) For use as integrative **framework** (this will create a copy of the nnU-Net code on your computer so that you
   can modify it as needed):
          ```bash
          git clone https://github.com/MIC-DKFZ/nnUNet.git
          cd nnUNet
          pip install -e .
          ```
3) nnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to
   set a few environment variables. Please follow the instructions [here](setting_up_paths.md).
4) (OPTIONAL) Install [hiddenlayer](https://github.com/waleedka/hiddenlayer). hiddenlayer enables nnU-net to generate
   plots of the network topologies it generates (see [Model training](how_to_use_nnunet.md#model-training)). 
To install hiddenlayer,
   run the following command:
    ```bash
    pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
    ```

Installing nnU-Net will add several new commands to your terminal. These commands are used to run the entire nnU-Net
pipeline. You can execute them from any location on your system. All nnU-Net commands have the prefix `nnUNetv2_` for
easy identification.

Note that these commands simply execute python scripts. If you installed nnU-Net in a virtual environment, this
environment must be activated when executing the commands. You can see what scripts/functions are executed by 
checking the project.scripts in the [pyproject.toml](../pyproject.toml) file.

All nnU-Net commands have a `-h` option which gives information on how to use them.

## Troubleshooting common "it is not working" issues

If nnU-Net does not run as expected, these are the most common causes:

1. **`nnUNetv2_*` commands are not found**
   - Cause: nnU-Net was not installed into the active Python environment, or the environment is not activated.
   - Fix: activate the intended environment and run `pip install nnunetv2` (or `pip install -e .` when working from source).

2. **`pip install -e .` fails while downloading build dependencies**
   - Cause: restricted network/proxy settings can block build-isolation dependency downloads.
   - Fix: ensure `pip` can reach your package index/proxy. In controlled environments with all dependencies already
     installed, `pip install -e . --no-build-isolation` can help.

3. **Runtime errors about missing nnU-Net paths (for example, preprocessed/model directories)**
   - Cause: required environment variables are missing.
   - Fix: configure paths as described in [setting_up_paths.md](setting_up_paths.md) (`nnUNet_raw`,
     `nnUNet_preprocessed`, `nnUNet_results`).

4. **Training is unexpectedly slow or unstable with specific PyTorch versions**
   - Cause: known performance regression in PyTorch `2.9.0` for 3D convolutions with AMP.
   - Fix: use PyTorch `2.8.0` or lower.

5. **Training fails due to memory (RAM/VRAM) issues**
   - Cause: insufficient free GPU memory or too many augmentation workers.
   - Fix: reduce parallel workload (for example by setting `nnUNet_n_proc_DA` lower), close other GPU processes,
     and verify hardware meets the requirements above.


## Troubleshooting `train_loss nan` / `val_loss nan`

What causes NaN **in the first place**? In nnU-Net, `train_loss` is logged directly from `self.loss(output, target)` in `train_step`, and `val_loss` from the same loss call in `validation_step`. So NaNs originate when that loss computation (or values feeding it) becomes non-finite.

If training starts but quickly reports `train_loss nan` and `val_loss nan`, typical root causes are:

1. **Invalid labels in the training set**
   - Cause: labels contain values outside the configured classes (for example unexpected integers, wrong background id, corrupted masks).
   - Why this creates NaNs: invalid targets can break loss computation and Dice statistics.
   - Fix: verify labels are valid for your `dataset.json` definition and run dataset integrity verification before training.

2. **Images/labels are misaligned or preprocessing produced invalid values**
   - Cause: mismatched geometry between image and segmentation, or volumes containing NaN/Inf intensities.
   - Why this creates NaNs: transformations and normalization can propagate invalid numeric values into the network.
   - Fix: inspect raw cases for NaN/Inf and confirm image-label correspondence/orientation/spacing are correct.

3. **Mixed precision instability on your hardware/software stack**
   - Cause: AMP can be unstable in some environments and has known regressions with certain PyTorch versions.
   - Why this creates NaNs: gradients or activations overflow in fp16 paths.
   - Fix: switch to a recommended PyTorch version (see warning above), and test one run without AMP to confirm stability.

4. **Learning-rate/optimization instability after custom changes**
   - Cause: modified trainer, custom loss, custom augmentation, or extreme optimizer settings.
   - Why this creates NaNs: exploding gradients or invalid ops introduced by custom code.
   - Fix: fall back to default trainer/plans first, then reintroduce customizations incrementally.

5. **Path/configuration mismatches causing wrong data to be loaded**
   - Cause: `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` pointing to stale/wrong folders.
   - Why this creates NaNs: model may train on unintended or partially processed data.
   - Fix: re-check environment variables and dataset id/fold/configuration arguments.

A practical debug sequence is:
- verify dataset integrity and class ids,
- inspect a few training cases manually after conversion,
- run a short training with defaults,
- then disable AMP for one sanity-check run,
- and finally add customizations back one by one.


Quick checks you can run immediately:
- `nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity`
- verify your raw labels contain only expected class ids from `dataset.json`
- run one short sanity training without custom trainer/loss changes
- run one sanity training without AMP to test fp16 overflow sensitivity

