# DRL-CFD: Deep Reinforcement Learning

`drl-cfd` is a tool for combining the policy-based optimization (PBO) deep reinforcement learning library with Computational Fluid Dynamics (CFD) to solve optimizations of complex problems in computational fluid dynamics.


This repo provides an integration of the [pbo](https://github.com/theodore-michel/pbo)

## Setup

```bash
git clone git@github.com:theodore-michel/graph-drl.git
cd graph-drl
git submodule update --init --recursive
```

### Environment setup for graph-drl

Some packages in this repo (notably `torch-scatter`, `torch-sparse`, and
other PyTorch-related packages) require `torch` to be importable during their
build/metadata stage. Installing everything with a single `pip -r requirements.txt`
step may fail because `torch` hasn't been installed yet.

Best option is to do a 2-step installation.
Pick the GPU or CPU instructions depending on your machine.

### GPU (CUDA) - recommended when you have a compatible NVIDIA GPU

**Step 1:** Create and activate a minimal conda environment:

```bash
conda create -n graphdrl python=3.11 pip -y
conda activate graphdrl
```

**Step 2:** Install PyTorch wheels that match your CUDA version. Example for CUDA 12.4:

```bash
# install torch + torchvision + torchaudio via pip wheels (CUDA 12.4 example)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# install PyTorch Geometric related wheels that match the torch+cuda build
pip install --find-links https://pytorch-geometric.com/whl/torch-2.4.0+124.html \
  torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2
```

**Step 3:** Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

### CPU-only (no GPU)

If you don't have an NVIDIA GPU or prefer a CPU-only environment, install CPU
wheels for PyTorch and the matching PyG wheels.

```bash
conda create -n graphdrl python=3.11 pip -y
conda activate graphdrl

# Example: install CPU-only torch wheels
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
pip install torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

pip install --find-links https://pytorch-geometric.com/whl/torch-2.4.0+cpu.html \
  torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2

pip install -r requirements.txt
```

> [!Note]
> Environment setup troubleshooting tips :
>
> - Make sure the wheel tags & find-links match the torch + CUDA combination you
  installed (for example `torch-2.4.0+124` for CUDA 12.4).
> - If pip still tries to compile native extensions, ensure you have a C compiler
  toolchain available.
> - For PyG, prefer their prebuilt wheels on <https://pytorch-geometric.com/>

Watch out for CUDA and CuDNN version compatibility issues between build and runtime, you might get an error like:

```bash
FailedPreconditionError: DNN library initialization failed
```

in `tensorflow` or `keras` when running the pbo scripts.

> [!Warning]
> You might need to adjust cudnn version in `requirements.txt` and `pyproject.toml` to your CUDA build and runtime versions.
> Make sure you that the loaded CuDNN library is of matching or higher version than the CuDNN built version (one used at runtime).
> For example, for cuda 12.4, use `nvidia-cudnn-cu12==9.3.0.75` because `tensorflow==2.19.0` is built with cudnn 9.3.0 but the `torch=2.4.0+124` build version requires cudnn 9.1.0 and nothing above.
>
> To resolve this, we currently advise you simply install the env with `nvidia-cudnn-cu12==9.1.0.70` and then manually upgrade to `nvidia-cudnn-cu12==9.3.0.75` (see below) so that the library binary for `9.3.0.75` is available.
> At runtime, as long as `torch` and `tf` are not called on same import then they will each look for the correct cudnn version they were built with without conflict. An update/fix on this will be made in future releases.
>
> To upgrade cudnn in the conda environment, first run the install with `nvidia-cudnn-cu12==9.1.0.70` as is the case with the current `.toml`, then after this install run:
>
> ```bash
> pip install --upgrade nvidia-cudnn-cu12==9.3.0.75
> ```

You should be set to go now, with a clean conda environment with all dependencies installed.
> [!Note]
> The `meshio` package is installed from a custom [git repository](https://github.com/theodore-michel/meshio.git) to include some recent fixes that are not yet available in the latest release on PyPI. If you encounter any issues with the custom `meshio` package, just switch to the official meshio by replacing the corresponding line in `requirements.txt` and `pyproject.toml` with `meshio==5.3.5`.

## Install the graphdrl package

Once the virtual environment is set up and activated, you can install the `graphdrl` package in editable mode so that you can import it from anywhere.

First start by initializing and installing the submodules so that the directories are populated and that `pbo` and `graph-physics` are available for import anywhere:

```bash
git submodule update --init --recursive
pip install -e external/pbo 
pip install -e external/graph-physics
```

Then install the `graphdrl` package itself:

```bash
pip install -e .
```

Now `import graphdrl` should work anywhere.

## Update the submodules

The repo uses git submodules to include the `pbo` and `graph-physics` libraries. Make sure to initialize and update the submodules after cloning the repository. To update the submodules to their latest versions, you can run:

```bash
git submodule update --remote --merge
```

Updating the submodules will fetch the latest changes from their respective repositories and merge them into your local copy, allowing you to benefit from any new features or bug fixes. This will also update the commit reference in the main repo, so make sure to commit the updated submodule references afterwards.

> [!Note]
> Editing submodule code:
> It is highly recommended that you edit the submodule codes through PRs to the original repositories. This ensures that any changes you make are properly tracked and can be easily merged with future updates from the original repositories.
> Every time an edit in the source code is made at the origin, you need to pull the changes in the submodule using the command shown above, and then commit the updated submodule reference in the main repo.
> [!Note]
> To make a submodule follow a specific branch (e.g., `predict-panels` branch for `graph-physics`), you can simply edit the `.gitmodules` file to specify the branch for that submodule via the `branch` attribute.

## Run Example

```bash
pbo scripts/panels.json
```

This will run a PBO optimization on a fluid dynamics problem defined in the `scripts/panels.py` file using a GNN model to represent the fluid dynamics system. The pbo optimization meta parameters are defined in the `scripts/panels.json` file.

Before running, make sure `mpirun` is available and properly configured (for example use `module load openmpi` if you have it).
Configure your case in `scripts/panels.py`, especially the paths to the cimlib driver binary, and check that you configured all of the environment parameters (json files) in [environment_config/](graphdrl/environment_config.py).

Make sure that you set the globale variable `TF_USE_LEGACY_KERAS=True` in your environment, as the default Keras version is not compatible with the current PBO implementation. You can do this by running:

```bash
export TF_USE_LEGACY_KERAS=True
```

You can also set this variable in your job bash script (sh file in [scripts/](scripts/)) before calling the `pbo` command, to avoid conflicts with other TensorFlow/Keras projects.

>[!Note]
> A sign that this variable is not set correctly is if you get an error saying that there were no gradients provided for any variable when running the pbo command.

## Run Tests

A series of unit tests are provided in the `tests` directory to validate the functionality of the library and its different components. Run the tests using `pytest`:

```bash
pytest -q
```

Make sure to run the tests in an environment where the `graphdrl` package is installed and all dependencies are satisfied.

> [!Note]
> During testing, makes sure mpirun is available and properly configured, make sure you give correct paths to driver binaries and gnn models checkpoints as these will throw errors if not found.
