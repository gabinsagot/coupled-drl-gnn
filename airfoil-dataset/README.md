# Generating GNN training datasets of airfoil CFD simulations using CIMLIB

## Overview

This repository contains code for launching different CFD (Computational Fluid Dynamics) simulations with varying geometrical configurations and mesh refinements and resolutions, compressing the results, and outputting them in a dataset format ready for GNN training.
The simulations run using our in-house finite elements software [CIMLIB](https://doi.org/10.1063/1.2740823).

We offer a set of different configuration scripts that cover different options:

- Geometries:
  - 2D:
    * [x] airfoils
- Dynamics:
  - Numerical solver: input any solver you would like using a compiled CIMLIB driver
    * [x] Incompressible Navier-Stokes with [VMS](https://doi.org/10.1016/j.jcp.2010.07.030)
    * [ ] Incompressible Navier-Stokes with turbulence models
    * [ ] Compressible Navier-Stokes
  - Physics:
    * [x] User-defined physics parameters (e.g. $Re$, $\rho$, $U_\infty$, $\mu$, $\Delta t$)
- Mesh resolution:
  * [x] Anisotropic Boundary Layer Mesh adaptation (in-house, with user-defined mesh size fields)
  * [x] Multi-Grid BLM adapation (simulation on fine resolution, interpolation on coarser BLM mesh)
  * [x] Isotropic adapted mesh (using [GMSH](https://gitlab.onelab.info/gmsh/gmsh))
- Parallelization:
  * [x] MPI for each parallelization of each simulation run
  * [x] Multiprocessing for multiple simulations at the same time
  * [x] User-defined parallelization and resource allocation
- Postprocessing:
  * [x] Selected features output  
  * [x] Force computation on objects surfaces
  * [x] Sensor point data extraction
  * [x] Compression/Decompression to .xdmf/.vtu
  * [x] Edit/rename/check features
  * [x] Zip dataset
  * [x] Plot dataset statistics and object distribution
  * [x] Pre-split datasets into train/test/predict

The code uses Python and is designed to run both on a standard PC and on a high-performance computing cluster infrastructure using SLURM.

Feel free to open a PR if you want to implement a new feature, or an issue to request one.

## Installation and Requirements

To set up the repository, follow these steps:

```bash
git clone https://github.com/theodore-michel/airfoil-dataset.git
cd airfoil-dataset
pip install -e .
```

## General code description

### Setting up

Generating a dataset case requires that you create a specific meta parameters file (.json) to define the dataset details and simulation parameters.

Let's run through the different parameters to set. Examples for each implemented case are available [here](https://github.com/theodore-michel/airfoil-dataset/tree/main/config).

#### General parameters

The first parameters are general parameters:

```json
{
"case": "airfoil",
"case_name": "airfoil_Re100",
"num_configs": 150,
"configs_pool_path": "./pools/configs_pool_airfoil_Re100.txt",
"driver": "/path/to/your/cimlib_CFD_driver",
"shift_x_objects": true,
"number_parallel_jobs": 4,
"dim": 2,
```

- `case`: the geometry case you will use (for now only "airfoil").
- `case_name`: personalized name of your dataset, such as "my_airfoil_case" or "airfoil_Re100".
- `num_configs`: the number of unique configurations to create or load.
- `configs_pool_path`: the path to the the dataframe of configurations you will save (if you specify that you are creating configs) or load (if not creating).
- `driver`: the path to the CIMLIB binary driver to run simulations with (obtained by compiling your desired CIMLIB build, an example binary that runs on LAFFITTE cluster is given in [drivers/](https://github.com/theodore-michel/airfoil-dataset/tree/main/drivers)).
- `shift_x_objects`: whether to shift all objects so that the leftmost one is always located at the same x origin.
- `number_parallel_jobs`: how many parallel simulations to launch during dataset generation.
- `dim`: dimension of the case (3D not implemented yet).

> [!WARNING]
> `config_pool_path` is the path to the txt pool format, but it requires a `pkl` (`pandas` pickle) twin file if you wish to load one. These twin files are automatically generated when creating a configs pool.
> Examples of configuration pools can be found [here](https://github.com/theodore-michel/airfoil-dataset/tree/main/pools).

#### Geometry parameters

Each case has its case-specific geometry parameters to help design the configurations simulated in the dataset.

```json
"geometry_parameters":{
    "number_airfoils": [1, 1],
    "airfoil_type": "NACA0012",
    "chord": [0.5, 2.0],
    "thickness": [0.5, 3.0],
    "angle_of_attack": [-30.0, 30.0],
    "x_object": [-2,3],
    "y_object": [-3,3],
}
```

For the airfoil, we generate uniformly random configurations with parameters sampled from:

- `number_airfoils`: min and max number of airfoils to create in each configuration.
- `airfoil_type`: type of airfoil to use (NACA4 series, e.g. "NACA0010", "NACA0057", or "morphed" airfoils).
- `chord`: min and max of chord length distribution interval.
- `thickness`: min and max of thickness distribution interval.
- `angle_of_attack`: min and max of angle of attack distribution interval, in degrees.
- `x_object`: min and max of x coordinate distribution (if `shift_x_objects` is set to true, all objects will be shifted so that the first object is always centered on the min x coordinate).
- `y_object`: min and max of y coordinate distribution.
- `number_airfoils`: min and max number of objects to create in each configuration. 

> [!NOTE]
> To generate different random configurations, sets of geometry parameters are randomly selected in the user-provided intervals using the uniform sampling method of `numpy`: `numpy.random.uniform`.
> If you wish to keep a parameter fixed, simply set identical interval bounds: `"number_airfoils":[1,1]`.

##### Morphed airfoil parameters

Morphed airfoils are generated by specifying camber and thickness at control points. Set `"airfoil_type":"morphed"` and provide lists of min/max bounds for thickness and camber control points. The morphed airfoil is obtained in the following way:

- Airfoil morphs from a NACA0010 baseline with the same total point count
- Leading and trailing edges are fixed
- Control points apply to the upper surface; the lower surface is mirrored along the camber line
- Camber values apply to all points between leading and trailing edge points (excluded)
- Thickness values apply to the same points, except those adjacent to leading/trailing edge points
- If you provide `n` camber control points, provide `n-2` thickness control points
- The resulting airfoil will have `2n+2` total points

Below is an example of morphed airfoil geometry parameters for 3 control points:

```json
"geometry_parameters":{
    "number_airfoils": [1, 1],
    "airfoil_type": "morphed",
    "chord": [1.0, 1.0],
    "thickness": [[min1,max1], [min2,max2]],
    "camber": [[min1,max1], [min2,max2], [min3,max3], [min4,max4]],
    "angle_of_attack": [-45, 45],
    "x_object": [-2,3],
    "y_object": [-3,3],
}
```

- `thickness`: list of lists of min and max thickness control point values.
- `camber`: list of lists of min and max camber control point values.

#### Domain parameters

You also need to define the fluid domain parameters for the simulations. These parameters are common to all configurations, each simulation of a geometry case runs with the same domain parameters.

```json
"domain_parameters":{
    "origin_x": -5,
    "origin_y": -5,
    "origin_z": 0,
    "dx": 20,
    "dy": 10,
    "dz": 0
}
```

- `origin_`: This indicates the lower left corner of the domain. If 2D, set z-relative parameters to 0.
- `dx`, `dy`, `dz`: The dimensions of the domain.

#### CFD parameters

Then, you need to define the simulation parameters. This is where you specify the physics of your simulation, based on tunable parameters of your `IHM.mtc` simulation interaction file. Thus, you can replace "`PasDeTemps`", ..., "`FrequenceStockage_vtk`" by whatever is used in your IHM file.

```json
"cfd_parameters":{
    "PasDeTemps": 0.01,
    "Rho1": 1,
    "Eta1": 1e-2,
    "VxIn": 1,
    "FrequenceStockage_vtk": 20,
    "Hbox123": [0.00125, 0.25, 1.0],
    "mesh_adapt": true,
    "calc_force": true,
    "variable_inlet":false,
    "number_cores": 16
}
```

- `PasDeTemps`: The timestep of your simulation.
- `Rho1`: The volumic mass of your fluid.
- `Eta1`: The viscosity of your fluid.
- `VxIn`: The freestream velocity of your simulation. Can be passed as `[min,max]` bounds if `"variable_inlet"=true` to uniformly sampled inlet velocity amplitudes.
- `FrequenceStockage_vtk`: How many timesteps pass by between each save of a VTU file. This determines final dataset timestep between snapshots.
- `Hbox123`: The mesh size of the BLM mesh fields Hbox1 (object boundary), Hbox2 (intermediary box, automatically computed from object size and location), and Hbox3 (domain boundary). Ignored if `mesh_adapt` is set to false.
- `mesh_adapt`: Toggles the BLM anisotropic mesh adaptation. If set to false, mesh used for simulation will default to GMSH isotropic adapted mesh.
- `calc_force`: Toggles the Drag, Lift, Torque computation at object surface. When true, this will generate csv files that contain force signals for each timestep and object for each simulation.
- `number_cores`: How many cores are allocated per simulation (each simulation runs with MPI)

> [!NOTE]
> For choosing the Hbox123 (`Hbox1`, `Hbox2`, `Hbox3` in `IHM.mtc`) parameters to create a clean adapted mesh, there is a recommended ratio to keep between Hbox1 and 2: $h_1 < \frac{1}{50} h_2$. Fore more information on BLM method and parameter choice/influence, see [Michel et al.](https://doi.org/10.1063/5.0233709).


> [!NOTE]
> Any parameter of your `IHM.mtc` file that you wish to control for the dataset generation can be listed in the `cfd_parameters`. 
> Simply put double quotes around the parameter name, and specify its value. It will then be automatically applied to every IHM of the dataset simulations.

#### Graph parameters

These last parameters are for the postprocess and more specifically address the needs you may have if generating a dataset for GNN training:

```json
"graph_parameters":{
    "vtu_start_in_seconds": 0,
    "how_many_vtu": 600,
    "features": ["Temps","Vitesse","Pression","LevelSetObject","NodeType","Reynolds"],
    "multigrid": false
}
```

- `vtu_start_in_seconds`: When to start saving snapshots in the simulation (corresponds to time in seconds of your simulation time, not increment number).
- `how_many_vtu`: How many frames to save for one trajectory. The end time of the simulation will automatically be computed from the start time, timestep, frequency of snapshot saves, and number of vtus you impose. 
- `features`: Which features to keep in simulation outputs. Only provide features already implemented in outputs of simulation.
- `multigrid`: Toggles whether to run a multigrid dataset generation (simulation on a fine mesh, interpolation on a coarser adapted mesh, saves both).
  
>[!WARNING]
> Currently in the multigrid approach, the coarse mesh used is an initial increment (increment number 2) of the BLM mesh adaptation, thus you need to be using a working version of a multigrid simulation setup (see examples [here](https://github.com/theodore-michel/airfoil-dataset/tree/main/cfd_bank/cfd_airfoil_multigrid))

### Configurations Pool

Generating a dataset requires that you either create or load a pool of configurations that you will then simulate. A configuration is simply a unique ID number associated to geometric parameters. Keeping the configuration pool file is essential to keep a trace/knowledge of which configuration is which. Thus, the pool file used for a dataset is systematically copied to the dataset directory.
In the code, the configuration pool is manipulated as a `Configs` class object, inherited from `pandas.DataFrame`. It is saved/loaded from the path provided in the meta parameters file. See an example [configurations pool for Re100 airfoil](https://github.com/theodore-michel/airfoil-dataset/blob/main/pools/configs_pool_airfoil_Re100.txt).

> [!WARNING]
> Creating a configuration pool of same name as an existing one (i.e. providing a `configs_pool_path` that already exists) will throw an error to avoid overwrite and will prompt you to either rename the existing pool or discard it.

> [!NOTE]
> When selecting configurations from an existing pool, pool files (txt and pkl) of the selected configurations will be saved with suffix `_selected`.
> Note that they will have been shuffled, but they are the same configuration/parameters pairs.
>
> Each configuration ID is unique within each configuration pool. The ID is an integer (in string format), whose first digit indicates the number of objects in the configuration (10001 for 1 cylinder, 20001 for 2 cylinders, etc.). Numbering starts at 0, ends at `num_configs`.

### Geometries

Geometries are created using the GMSH [OpenCascade](http://www.opencascade.com) kernel. If you wish to implement or adapt more geometries, do so by creating a new class of geometry in [src/geometries.py](https://github.com/theodore-michel/airfoil-dataset/blob/main/src/geometries.py). Then you can add them in the dataset creation cases in [src/dataset.py](https://github.com/theodore-michel/airfoil-dataset/blob/main/src/dataset.py).

For simulations, we create 2 meshes for each geometry: an object mesh and a domain mesh. We use a body-fitted approach to ensure stricter boundary conditions and avoid flow problems (such as leaks or irregular boundary with immersed approach). This means that the object is cut away from the domain, leaving a hole in its place. In the case of multiple objects, both a general object and single object meshes are generated.

For any additional information and documentation on how these geometries and meshes are constructed, see the [GMSH 4.13 documentation](https://gmsh.info/doc/texinfo/gmsh.html).

### CFDs

Simulations are launched by copying a setup (taken from the [cfd_bank](https://github.com/theodore-michel/airfoil-dataset/tree/main/cfd_bank)) to the simulation directory `simu/simu_{case_name}` (created at the beginning of the dataset creation), and then running it using the user-specified CIMLIB driver. Each configuration is assigned its specific simulation directory into which we generate the object and domain meshes, convert them to CIMLIB format, and then prepare `mtc` scripts for running the simulation.

As the simulation runs, VTU outputs are generated in the `/Resultats` folder of each simulation setup directory, as well as a log file `log.out` in the root of each simulation setup directory.
Once the simulation is finished, the results are compressed to xdmf and moved to the corresponding case dataset in the dataset directory `dataset/dataset_{case_name}`. The configuration simulation setup directory is then deleted.

> [!WARNING]
> If a configuration simulation run fails or returns a non-zero exit signal, the simulation setup directory of that specific configuration will not be deleted for debugging purposes (you will be able to look at the simulation as it was launched, along with its log file `log.out`).

> [!NOTE]
> With this approach, you can generate new datasets of your own simulation cases by simply adding your simulation setup template in `cfd_bank` as `cfd_yourcase`, and creating a new case in [`src/configs.py`](https://github.com/theodore-michel/airfoil-dataset/blob/main/src/configs.py), as well as in [`src/geometries.py`](https://github.com/theodore-michel/airfoil-dataset/blob/main/src/geometries.py). Then add it in the casing in [`src/dataset.py`](https://github.com/theodore-michel/airfoil-dataset/blob/main/src/dataset.py):

```python
def which_config_type(self, meta_dict: str, create: bool) -> Configs:
    """Return the config pool class (Configs type) based on the case type.

    Args:
        meta_dict (str):  meta parameters dictionary
        create (bool): whether to create the pool of configurations
    """
    if self.type == "airfoil":
        return ConfigsAirfoil(
            meta_dict=meta_dict,
            path=self.configs_pool_path,
            num_configs=self.n_configs,
            create=create,
        )
    else:
        raise ValueError(
            f"Invalid case type: {self.type}. Available options are: airfoil."
        )
````

### Dataset

At the end of a dataset generation, all successful simulations will have generated a pair of `xdmf` and `h5` files (and one `csv` file if force signals were output), named after the case and configuration ID. This constitutes a dataset directory. You should then move this directory to a convenient location.

You can perform some postprocessing on this dataset such as editing the xdmf files, the feature names, the timestep values, check that the feature values don't exceed cerain thresholds, split the dataset into train/test/predict directories, plot the configurations, etc. These commands are provided in the next section and can be accessed in script format. For example [here](https://github.com/theodore-michel/airfoil-dataset/tree/main/src/postprocess) for postprocess commands.

## Useful Commands

After installing this repository, you will have access to several inline commands for managing datasets, geometries, simulations, file conversions, and postprocessing. All commands support `-h` or `--help` for more details.

---

### Dataset Generation

**Command:** `dataset`  
**Description:** Generate a dataset using a configuration JSON file. Supports creating/loading configuration pools and zipping the final dataset.

**Example:**

```bash
dataset \
  --dataset_config_file=config/meta.json \
  --create_configs_pool=True \
  --zip
```

---

### Geometry Creation

**Command:** `geometry`  
**Description:** Create geometries (objects and computational domains) and their meshes. Supports multiple geometry types.

**Subcommands:**  
- `airfoil`

**Example:**

```bash
geometry airfoil \
  --dataset_config=config/meta.json \
  --path=/path/to/cfd \
  --output_dir=/path/to/cfd/meshes \
  --dim=2 \
  ... \
  -x=[x1,x2,...,x5] \
  -y=[y1,y2,...,y5] \
  -n=5 \
```

> [!WARNING]
> All angle values provided by user should be in degrees, positive angles lead to a counter-clock-wise (trigonometric) rotation.

---

### Simulation

**Command:** `simu`  
**Description:** Run a CFD simulation for a given configuration, compress results, and save signals to CSV.

**Example:**
```bash
simu airfoil \
  --simu_config=/path/to/config.json \
  --path=/path/to/cfd \
  --dim=2 \
  --num_airfoils=1 \
  ... \
```

---

### File Conversion

**Command:** `convert`  
**Description:** Convert between mesh and result file formats.

**Subcommands:**

- `mesh` (GMSH mesh to MTC format)
- `vtu2h5` (Compress VTU files to XDMF/H5)
- `h52vtu` (Decompress XDMF/H5 to VTU)

**Example:**

```bash
convert mesh \
  path/to/meshfile.msh

convert vtu2h5 \
  path/to/files/*.vtu \
  --outfile=path/to/outfile.xdmf

convert h52vtu \
  path/to/file/infile.xdmf \
  --outfile=prefix_of_output
```

---

### Postprocessing

**Command:** `postprocess`  
**Description:** Tools for managing and analyzing datasets.

**Subcommands:**

- `edit` (Check/rename/drop fields in XDMF/H5 pairs)
- `reorder` (Reorder timesteps, impose new timestep)
- `split` (Split dataset into train/test/predict)
- `test` (Validate dataset quality)
- `interpolate` (Interpolate values between mesh resolutions)
- `simu` (Process simulation results: compress and save signals)
- `postsplits` (Analyze splits, plot PCA)

**Example:**

```bash
postprocess edit \
  --directory path/to/dataset \
  --recursive \
  --rename_fields '{"Velocity_Coarse":"Velocity","Pressure_Coarse":"Pressure"}'

postprocess reorder \
  path/to/xdmf/files \
  --outpath path/to/output/xdmf/files \
  --timestep 0.1 \
  --verbose

postprocess split \
  path/to/xdmf/files \
  -train ./train \
  -test ./test \
  -predict ./predict \
  --ratio 80,10,10 \
  --sorted

postprocess test \
  path/to/xdmf/files \
  --threshold 10 \
  -n 600

postprocess interpolate \
  -fine path/to/finemesh/xdmf \
  -coarse path/to/coarse/mesh \
  -out path/to/output \
  --fields Vitesse NodeType

postprocess simu \
  path/to/simulation_dir \
  --save_path path/to/output \
  --dim 2 \
  --vtu_start 0 \
  --dt 0.2

postprocess postsplits \
  --config_pool_path path/to/configs_pool.pkl \
  --split_dir path/to/split_dir \
  --plot_pca \
  --param_cols x_objects y_objects \
  --plot_save_path path/to/pca_plot.png
```

---

### Plotting

**Command:** `plot`  
**Description:** Visualize configuration distributions and postprocess simulation results.

**Example:**

```bash
plot \
  path/to/meta.json
```