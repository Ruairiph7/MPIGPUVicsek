# MPIGPUVicsek
Large-scale simulations of Vicsek-type models using CUDA-aware MPI with the Julia programming language.

Note this was written for Julia Version 1.11.1 (2024-10-16).

https://github.com/user-attachments/assets/a73e9483-84e8-4321-a2aa-dfc8ab76a564

# Installation

## 1) Install Julia
- Pick installation directory, e.g. a "work" directory - ```export WORK="/scratch/user"```
- Install Julia 1.11.1:
```
cd $WORK
wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.1-linux-x86_64.tar.gz
tar zxvf julia-1.11.1-linux-x86_64.tar.gz
rm ./julia-1.11.1-linux-x86_64.tar.gz
```

- Make a directory for packages:
```
mkdir ./.julia
```

- Update your .bashrc file with following lines, and then call ```source ~/.bashrc```
```
export WORK="/scratch/user"
export JULIA_DEPOT_PATH="$WORK/.julia"
export PATH="$PATH:$WORK/julia-1.11.1/bin"
export PATH="$PATH:$JULIA_DEPOT_PATH/bin"
```

- Note: automatic threading from system libraries may cause issues on HPC systems, so it is good to also include
```
export JULIA_NUM_THREADS=1
```

## 2) Clone this repository

- Pick where to install MPIGPUVicsek, e.g. into a "work" directory:
  ```
  cd $WORK
  git clone git@github.com:Ruairiph7/MPIGPUVicsek
  ```

## 3) Download required packages

- Run Julia from a node/machine with access to the internet, using the MPIGPUVicsek project:
```
julia --project="path/to/MPIGPUVicsek"
```
Note that Julia should always be launched like this when using this project.
 
- Instantiate the project from inside the Julia REPL:
```
using Pkg; Pkg.instantiate()
```

## 3) Configure CUDA.jl

- Run Julia from a node/machine with a GPU (you may need to launch an interactive job if on a HPC system), then load the CUDA.jl package. Note that you will need to wait a while for it to precompile.

```
using CUDA
```

- Set the CUDA runtime version to match the version of CUDA you will be using. On a HPC system this could be the version you load using e.g. ```module load cuda/xx.x.x```, or it could be the version found with the ```nvidia-smi``` command. For example, if it is CUDA 13.0.0, then call:
```
CUDA.set_runtime_version!(v"13.0.0")
```
	
- You will then need to restart Julia, and it is good to check afterwards that all is working. The following code should return the version you requested and successfully allocate an array on the GPU:

```
using CUDA
CUDA.runtime_version()
test = CuArray([1,2,3])
```

## 4) Configure MPI.jl

## 5) Final precompilation
- To ensure compilation doesn't occur when launching simulations, perform one final precompilation:
```
using Pkg; Pkg.precompile()
```
