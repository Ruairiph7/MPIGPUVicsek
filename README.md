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

- If on a HPC system, begin by launching an interactive job, and then you may need to first load a CUDA module via e.g. "module load cuda/xx.x.x", or if the system uses NVHPC to bundle CUDA together with other tools such as CUDA-aware MPI, then e.g. "module load nvhpc/xx.x". If you intend to use multiple GPUs, then you will later need CUDA-aware MPI, so choose a CUDA installation which is compatible.

- Run Julia from a node/machine with a GPU (inside your interactive job if on a HPC system), then load the CUDA.jl package. Note that you will need to wait a while for it to precompile.

```
using CUDA
```

- Next, configure CUDA.jl to use your local CUDA installation via:

```
CUDA.set_runtime_version!(local_toolkit=true)
```

- If this does not work, you can try to explicitly set the CUDA runtime version to match the version of CUDA you will be using. On a HPC system this could be the version you load using e.g. ```module load cuda/xx.x.x```, or it could be the version found with the ```nvidia-smi``` command. For example, if it is CUDA 13.0.0, then call:
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

- If you only intend to use a single GPU, then the code requires an MPI installation, but you do not need to worry about specifics or if it is CUDA-aware. If ```which mpirun``` returns a path, then it should be sufficient. If it does not, then on an HPC system you may have to first load an MPI module via e.g. "module load openmpi/x.x.x".

- If using multiple GPUs, you need a CUDA-aware MPI installation that is compatible with your CUDA version. Setting this up can be very fiddly and different on each system, but the following is a good starting point. 

- On HPC systems, CUDA and CUDA-aware MPI may come as separate modules you must load, or they may be bundled together into a single NVHPC module. Once you find an MPI installation, you can test whether it should be CUDA-aware with the following shell commands:
```
ompi_info | grep -i cuda
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
```

- If these prove successful, then to configure MPI.jl first make sure that the installation is in the correct PATH environment variables outside of Julia. If ```which mpirun``` returns ```/xxx/yyy/zzz/bin/mpirun```, then call:
```
export PATH=/xxx/yyy/zzz/bin/:$PATH
export LD_LIBRARY_PATH=/xxx/yyy/zzz/lib:$LD_LIBRARY_PATH
```
Note that these commands, on top of any calls to ```module load``` that you have used, will need to be added to any job scripts before launching julia on a HPC system.

- Next load Julia and tell it which MPI installation to use:
```
using MPIPreferences
MPIPreferences.use_system_binary(
    mpiexec="/xxx/yyy/zzz/bin/mpirun",
    library_names=["libmpi.so"],
)
```

- If this does not return any error messages, then restart Julia to implement the changes and check that the correct MPI installation is being used:
```
using Pkg
Pkg.build("MPI")
using MPI

MPI.versioninfo()
```

- Now you can test whether your configuration is working with a script like the following:
```
using MPI
using CUDA

MPI.Init()

println("Hello from rank ", MPI.Comm_rank(MPI.COMM_WORLD))

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

sendbuf = CUDA.fill(Float32(rank + 1), 4)
recvbuf = CUDA.zeros(Float32, 4)

MPI.Allreduce!(sendbuf, recvbuf, MPI.SUM, comm)

synchronize()

println("Rank $rank sees: ", Array(recvbuf))

MPI.Finalize()
```
- This should be launched like ```mpirun -n 2 julia --project=/path/to/MPIGPUVicsek scriptname.jl```, and return a result that looks like:
```
Hello from rank 0
Hello from rank 1
Rank 0 sees: Float32[3.0, 3.0, 3.0, 3.0]
Rank 1 sees: Float32[3.0, 3.0, 3.0, 3.0]
```

## 5) Final precompilation
- To ensure compilation doesn't occur when launching simulations, perform one final precompilation:
```
using Pkg; Pkg.precompile()
```
