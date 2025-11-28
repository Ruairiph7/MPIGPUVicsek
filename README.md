Did CUDA, then MPI. Need to set CUDA.runtime_version; take from nvidia-smi//maybe module avail/load stuff. For MPI, need to use MPIPreferences to set to system version.. Then do Pkg.build for MPI, plus instantiate and precompile

Note this was written for Julia Version 1.11.1 (2024-10-16)
