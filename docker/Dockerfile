# FEniCS/DOLFINx research environment for DarkVision ultrasound wave-sim RnD.
# Base image = official DOLFINx stable: Python + C++ FEM stack, PETSc, MPI, Basix/UFL/FFCx.
FROM dolfinx/dolfinx:stable

# Extra Python deps (meshing, visualization, IO) kept out of the base image.
# numpy / mpi4py / petsc4py are already provided by the base image.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /work
