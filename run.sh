#!/bin/bash
#SBATCH -J mirg
#SBATCH -o mirg.o%j
#SBATCH -t 99:00:00
#SBATCH -n 1
#SBATCH --gpus=2

module add Miniforge3/py3.10
module add cudatoolkit/12.4
source activate /project/hoskere/jkgao/.cache/conda/mirg

export CONDA_PKGS_DIRS=/project/hoskere/jkgao/.cache/conda_pkgs_dir/
export XDG_CACHE_HOME=/project/hoskere/jkgao/.cache/cache/
export PYTHONPATH=/project/hoskere/jkgao/.cache/local/

nvidia-smi
python /project/hoskere/jkgao/mirg/data/public_inspection_reports/prepare.py