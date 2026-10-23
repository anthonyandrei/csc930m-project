# csc930m-project

## JupyterHub setup

On JupyterHub, clone the repository and place StreetSurfaceVis under `data` (`streetSurfaceVis_v1_0.csv` and `s_256`). Create the environment with `python -m venv --system-site-packages ~/.venvs/csc930m-knn-gpu`, then use its Python to run `-m pip install -r requirements-gpu.txt` and `-m ipykernel install --user --name csc930m-knn-gpu --display-name "CSC930M GPU KNN"`. Select that kernel before execution. It inherits course packages and pins PyTorch 2.14.0 with CUDA 12.6 for V100 support.
