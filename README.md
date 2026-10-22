# csc930m-project

## AI usage log

| Tool | Use and outcome |
| --- | --- |
| OpenAI GPT-6 | Used AI to brainstorm and break tasks down for pair delegation. AI drafted the description of issues, including scope, acceptance criteria, and dependencies. Anthony and Ramon reviewed and approved the breakdown.|
| Google Antigravity | Used as a reference summary and a grammar checker for the Phase 1 requirements and dataset description. |
| GitHub Copilot | Used to generate git commit messages. |

## JupyterHub setup

On JupyterHub, clone the repository and place StreetSurfaceVis under `data` (`streetSurfaceVis_v1_0.csv` and `s_256`). Create the environment with `python -m venv --system-site-packages ~/.venvs/csc930m-knn-gpu`, then use its Python to run `-m pip install -r requirements-gpu.txt` and `-m ipykernel install --user --name csc930m-knn-gpu --display-name "CSC930M GPU KNN"`. Select that kernel before execution. It inherits course packages and pins PyTorch 2.14.0 with CUDA 12.6 for V100 support.
