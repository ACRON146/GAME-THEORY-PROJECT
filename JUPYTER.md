# Run the project in Jupyter

1. Install Jupyter Notebook or JupyterLab in the Python environment you want to use.
2. Open `notebooks/Run_Project.ipynb` from this project folder.
3. Run the notebook cells from top to bottom. If project libraries are missing, run the dependency installation cell.
4. Keep `RUN_MODE = "full"` for the complete configured study, or set it to `"smoke"` for a shorter pipeline check.

The notebook calls the existing project modules and uses `configs/default.yaml`; it does not replace the command-line interface. The first data preparation step downloads UCI Adult dataset files if needed. Full experiments may take a long time.
