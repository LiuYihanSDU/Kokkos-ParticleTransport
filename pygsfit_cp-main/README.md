# pygsfit_cp

`pygsfit_cp` is a radio imaging spectroscopy package dedicated to performing spectrum fitting. 
### INSTALLATION:
### New conda/virtual env (highly recommanded) please replace myenv with a desired name
`conda create --name myenv`
`conda activate myenv`
`pip install pygsfit_cp`

### USAGE:
Step1: There are two ways to do step 1. 
## Method One (Recommended)

a. **Initial Setup and Exploration**
   - Run `/demo/pygsfit_cp.ipynb` to identify appropriate initial gas parameters and fitting ranges for each parameter.
   - Define and visualize the Region of Interest (ROI).

b. **Export Batch Script**
   - After obtaining the ideal fitting results, export a batch mode fitting script to perform the fitting with the same parameters.

## Method Two

a. **Direct Batch Mode**
   - If you are already familiar with your data, directly run `/demo/batch_mode_script.py`.
   - Manually input all the parameters of the ROI.

## Common Steps for Both Methods

2. **Execute Batch Script**
   - Execute the batch script to obtain the fitting results, which include:
     - A results file (`fitting_results.h5`)
     - An IDL script (`make_params_maps.pro`)
     - Several intermediate files.

3. **Generate Parameter Maps**
   - Run the IDL script (`make_params_maps.pro`) directly in `sswidl` to generate an IDL save file (`parameters_map.sav`).
   - The resulting save file can be opened in `gsfitview`, `sswidl`.
