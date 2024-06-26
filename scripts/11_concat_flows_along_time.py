#%%
"""
# openTCAMV -- 11_concat_flows_along_time.py
If the input files are generated by the script `10_conduct_tracking` with 
multiprocessing on the time axis, the output files are separated by the time axis. 
This script concatenates the output files along the time axis to create an input file for `20_finalize_tracking.py`.

## Data requirement
Input file: NetCDF files generated by `10_conduct_tracking.py`

## Example usage
$ python 11_concat_flows_along_time.py ../sample/2017_Lan_ns7_nt1_rot0.0007_it*.nc --exclude_texts concat -o ../sample/2017_Lan_ns7_nt1_concat.nc

## Reference
Tsukada, T., Horinouchi, T., & Tsujino, S. (2024). Wind distribution in the eye of tropical cyclone revealed by a novel atmospheric motion vector derivation. Journal of Geophysical Research: Atmospheres, 129, e2023JD040585. https://doi.org/10.1029/2023JD040585
"""

import os
import glob
import logging
import argparse

import numpy as np
import xarray as xr
import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger.info(f"PID: {os.getpid()}")


parser = argparse.ArgumentParser(description="Append netCDF files along `time` axis.")
parser.add_argument("glob_strings", type=str, help="File pattern to match netCDF files")
parser.add_argument("--exclude_texts", type=str, nargs="+", help="Exclude the filenames including this text from concat list")
parser.add_argument("--drop_vars", type=str, nargs="+", help="Variable names included in the output file")
parser.add_argument("-o", "--oname", type=str, help="Output filename")

sample_dir = f"{os.path.dirname(__file__)}/../sample"
test_args = f"{sample_dir}/2017_Lan_ns7_nt1_rot0.0000_it*.nc".split()

try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    args = parser.parse_args(test_args)
except:
    args = parser.parse_args()

#%% Find files
fnames = pd.Series(sorted(glob.glob(args.glob_strings)))

if len(fnames) == 0:
    raise FileNotFoundError(f"No files found with the pattern: {args.glob_strings}")

logger.info(f"Found {len(fnames)} files in {os.path.dirname(os.path.abspath(args.glob_strings))}/")

if args.exclude_texts:
    for except_text in args.exclude_texts:
        exclude_indices = fnames.str.contains(except_text)
        fnames = fnames[~exclude_indices]

for i, fname in enumerate(fnames):
    logger.info(f"Target {i}: {os.path.basename(fname)}")

if len(fnames) >= 20:
    logger.warning("Too many files to concat. Please make sure that is what you want.")
#%%
logger.info("Concatenating files...")
all = None
for fname in fnames:
    ds = xr.open_dataset(fname).swap_dims({"it":"time"})
    if args.drop_vars:
        ds = ds.drop_vars(args.drop_vars)
    if all is None:
        all = ds
    else:
        all = xr.concat([all, ds], "time")
#%%
all = all.drop_duplicates("time")
ditstep = all["it"][1]-all["it"][0]
all["it"] = xr.DataArray(np.arange(all["it"].min(), all["it"].min()+(all["it"].size-1)*ditstep+ditstep, ditstep), dims="time")
all = all.swap_dims({"time":"it"})

#%%
if args.oname is None:
    args.oname = f"{os.path.dirname(os.path.abspath(args.glob_strings))}/{os.path.basename(args.glob_strings).replace('*','_concat')}"
all.to_netcdf(args.oname)
logger.info(f"Saved to {args.oname}")
# %%
