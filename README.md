# openTCAMV
Open source TC-specific AMV derivation scripts

## References
- Tsukada, T., Horinouchi, T., & Tsujino, S. (2024). Wind distribution in the eye of tropical cyclone revealed by a novel atmospheric motion vector derivation. Journal of Geophysical Research: Atmospheres, 129, e2023JD040585. https://doi.org/10.1029/2023JD040585
- Horinouchi, T., S. Tsujino, M. Hayashi, U. Shimada, W. Yanase, A. Wada, and H. Yamada, 2023: Stationary and Transient Asymmetric Features in Tropical Cyclone Eye with Wavenumber-1 Instability: Case Study for Typhoon Haishen (2020) with Atmospheric Motion Vectors from 30-Second Imaging. Monthly Weather Review, 151, 253–273, https://doi.org/10.1175/MWR-D-22-0179.1.

## Version history
- 1.0.0 (2024-05-30): Initial release

## Directory structure
```
openTCAMV/
├── README.md
├── requirements.txt
├── docs
│   └── sample
│       ├── README.md
│       └── AMVs_it24.png
├── sample
│   ├── sample.sh
│   └── (2017_Lan_aeqd_sample.nc) # see docs/sample/README.md
└── scripts
    ├── 10_conduct_tracking.py
    ├── 11_concat_flows_along_time.py
    ├── 20_finalize_tracking.py
    └── 30_plot_velocity2d.py
```

## Dependencies
* numpy
* scipy
* pandas
* xarray
* dask
* netcdf4
* pyVTTrac

## Installation
`openTCAMV` is not a package, just a collection of scripts. Users can download the repository and use the scripts directly. `$` means the command line prompt; do not type it.
```bash
$ git clone https://github.com/tsukada-cs/openTCAMV.git
```

To use the scripts, users need to install the required libraries. The following command installs the required libraries.
```bash
$ pip install -r requirements.txt
```
For installing `pyVTTrac`, please refer to the following steps.
```
$ git clone --recurse-submodules https://github.com/tsukada-cs/pyVTTrac.git
$ cd pyVTTrac
$ pip install .
```

## Tutorial
There is sample data and code available for a tutorial.  
Please see [docs/sample/README.md](docs/sample/README.md)


## Scripts
### **10_conduct_tracking.py**
<details><summary>Click here to open the description</summary>

This script perform the cloud tracking for satellite imagery.
The tracking is conducted using the `pyVTTrac` library, which is a cloud tracking algorithm based on the template matching method.

For the details of the tracking algorithm, please refer to the [Tsukada et al. (2024)](https://doi.org/10.1029/2023JD040585).

#### Data requirements
Input file: NetCDF file that contains a tracked variable with (time, y, x) dimensions.
The x and y dimensions should have "units" attribute with "km".

#### Usage
**Basic usage**
```bash
$ python 10_conduct_tracking.py ifn [-s start] [-e end] [-o ofn] [-n ntrac] [--ward ward] [--tidstep tidstep] [--traj_int traj_int] [-v vagg] [--polar] [--use_init_temp] [--no_subgrid] [--itran itran] [--xgran xgran] [--xint xint] [--ygran ygran] [--yint yint] [--rgran rgran] [--rint rint] [--nath nath] [--ns ns] [--nsx nsx] [--nsy nsy] [--Vd Vd] [--Td Td] [--Vth Vth] [--Vs Vs] [--hs hs] [--Vc Vc] [--vlim vlim] [--Sth0 Sth0] [--Sth1 Sth1] [--Cth Cth] [--peak_inside_th peak_inside_th] [--itstep itstep] [--varname varname] [--maskvar maskvar] [--mask_lower_lim mask_lower_lim] [--mask_upper_lim mask_upper_lim] [--min_samples min_samples] [--out_subimage] [--out_score_ary] [--out_psr] [--sector sector] [--dtlimit dtlimit] [--revrot revrot] [--record_initpos record_initpos] [--record_alongtraj record_alongtraj] [--cth cth] [--out_cthmin] [--out_cthmax] [--complevel complevel]
```
**Example**
```bash
$ python 10_conduct_tracking.py ../sample/2017_Lan_aeqd_sample.nc --revrot 0.0020 --varname=B03 --ns=7 --ntrac=1 --Sth0=0.7 -o=../sample/2017_Lan_ns7_nt1_test.nc --ygran=-45:45 --xgran=-45:45 --traj_int=1 --Vs=10 --record_initpos cth B03 B13 B14 --out_cthmax --Vc=20 --Vd=20 --Td=60 --Vth=5
```

#### Command line arguments
| Argument           | Type  | Default  | Description                                                                                                                                                                                          | Choices                     |
| ------------------ | ----- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| ifn                | str   |          | file path to input NetCDF file                                                                                                                                                                       |                             |
| -s, --start        | str   |          | start time in yyyymmddTHHMMSS format                                                                                                                                                                 |                             |
| -e, --end          | str   |          | end time in yyyymmddTHHMMSS format                                                                                                                                                                   |                             |
| -o, --ofn          | str   | ./tmp.nc | output NetCDF file path                                                                                                                                                                              |                             |
| -n, --ntrac        | int   | 2        | The number of tracking for both forward and backward tracking                                                                                                                                        |                             |
| --ward             | str   | bothward | time direction for tracking                                                                                                                                                                          | bothward, forward, backward |
| --tidstep          | int   | 1        | time index interval of initial time for start tracking (1 means every time index)                                                                                                                    |                             |
| --traj_int         | int   |          | time index interval for output of trajectory                                                                                                                                                         |                             |
| -v, --vagg         | str   | mean     | how to aggregate the vectors; 'org' for original vectors without any aggregations, 'mean' for the velocity be averaging vectors, 'startend' for the veclocity by connecting the start and end points | org, mean, startend         |
| --polar            | bool  |          | if specified, use polar coordinates points as initial template positioning, if not use Cartesian grid                                                                                                |                             |
| --use_init_temp    | bool  |          | use initial template through tracking without updating the template                                                                                                                                  |                             |
| --no_subgrid       | bool  |          | if specified, do not perform a subgrid estimation                                                                                                                                                    |                             |
| --itran            | slice |          | time-axis colon-separated slice of initial time for start tracking, with higher priority over --start and --end                                                                                      |                             |
| --xgran            | slice | -50:50   | x-axis colon-separated slice of initial template positions (with interval of --xint)                                                                                                                 |                             |
| --xint             | float | 1.0      | x-axis interval of initial template positions in the (equally spaced grid)                                                                                                                           |                             |
| --ygran            | slice | -50:50   | y-axis colon-separated slice of initial template positions (with interval of --yint)                                                                                                                 |                             |
| --yint             | float | 1.0      | y-axis interval of initial template positions (equally spaced grid)                                                                                                                                  |                             |
| --rgran            | slice | 4:50     | r-axis colon-separated slice of initial template positions (equally spaced grid)                                                                                                                     |                             |
| --rint             | int   | 1        | r-axis interval of initial template positions (equally spaced grid)                                                                                                                                  |                             |
| --nath             | int   | 60       | number of azimuthal initial template positions in polar coordinates                                                                                                                                  |                             |
| --ns               | int   | 11       | template size in pixel dimension                                                                                                                                                                     |                             |
| --nsx              | int   |          | template width with higher priority over --ns                                                                                                                                                        |                             |
| --nsy              | int   |          | template height with higher priority over --ns                                                                                                                                                       |                             |
| --Vd               | float | 20.0     | threshold to limit the maximum velocity difference between velocities obtained from forward and backward tracking as vectors                                                                         |                             |
| --Td               | float |          | threshold to limit the maximum angle difference between velocities obtained from forward and backward tracking as vectors                                                                            |                             |
| --Vth              | float | 5.0      | threshold speed for screening with --Td                                                                                                                                                              |                             |
| --Vs               | float | 80.0     | search range for cloud tracking in velocity dimension (m/s)                                                                                                                                          |                             |
| --hs               | int   |          | Search range for cloud tracking in pixel count with higher priority over --Vs                                                                                                                        |                             |
| --Vc               | float | 20.0     | threshold to limit the maximum velocity change between consecutive images                                                                                                                            |                             |
| --vlim             | float | 120.0    | Threshold to limit the maximum speed (m/s)                                                                                                                                                           |                             |
| --Sth0             | float | 0.8      | minimum score required for the first-time tracking                                                                                                                                                   |                             |
| --Sth1             | float | 0.8      | minimum score required for the subsequent tracking                                                                                                                                                   |                             |
| --Cth              | float | 3        | minimum contrast to track the template                                                                                                                                                               |                             |
| --peak_inside_th   | float |          |                                                                                                                                                                                                      |                             |
| --itstep           | int   | 1        | if >1, skip                                                                                                                                                                                          |                             |
| --varname          | str   | B03      | variable name of tracking target                                                                                                                                                                     |                             |
| --maskvar          | str   |          | variable name for creating mask                                                                                                                                                                      |                             |
| --mask_lower_lim   | float |          | lower limit for mask variable. --maskvar <= --lower_limit will be ignored when scoring                                                                                                               |                             |
| --mask_upper_lim   | float |          | upper limit for mask variable. --maskvar >= --upper_limit will be ignored when scoring                                                                                                               |                             |
| --min_samples      | int   | 1        | minimum number of valid values to calculate score when using mask                                                                                                                                    |                             |
| --out_subimage     | bool  |          | if output subimages                                                                                                                                                                                  |                             |
| --out_score_ary    | bool  |          | if output score array                                                                                                                                                                                |                             |
| --out_psr          | bool  |          | if output Peak-To-Sidelobe ratio of the score field                                                                                                                                                  |                             |
| --sector           | str   |          | limiting sectors used for tracking                                                                                                                                                                   |                             |
| --dtlimit          | float | 200.0    | specify maximum dt (in seconds)                                                                                                                                                                      |                             |
| --revrot           | float | 0.0      | angular velocity to rotate images (in rad/s). Positive (negative) value make crockwise (counterclocwise) rotation over time                                                                          |                             |
| --record_initpos   | str   |          | Record specified variable at their initial position                                                                                                                                                  |                             |
| --record_alongtraj | str   |          | Record specified variable along their trajectory                                                                                                                                                     |                             |
| --cth              | str   | cth      | cloud-top height variable name                                                                                                                                                                       |                             |
| --out_cthmin       | bool  |          | if output minimum cloud-top height along each tracking                                                                                                                                               |                             |
| --out_cthmax       | bool  |          | if output maximum cloud-top height along each tracking                                                                                                                                               |                             |
| --complevel        | int   | 3        | compression level for output NetCDF file                                                                                                                                                             |                             |
</details>
<hr>

### **11_concat_flows_along_time.py**
<details><summary>Click here to open the description</summary>

If the input files are generated by the script `10_conduct_tracking` with multiprocessing on the time axis, the output files are separated by the time axis. This script concatenates the output files along the time axis to create an input file for `20_finalize_tracking.py`.

#### Data requirements
    Input file: NetCDF files generated by `10_conduct_tracking.py`

#### Usage
**Basic usage**
```bash
$ python 11_concat_flows_along_time.py glob_strings [--exclude_texts exclude_texts] [--drop_vars drop_vars] [-o oname]
```
**Example**
```bash
$ python 11_concat_flows_along_time.py ../sample/2017_Lan_ns7_nt1_rot0.0007_it*.nc --exclude_texts concat -o ../sample/2017_Lan_ns7_nt1_concat.nc
```


#### Command line arguments
| Argument        | Type | Default | Description                                                | Choices |
| --------------- | ---- | ------- | ---------------------------------------------------------- | ------- |
| glob_strings    | str  |         | File pattern to match netCDF files                         |         |
| --exclude_texts | str  |         | Exclude the filenames including this text from concat list |         |
| --drop_vars     | str  |         | Variable names included in the output file                 |         |
| -o, --oname     | str  |         | Output filename                                            |         |

"""
</details>
<hr>

### **20_finalize_tracking.py**
<details><summary>Click here to open the description</summary>

This script finalizes the tracking results by selecting the best solution from the candidate solutions.
The candidate solutions are generated by the script `10_conduct_tracking.py`.

#### Data requirements
Input file: NetCDF files generated by `10_conduct_tracking.py`

#### Usage
**Basic usage**
```bash
$ python 20_finalize_tracking.py ifn_ns<ns>_nt1_rot<omega>.nc [--ns ns] [--omega omega] [-o ofn] [--Tw Tw] [--Hw Hw] [--xw xw] [--yw yw] [--rw rw] [--aw aw] [--dth dth] [--dc dc] [--useV] [-e max_epoch] [--IRdiff IRdiff] [--dIR dIR] [--cth cth] [--cthmin cthmin] [--cthmax cthmax] [--score_th score_th] [--out_final_omega] [--out_final_ns] [--out_final_medians] [--priority priority] [--exclude exclude] [--ref_flows ref_flows] [--tau_stri tau_stri] [--v_stri v_stri] [--cth_stri cth_stri] [--omega_stri omega_stri]
```
**Example 1**
```bash
$ python 20_finalize_tracking.py ../sample/2017_Lan_ns7_nt1_rot<omega>.nc --ns 7 --omega 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025 --cthmax 10 --exclude stf stb score_ary psr -o ../sample/2017_Lan_ns7_nt1_ref.nc
```
**Example 2** (with reference flows to apply striations treatment using the result of Example 1)
```bash
$ python 20_finalize_tracking.py ../sample/2017_Lan_ns7_nt1_rot<omega>.nc --ns 7 --omega 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025 --cthmax 10 --exclude stf stb score_ary psr --ref_flows ../sample/2017_Lan_ns7_nt1_ref.nc --omega_stri 0.0015 -o ../sample/2017_Lan_ns7_nt1.nc
```

#### Command line arguments
| Argument            | Type    | Default | Description                                                                                                                                                                                                   | Choices      |
| ------------------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| ifns_rule           | str     |         | filename format used in f-string using <ns> as template size and <omega> as Ω_i                                                                                                                               |              |
| --ns                | int     |         | ns list                                                                                                                                                                                                       |              |
| --omega             | str     |         | Ω_i list                                                                                                                                                                                                      |              |
| -o, --ofn           | str     |         | output filename                                                                                                                                                                                               |              |
| --Tw                | float   | 1210    | time period of median filtering window (in sec)                                                                                                                                                               |              |
| --Hw                | float   | 6       | width & height of median filtering window for Cartesian grid (in km)                                                                                                                                          |              |
| --xw                | float   |         | x width of median filtering window for Cartesian grid (in km)                                                                                                                                                 |              |
| --yw                | float   |         | y height of median filtering window for Cartesian grid (in km)                                                                                                                                                |              |
| --rw                | float   | 6       | r width of median filtering window for polar grid (in km)                                                                                                                                                     |              |
| --aw                | float   | 30      | azimuth width of median filtering window for polar grid (in degree)                                                                                                                                           |              |
| --dth               | float   | 10      | threshold to limit the maximum d2 for rejection of candidates (in m/s)                                                                                                                                        |              |
| --dc                | float   | 0.5     | coefficient to be multiplied by the L2 norm of median wind speed for rejection of candidates (0: all rejected, 0.5: half length, 1: same length)                                                              |              |
| --useV              | boolean |         | if specified, use absolute wind speed for median filtering, not using each velocity component                                                                                                                 |              |
| -e, --max_epoch     | float   | 20      | the maximum number of epochs for updating the solutions                                                                                                                                                       |              |
| --IRdiff            | str     |         | the formula for IR difference. It evaluates the difference between two variables parsed with a minus sign. The grid with `abs(IR1-IR2) >= --dIR` are masked                                                   |              |
| --dIR               | float   | 2       | threshold to limit maximum IR difference (in K). The values larger than it will be masked.                                                                                                                    |              |
| --cth               | str     | cth     | cloud top height variable name                                                                                                                                                                                |              |
| --cthmin            | float   |         | threshold to limit the minimum cloud top height at initial position (in km)                                                                                                                                   |              |
| --cthmax            | float   |         | threshold to limit the maximum cloud top height at initial position (in km)                                                                                                                                   |              |
| --score_th          | float   |         | threshold to limit the minimum score of the candidates                                                                                                                                                        |              |
| --out_final_omega   | boolean |         | output `final_omega` which is the used omega values in final solutions                                                                                                                                        |              |
| --out_final_ns      | boolean |         | output `final_ns` which is the used ns in final solutions                                                                                                                                                     |              |
| --out_final_medians | boolean |         | output `final_medians` which is the final median wind speeds                                                                                                                                                  |              |
| --priority          | str     | score   | priority for selecting the best solution. `score` selects the solution with the highest score. `dangv` selects the solution with the smallest angular velocity difference from the reference angular velocity | score, dangv |
| --exclude           | str     |         | drop specified variables from the final output file                                                                                                                                                           |              |
| --ref_flows         | str     |         | Reference flows to apply omega_stri procedure                                                                                                                                                                 |              |
| --tau_stri          | int     | 24      | temporal window size (in index space) to compute the velocity contrast of --ref_flows for detection of striations                                                                                             |              |
| --v_stri            | float   | 20      | threshold to limit the maximum velocity contrast for detection of striations (m/s)                                                                                                                            |              |
| --cth_stri          | int     |         | threshold to limit the maximum cloud top height to get reference contrast                                                                                                                                     |              |
| --omega_stri        | float   | 0.0     | threshold to limit the minimum Ω_i for the striation grids (rad/s)                                                                                                                                            |              |
</details>
<hr>

### **30_plot_velocity2d.py**
<details><summary>Click here to open the description</summary>

This script plots the 2D velocity field on the map.

#### Data requirements
Input file: NetCDF file generated by `20_finalize_tracking.py`

#### Usage
**Basic usage**
```bash
$ python 30_plot_velocity2d.py ifn [--it it] [--sstep sstep] [--vector_duration vector_duration] [--key_speed key_speed] [-o odir]
```
**Example**
```bash
$ python 30_plot_velocity2d.py ../sample/2017_Lan_ns7_nt1.nc --it=24
```

#### Command line arguments
| Argument            | Type | Default | Description                                                                                                  | Choices |
| ------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------ | ------- |
| "ifn"               |      |         | Directory containing sample data                                                                             |         |
| "--it"              | int  | 24      | Time index to plot                                                                                           |         |
| "--sstep"           | int  | 2       | Spacing between drawn vectors                                                                                |         |
| "--vector_duration" | int  | 90      | Duration of each vector in seconds. If 90, then vector length corresponds to the travel length during 90 sec |         |
| "--key_speed"       | int  | 50      | Speed of the key vector                                                                                      |         |
| "-o", "--odir"      | str  |         | Output directory. If None, make 'outputs' directory in the same directory of the input file                  |         |
</details>
<hr>