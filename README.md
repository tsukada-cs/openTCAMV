# openTCAMV
Open source TC-specific AMV derivation scripts.

## References
[![T24](https://img.shields.io/badge/T24-10.1029/2023JD040585-blue.svg?logo=doi)](https://doi.org/10.1029/2023JD040585)
[![H23](https://img.shields.io/badge/H23-10.1175/MWR--D--22--0179.1-blue.svg?logo=doi)](https://doi.org/10.1175/MWR-D-22-0179.1)
- Tsukada, T., Horinouchi, T., & Tsujino, S. (2024). Wind distribution in the eye of tropical cyclone revealed by a novel atmospheric motion vector derivation. Journal of Geophysical Research: Atmospheres, 129, e2023JD040585. https://doi.org/10.1029/2023JD040585  
- Horinouchi, T., S. Tsujino, M. Hayashi, U. Shimada, W. Yanase, A. Wada, and H. Yamada, 2023: Stationary and Transient Asymmetric Features in Tropical Cyclone Eye with Wavenumber-1 Instability: Case Study for Typhoon Haishen (2020) with Atmospheric Motion Vectors from 30-Second Imaging. Monthly Weather Review, 151, 253–273, https://doi.org/10.1175/MWR-D-22-0179.1.  

## Version history
[![2.0.0](https://img.shields.io/badge/2026--08--08-2.0.0-purple.svg?logo=github)](https://github.com/tsukada-cs/openTCAMV/releases/tag/2.0.0)
[![1.0.0](https://img.shields.io/badge/2024--05--30-1.0.0-purple.svg?logo=github)](https://github.com/tsukada-cs/openTCAMV/releases/tag/1.0.0)

v2.0.0 migrates to pyVTTrac v2 (native Fortran core, no Julia dependency)
and moves the core tracking logic into an installable `opentcamv/` package.
See `CHANGELOG.md` for the full list of changes, including bug fixes that
change output versus v1 -- if you're upgrading from v1, read it before
comparing results.

## Directory structure
```
openTCAMV/
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── pyproject.toml
├── opentcamv/              # core tracking logic (installable package)
│   ├── cli.py               # argparse definition + normalization
│   ├── data.py               # input reading, time axis, Grid, mask
│   ├── params.py             # TrackingSetup (derived parameters)
│   ├── output.py             # output coords/Dataset/attrs, write, combine_omega
│   ├── rotation.py           # --revrot window rotation
│   ├── tracking.py           # per-tid0 time-window slicing + pyvttrac.track()
│   ├── aggregate.py          # --vagg, trajectory concat, polar conversion
│   ├── screening.py          # revrot restore, --vlim/--Vd/--Td/--Vth
│   ├── records.py            # --record_initpos/--record_alongtraj/--out_cthmin/max/psr
│   ├── concat.py             # time-axis concatenation (11_'s logic)
│   ├── trajectory_movie.py   # trajectory animation (31_'s logic; self-contained, copy-and-use)
│   └── finalize/             # candidate selection (20_'s logic)
│       ├── io.py               # load_candidates: (ns, omega, ...) working dataset
│       ├── window.py           # median-filter window sizes
│       ├── masking.py          # cth/IRdiff/score-threshold/striation masking
│       ├── candidates.py       # priority computation, final assembly
│       └── selectors/          # iterative_median (default), relaxation (skeleton)
├── scripts/                # thin CLI drivers over opentcamv/
│   ├── 10_conduct_tracking.py
│   ├── 11_concat_flows_along_time.py
│   ├── 20_finalize_tracking.py
│   ├── 30_plot_velocity2d.py
│   └── 31_animate_trajectories.py
├── tests/
├── docs
│   └── sample
│       ├── README.md
│       ├── AMVs_it24.png
│       └── trajectories.gif
└── sample
    ├── sample.sh
    └── (2017_Lan_aeqd_sample.nc) # see docs/sample/README.md
```

## Dependencies
* numpy
* scipy
* pandas
* xarray
* dask
* pillow
* netcdf4
* pyVTTrac >= 2.2
* matplotlib (only for `30_plot_velocity2d.py` and `31_animate_trajectories.py`)
* ffmpeg (only for `.mp4` movies from `31_animate_trajectories.py`; `.gif` needs nothing extra)

Target platforms are Linux and macOS (pyVTTrac v2's native Fortran core doesn't support Windows).

## Installation
```bash
$ git clone https://github.com/tsukada-cs/openTCAMV.git
$ pip install -e openTCAMV
```
This installs `opentcamv` (editable) along with its dependencies, including `pyVTTrac` -- a plain `pip install pyVTTrac` is all that library itself needs now (no Julia, no git submodule). `scripts/*.py` can then be run directly; they `import opentcamv` without any `sys.path` setup.

For the plotting scripts (`30_plot_velocity2d.py`, `31_animate_trajectories.py`), also run `pip install matplotlib` (or `pip install -e "openTCAMV[plot]"`).

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
$ python 10_conduct_tracking.py ifn [-s start] [-e end] [-o ofn] [-n ntrac] [--ward ward] [--tidstep tidstep] [--traj_int traj_int] [-v vagg] [--polar] [--use_init_temp] [--no_subgrid] [--subgrid subgrid] [--method method] [--workers workers] [--itran itran] [--xgran xgran] [--xint xint] [--ygran ygran] [--yint yint] [--rgran rgran] [--rint rint] [--nath nath] [--ns ns [ns ...]] [--nsx nsx] [--nsy nsy] [--Vd Vd] [--Td Td] [--Vth Vth] [--Vs Vs] [--hs hs] [--Vc Vc] [--vlim vlim] [--Sth0 Sth0] [--Sth1 Sth1] [--Cth Cth] [--peak_inside_th peak_inside_th] [--itstep itstep] [--varname varname] [--maskvar maskvar] [--mask_lower_lim mask_lower_lim] [--mask_upper_lim mask_upper_lim] [--min_samples min_samples] [--out_subimage] [--out_score_ary] [--out_psr] [--sector sector] [--dtlimit dtlimit] [--ref_dt ref_dt] [--revrot revrot [revrot ...]] [--split_omega] [--record_initpos record_initpos] [--record_alongtraj record_alongtraj] [--cth cth] [--out_cthmin] [--out_cthmax] [--complevel complevel]
```
**Example** (single Omega)
```bash
$ python 10_conduct_tracking.py ../sample/2017_Lan_aeqd_sample.nc --revrot 0.0020 --varname=B03 --ns=7 --ntrac=1 --Sth0=0.7 -o=../sample/2017_Lan_ns7_nt1_test.nc --ygran=-45:45 --xgran=-45:45 --traj_int=1 --Vs=10 --record_initpos cth B03 B13 B14 --out_cthmax --Vc=20 --Vd=20 --Td=60 --Vth=5
```
**Example** (multiple Omega values, looped over within one process; default output is a single file with an `omega` dimension holding every value)
```bash
$ python 10_conduct_tracking.py ../sample/2017_Lan_aeqd_sample.nc --revrot 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025 --varname=B03 --ns=7 --ntrac=1 --Sth0=0.7 -o=../sample/2017_Lan_ns7_nt1.nc --ygran=-45:45 --xgran=-45:45 --traj_int=1 --Vs=10 --record_initpos cth B03 B13 B14 --out_cthmax --Vc=20 --Vd=20 --Td=60 --Vth=5
```
**Example** (same, but one file per Omega instead -- `-o` needs an `<omega>` placeholder)
```bash
$ python 10_conduct_tracking.py ../sample/2017_Lan_aeqd_sample.nc --revrot 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025 --split_omega --varname=B03 --ns=7 --ntrac=1 --Sth0=0.7 -o='../sample/2017_Lan_ns7_nt1_rot<omega>.nc' --ygran=-45:45 --xgran=-45:45 --traj_int=1 --Vs=10 --record_initpos cth B03 B13 B14 --out_cthmax --Vc=20 --Vd=20 --Td=60 --Vth=5
```
**Example** (multiple `ns` values as well -- one file per `ns`, each holding every Omega as above; `-o` needs an `<ns>` placeholder. The `ns` values run one after another, so peak memory is that of a single-`ns` run)
```bash
$ python 10_conduct_tracking.py ../sample/2017_Lan_aeqd_sample.nc --revrot 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025 --varname=B03 --ns 7 9 11 --ntrac=1 --Sth0=0.7 -o='../sample/2017_Lan_ns<ns>_nt1.nc' --ygran=-45:45 --xgran=-45:45 --traj_int=1 --Vs=10 --record_initpos cth B03 B13 B14 --out_cthmax --Vc=20 --Vd=20 --Td=60 --Vth=5
```

#### Command line arguments
| Argument           | Type  | Default  | Description                                                                                                                                                                                          | Choices                     |
| ------------------ | ----- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| ifn                | str   |          | file path to input NetCDF file                                                                                                                                                                       |                             |
| -s, --start        | str   |          | start time in yyyymmddTHHMMSS format                                                                                                                                                                 |                             |
| -e, --end          | str   |          | end time in yyyymmddTHHMMSS format                                                                                                                                                                   |                             |
| -o, --ofn          | str   | ./tmp.nc | output NetCDF file path. Needs a `<ns>` placeholder when more than one `--ns` value is given, and a `<omega>` placeholder when `--split_omega` is set with more than one `--revrot` value             |                             |
| -n, --ntrac        | int   | 2        | The number of tracking for both forward and backward tracking                                                                                                                                        |                             |
| --ward             | str   | bothward | time direction for tracking                                                                                                                                                                          | bothward, forward, backward |
| --tidstep          | int   | 1        | time index interval of initial time for start tracking (1 means every time index)                                                                                                                    |                             |
| --traj_int         | int   |          | time index interval for output of trajectory                                                                                                                                                         |                             |
| -v, --vagg         | str   | mean     | how to aggregate the vectors; 'org' for original vectors without any aggregations, 'mean' for the velocity be averaging vectors, 'startend' for the veclocity by connecting the start and end points | org, mean, startend         |
| --polar            | bool  |          | if specified, use polar coordinates points as initial template positioning, if not use Cartesian grid                                                                                                |                             |
| --use_init_temp    | bool  |          | use initial template through tracking without updating the template                                                                                                                                  |                             |
| --no_subgrid       | bool  |          | alias for `--subgrid=none`                                                                                                                                                                           |                             |
| --subgrid          | str   | paraboloid | subgrid peak-refinement method                                                                                                                                                                     | paraboloid, gaussian, none  |
| --method           | str   | xcor     | score method                                                                                                                                                                                          | xcor, ncov                  |
| --workers          | int   |          | OpenMP thread count. Unset = OpenMP default (usually all cores), 1 = sequential                                                                                                                      |                             |
| --itran            | slice |          | time-axis colon-separated slice of initial time for start tracking, with higher priority over --start and --end                                                                                      |                             |
| --xgran            | slice | -50:50   | x-axis colon-separated slice of initial template positions (with interval of --xint)                                                                                                                 |                             |
| --xint             | float | 1.0      | x-axis interval of initial template positions in the (equally spaced grid)                                                                                                                           |                             |
| --ygran            | slice | -50:50   | y-axis colon-separated slice of initial template positions (with interval of --yint)                                                                                                                 |                             |
| --yint             | float | 1.0      | y-axis interval of initial template positions (equally spaced grid)                                                                                                                                  |                             |
| --rgran            | slice | 4:50     | r-axis colon-separated slice of initial template positions (equally spaced grid)                                                                                                                     |                             |
| --rint             | int   | 1        | r-axis interval of initial template positions (equally spaced grid)                                                                                                                                  |                             |
| --nath             | int   | 60       | number of azimuthal initial template positions in polar coordinates                                                                                                                                  |                             |
| --ns               | int   | [11]     | template size(s) in pixel dimension. Accepts multiple values (`nargs="+"`), looped over one at a time within one process, each writing its own output file(s) (see `-o`)                          |                             |
| --nsx              | int   |          | template width with higher priority over --ns (single `--ns` value only)                                                                                                                             |                             |
| --nsy              | int   |          | template height with higher priority over --ns (single `--ns` value only)                                                                                                                            |                             |
| --Vd               | float | 20.0     | threshold to limit the maximum velocity difference between velocities obtained from forward and backward tracking as vectors (compared directly against Vd -- fixed in v2.0.0, see CHANGELOG.md)     |                             |
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
| --dtlimit          | float | 200.0    | specify maximum time interval (in seconds)                                                                                                                                                           |                             |
| --ref_dt           | float |          | reference time interval for calculating ixhw and iyhw from Vs (in seconds)                                                                                                                           |                             |
| --revrot           | float | [0.0]    | angular velocity/velocities to rotate images (in rad/s). Positive (negative) value make crockwise (counterclocwise) rotation over time. Accepts multiple values (`nargs="+"`), looped over within one process; default output is a single file with all values along an `omega` dimension (see `--split_omega`) |                             |
| --split_omega      | bool  |          | write one file per `--revrot` value (v1-compatible) instead of the default single combined file; requires a `<omega>` placeholder in `-o`/`--ofn`                                                  |                             |
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

If tracking itself was split across processes/machines by time range
(memory constraints, cluster distribution -- **not** for speed: `10_
conduct_tracking.py` already tracks every Omega and `ns` value within a
single process, so there's no per-Omega parallelism left to reassemble
here; for within-process speed use `10_`'s `--workers`), this script
concatenates the resulting `10_conduct_tracking.py` output files along the
`time` axis into a single input file for `20_finalize_tracking.py`. Works
unchanged whether or not the inputs carry `10_`'s `omega`/multi-`ns`
dimensions -- neither is ever the concat axis here.

#### Data requirements
Input file: NetCDF files generated by `10_conduct_tracking.py`

#### Usage
**Basic usage**
```bash
$ python 11_concat_flows_along_time.py [-t target [target ...]] [-g glob_strings] [-e exclude_texts [exclude_texts ...]] [-d drop_vars [drop_vars ...]] [-o oname]
```
**Example**
```bash
$ python 11_concat_flows_along_time.py -g '../sample/2017_Lan_ns7_nt1_it*.nc' --exclude_texts concat -o ../sample/2017_Lan_ns7_nt1_concat.nc
```

#### Command line arguments
| Argument              | Type | Default | Description                                                          | Choices |
| ---------------------- | ---- | ------- | --------------------------------------------------------------------- | ------- |
| -t, --target           | str  |         | NetCDF filenames to concatenate (alternative to `-g`)                 |         |
| -g, --glob_strings     | str  |         | File pattern to match netCDF files (alternative to `-t`)              |         |
| -e, --exclude_texts    | str  |         | Exclude the filenames including this text from the concat list        |         |
| -d, --drop_vars        | str  |         | Variable names to drop from the output file                           |         |
| -o, --oname            | str  |         | Output filename. Derived from `--glob_strings` if omitted             |         |

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
| "ifn"               | str  |         | Directory containing sample data                                                                             |         |
| "--it"              | int  | 24      | Time index to plot                                                                                           |         |
| "--sstep"           | int  | 2       | Spacing between drawn vectors                                                                                |         |
| "--vector_duration" | int  | 90      | Duration of each vector in seconds. If 90, then vector length corresponds to the travel length during 90 sec |         |
| "--key_speed"       | int  | 50      | Speed of the key vector                                                                                      |         |
| "-o", "--odir"      | str  |         | Output directory. If None, make 'outputs' directory in the same directory of the input file                  |         |
</details>
<hr>

### **31_animate_trajectories.py**
This script animates the tracked parcels behind the AMVs: every template moves along its trajectory over the images it was tracked in, leaving a trail that thins with age -- a direct check that the tracking followed the clouds.
The drawing lives in `opentcamv/trajectory_movie.py`, a single self-contained file (numpy + matplotlib only) that you can copy into your own project without installing openTCAMV (see "Using `trajectory_movie.py` in your own code" below).

<details><summary>Click here to open the description</summary>

The movie walks through the images once. The parcels of the AMVs centered on an image move through the `2 ntrac + 1` images of their trajectories (one cycle), then one image is shown alone and the next cycle starts, so every image appears exactly once, at the image cadence. Each trail is widest at the parcel and thins linearly with age. With `--corotate`, every cycle is drawn in a frame rotating with the vortex (it coincides with the storm-centered frame at the AMV time, and the image turns with it), where the motion relative to the rotation -- what `--revrot` tracks -- is easier to follow.

#### Data requirements
* AMV file: NetCDF file generated by `10_conduct_tracking.py` or `20_finalize_tracking.py`, with `xloc`/`yloc` (`rloc`/`aloc` under `--polar`) and `time2`. Track with `--traj_int=1` so that every image along a trajectory is recorded.
* Image file: the NetCDF file that `10_conduct_tracking.py` tracked (default: the AMV file's `ifn` attribute).

#### Usage
**Basic usage**
```bash
$ python 31_animate_trajectories.py ifn [--image image] [--varname varname] [-o ofn] [-s start] [-e end] [--omega omega] [--xlim xmin xmax] [--ylim ymin ymax] [--stride stride] [--checker] [--zoom cx cy half] [--zoom_stride zoom_stride] [--zoom_checker] [--titles titles ...] [--corotate corotate] [--color color] [--clim cmin cmax] [--linewidth linewidth] [--tail_width tail_width] [--log] [--contrast contrast] [--contrast_fwhm contrast_fwhm] [--vlim vmin vmax] [--rings rings ...] [--rest rest] [--fps fps] [--dpi dpi] [--max_frames max_frames] [--title title]
```
**Example** (the tutorial sample: the whole domain plus a zoom on the eye)
```bash
$ python 31_animate_trajectories.py ../sample/2017_Lan_ns7_nt1.nc --log --contrast 0.5 --zoom 0 0 15 --titles "Inner core" "Eye" --rings 10 20
```
**Example** (in a frame co-rotating at 1.5e-3 rad/s, parcels colored by their speed relative to it)
```bash
$ python 31_animate_trajectories.py ../sample/2017_Lan_ns7_nt1.nc --log --contrast 0.5 --corotate 0.0015 --color speed --zoom 0 0 15 --titles "Inner core" "Eye" --rings 10 20 -o ../sample/outputs/2017_Lan_ns7_nt1_corot.mp4
```

#### Command line arguments
| Argument         | Type     | Default                                | Description                                                                                                         |
| ---------------- | -------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| ifn              | str      |                                        | AMV file from `10_conduct_tracking.py` or `20_finalize_tracking.py`                                                  |
| --image          | str      | the AMV file's `ifn` attribute         | image NetCDF the AMVs were tracked in                                                                               |
| --varname        | str      | the AMV file's `varname` attribute     | image variable                                                                                                      |
| -o, --ofn        | str      | `outputs/<ifn name>_trajectories.mp4` next to `ifn` | output movie: `.mp4` (needs ffmpeg) or `.gif`                                                          |
| -s, --start      | str      |                                        | first image to show, e.g. `2017-10-21T02:30`                                                                        |
| -e, --end        | str      |                                        | last image to show                                                                                                  |
| --omega          | float    |                                        | Ω_i candidate to show from a `10_` output holding several (`omega` dimension)                                       |
| --xlim, --ylim   | float x2 | the AMV grid                           | x and y range of the main panel (km)                                                                                |
| --stride         | int      | 2                                      | draw every N-th AMV grid point in the main panel                                                                    |
| --checker        | bool     |                                        | of those, keep every other one in a checkerboard (spacing x sqrt 2)                                                 |
| --zoom           | float x3 |                                        | add a zoom panel: center x, center y, half-width (km); repeatable                                                   |
| --zoom_stride    | int      | 1                                      | `--stride` of the zoom panels                                                                                       |
| --zoom_checker   | bool     |                                        | `--checker` of the zoom panels                                                                                      |
| --titles         | str ...  |                                        | panel titles: the main panel's, then the zoom panels'                                                               |
| --corotate       | float    | 0.0                                    | draw each cycle in a frame rotating counterclockwise at this angular velocity about the center (rad/s; 0 = storm-centered) |
| --color          | str      | yellow                                 | parcel color, or `speed` to color by the AMV speed (storm-relative, or relative to the `--corotate` frame)          |
| --clim           | float x2 | 0 and the 98th percentile              | speed range of `--color speed` (m/s)                                                                                |
| --linewidth      | float    | 2.2                                    | trail width at the parcel (pt; 0.75x in the main panel when zoom panels are added)                                  |
| --tail_width     | float    | 0.1                                    | trail width at the start of the trajectory, as a fraction of `--linewidth`                                          |
| --log            | bool     |                                        | show the log of the image (visible reflectance)                                                                     |
| --contrast       | float    | 0.0                                    | fraction of the local mean removed from the image; 0.5 brings out the cloud texture                                 |
| --contrast_fwhm  | float    | 8.0                                    | width of that local mean (km)                                                                                       |
| --vlim           | float x2 | -2.4 2.4                               | gray-scale limits of the standardized image                                                                         |
| --rings          | float ...|                                        | radii of range rings about the center (km)                                                                          |
| --rest           | int      | 1                                      | images shown alone after each cycle                                                                                 |
| --fps            | float    | 2.0                                    | movie frames per second                                                                                             |
| --dpi            | float    | 110                                    | movie resolution                                                                                                    |
| --max_frames     | int      |                                        | only the first N movie frames (quick test)                                                                          |
| --title          | str      | the image's `name` attribute + variable | frame-label prefix                                                                                                 |

#### Using `trajectory_movie.py` in your own code
`opentcamv/trajectory_movie.py` imports nothing from openTCAMV: copy that one file next to your scripts and it works with any tracker's output, given numpy and matplotlib (scipy only for its `local_contrast` helper).
```python
from trajectory_movie import animate_trajectories, match_frames   # with openTCAMV installed: from opentcamv.trajectory_movie import ...

# images: (nt, ny, nx) on pixel centers x (nx,), y (ny,) in km, taken at times (nt,) (datetime64)
# xloc, yloc: (nset, nstep, ...) parcel positions in km -- one set per AMV time, steps in time order, NaN where not tracked
# step_times: (nset, nstep) time of every step
frame_index = match_frames(step_times, times)          # image index of every step
movie = animate_trajectories(images, x, y, xloc, yloc, frame_index, times=times, center=(0, 0), rings=[10, 20],
                             panels=[dict(stride=2), dict(center=(0, 0), half=15, title="Eye")])
movie.save("trajectories.mp4")                         # or .gif, which needs no ffmpeg
```
For an openTCAMV output this reads (what `31_` does, without its options):
```python
import xarray as xr
A = xr.open_dataset("2017_Lan_ns7_nt1.nc")             # 20_ output, tracked with --traj_int=1
I = xr.open_dataset("2017_Lan_aeqd_sample.nc")         # the file 10_ tracked
movie = animate_trajectories(I.B03, I.x.values, I.y.values, A.xloc.values, A.yloc.values,
                             match_frames(A.time2.values, I.t.values), times=I.t.values,
                             valid=A.vx.notnull().values, center=(0, 0), panels=[dict(stride=2)])
```
`images` is read one frame at a time, so a lazily opened `xarray.DataArray` never has to fit in memory. `movie.draw(i)` renders movie frame `i` (`movie.fig.savefig(...)` then saves it), and `movie.animation()` returns a matplotlib `FuncAnimation` (in Jupyter: `HTML(movie.animation().to_jshtml())`). The docstring of `animate_trajectories` lists every option: panels, thinning, coloring by speed or any value, the co-rotating frame, and the image display.
</details>
<hr>