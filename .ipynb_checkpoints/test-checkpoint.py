# Bias correct the 1961-2013 HadGEM data, save the corrected data for plotting, and the store the per grid mapping
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import os

# ---------------- user settings ----------------
obs_file = "/home/users/weijia_6/Yangtze_code/tm_CN05_onHadGEMgrid_xesmf_1.nc"
mod_file = "/home/users/weijia_6/Yangtze_code/HADGEM_1961_2013_his_358.nc"
var_obs = "tm"
var_mod = "tas"

out_bc_file = "/home/users/weijia_6/Yangtze_code/HADGEM_1961_2013_his_qm_pergrid.nc"
out_map_file = "/home/users/weijia_6/Yangtze_code/qm_mapping_pergrid.nc"

nbins = 200   # number of bins for each grid cell mapping

# ---------------- load datasets lazily ----------------
obs_ds = xr.open_dataset(obs_file)
mod_ds = xr.open_dataset(mod_file)

da_obs = obs_ds[var_obs]
da_mod = mod_ds[var_mod]

lat = da_obs.lat
lon = da_obs.lon

# ---------------- create output datasets ----------------
# bias-corrected model
mod_bc = xr.Dataset(
    {var_mod: (da_mod.dims, np.full(da_mod.shape, np.nan, dtype=np.float32))},
    coords=mod_ds.coords
)

# mapping storage: (lat, lon, nbins)
qm_bin_centres = xr.DataArray(
    np.full((len(lat), len(lon), nbins), np.nan, dtype=np.float32),
    dims=("lat", "lon", "bins"),
    coords={"lat": lat, "lon": lon, "bins": np.arange(nbins)}
)

qm_model_to_obs = xr.DataArray(
    np.full((len(lat), len(lon), nbins), np.nan, dtype=np.float32),
    dims=("lat", "lon", "bins"),
    coords={"lat": lat, "lon": lon, "bins": np.arange(nbins)}
)

# ---------------- loop over grid cells ----------------
for i in range(len(lat)):
    for j in range(len(lon)):

        # extract obs + model time series at this grid cell
        obs_ts = da_obs[:, i, j].values
        mod_ts = da_mod[:, :, i, j].values if "number" in da_mod.dims else da_mod[:, i, j].values

        # skip if obs is all NaN (outside China)
        if np.all(np.isnan(obs_ts)):
            continue

        # drop NaNs
        obs_valid = obs_ts[np.isfinite(obs_ts)]
        mod_valid = mod_ts[np.isfinite(mod_ts)]

        if len(obs_valid) < 50:
            continue

        # ---------------- build QM mapping for this grid cell ----------------
        vmin = min(obs_valid.min(), mod_valid.min())
        vmax = max(obs_valid.max(), mod_valid.max())

        bins = np.linspace(vmin, vmax, nbins + 1)
        centres = 0.5 * (bins[:-1] + bins[1:])

        # obs CDF
        obs_counts, _ = np.histogram(obs_valid, bins=bins)
        obs_pdf = obs_counts / obs_counts.sum()
        obs_cdf = np.cumsum(obs_pdf)
        obs_cdf = np.maximum.accumulate(obs_cdf)
        obs_cdf /= obs_cdf[-1]

        # model CDF
        mod_counts, _ = np.histogram(mod_valid, bins=bins)
        mod_pdf = mod_counts / mod_counts.sum()
        mod_cdf = np.cumsum(mod_pdf)
        mod_cdf = np.maximum.accumulate(mod_cdf)
        mod_cdf /= mod_cdf[-1]

        # build interpolators
        mod_to_p = interp1d(
            centres, mod_cdf,
            bounds_error=False, fill_value=(0, 1), assume_sorted=True
        )
        p_to_obs = interp1d(
            obs_cdf, centres,
            bounds_error=False,
            fill_value=(centres[0], centres[-1]), assume_sorted=True
        )

        # ---------------- apply QM to full model time series ----------------
        mod_flat = mod_ts.ravel()
        p = mod_to_p(mod_flat)
        mapped = p_to_obs(p)
        mapped = mapped.reshape(mod_ts.shape)

        # ---------------- write corrected values ----------------
        mod_bc[var_mod][:, :, i, j] = mapped

        # ---------------- store mapping for this grid cell ----------------
        qm_bin_centres[i, j, :] = centres
        qm_model_to_obs[i, j, :] = p_to_obs(mod_to_p(centres))

    print(f"Completed latitude {i+1}/{len(lat)}")

# ---------------- save outputs ----------------
mod_bc.to_netcdf(out_bc_file)
print("Saved bias-corrected HadGEM:", out_bc_file)

qm_map_ds = xr.Dataset({
    "qm_bin_centres": qm_bin_centres,
    "qm_model_to_obs": qm_model_to_obs
})
qm_map_ds.to_netcdf(out_map_file)
print("Saved per-grid-cell QM mapping:", out_map_file)

