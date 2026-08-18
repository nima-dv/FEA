r"""
Form the TT-T image from channel data, using the research team's OWN beamformer.

Shared by the round-trip test and the FEM-vs-k-Wave comparison, so that BOTH datasets go
through byte-identical image formation. That is what makes an image difference
attributable to the forward solver rather than to imaging choices.

TT-T = transmit takes a HALF-SKIP off the OD, receive is DIRECT; both legs are shear,
mode-converted at the ID. This is the mode their published `imgz_TT_T` figures use.

Pipeline, exactly as their beamformingscript does it:
    decimate to ~2x the max frequency (factor 23 -> 16.52 MHz)
    sparse receive aperture: every 2nd element (128 of 256)
    bandpass 0.6-1.4 f0, with dt in MICROSECONDS and cutoffs in MHz
    Hilbert -> complex64, so the delay-and-sum is coherent-complex
    transmit TOF : polar_wave_ray_pipe_two_walls_reflect (Snell ray shooting)
    receive TOF  : omnidirectional_ray_pipe_one_wall
    Kirchhoff migration with the 60/80 deg angle filter that TT-T uses (TT and LL do not)

ONE DECLARED SUBSTITUTION: their receive TOF (`omnidirectional_fermat_pipe_one_wall`)
raises NotImplementedError for any engine other than 'cuda'. We use the ray-shooting
sibling from the same package so no GPU is required. Because BOTH datasets go through this
same substitution, it cancels in the comparison. Measured effect on absolute agreement
with their archived image: correlation 0.929, exact peak pixel, 2.4% amplitude.
Do NOT raise `nrays` to "improve" it - higher is worse (see validation/bf_roundtrip.py).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

# The frozen scenario (SI). Used for our own FEM output, where there is no workspace file
# to read geometry from. Matches mesh/ili_mesh.py and repro/ili_forward.py.
FROZEN = dict(
    f0=4.0e6, n_elem=256, pitch=0.30e-3, kerf=0.05e-3,
    c_f=1500.0, c_P=5700.0, c_S=3100.0,
    r_id=0.193675, r_od=0.203200,
    x_c=0.03825, z_c=-0.173675,          # pipe centre; r_id + z_c = 20 mm standoff
    # Notch ground truth. Lives here so metrics and figure captions read it rather than
    # assuming it - a different notch (or a different angle) then needs no code edits.
    notch_x=0.03825, notch_depth=0.004,
)


def params_from_kwave(d) -> dict:
    """Build the parameter dict from a tools/extract_kwave_case.py .npz (their run)."""
    tref = np.asarray(d["transducer_ref_zx"]).ravel()      # stored (z, x) - REVERSED
    ppos = np.asarray(d["pipe_pos_m_zx"]).ravel()
    n_elem = int(d["element_count"])
    pitch = float(d["pitch"])
    x_el = np.arange(n_elem) * pitch
    return dict(
        f0=float(d["frequency"]), n_elem=n_elem, pitch=pitch, kerf=float(d["kerf"]),
        c_f=float(d["c_fluid"]), c_P=float(d["c_L_steel"]), c_S=float(d["c_S_steel"]),
        r_id=float(d["ID"]) / 2, r_od=float(d["OD"]) / 2,
        # their convention: pipe centre relative to the transducer reference, shifted to
        # the aperture midpoint
        x_c=float(ppos[1] - tref[1]) + 0.5 * (x_el[0] + x_el[-1]),
        z_c=float(ppos[0] - tref[0]),
        # Notch truth read from THEIR run rather than assumed, so a different crack height or
        # offset in a future case is picked up automatically. crack_offset is measured from
        # the pipe axis, which is x_c.
        notch_x=(float(ppos[1] - tref[1]) + 0.5 * (x_el[0] + x_el[-1])
                 + float(np.asarray(d["crack_offset"]).ravel()[0])
                 if "crack_offset" in d else FROZEN["notch_x"]),
        notch_depth=(float(np.asarray(d["crack_height"]).ravel()[0])
                     if "crack_height" in d else FROZEN["notch_depth"]),
    )


def tt_t_image(bf, channel_data: np.ndarray, dt: float, angle_deg: float,
               params: dict | None = None, verbose: bool = False):
    """channel_data (n_t, n_elem) time-major -> (|image|, x_ax_mm, z_ax_mm)."""
    p = dict(FROZEN if params is None else params)
    f0 = p["f0"]
    n_elem = p["n_elem"]

    # --- decimate + sparse receive + bandpass + analytic signal ------------------------
    decimate = max(1, int(np.floor((1 / (2 * 2 * f0)) / dt)))
    dt_d = decimate * dt
    CH = channel_data[::decimate, :n_elem].T.copy()[np.newaxis, ...]
    dt_us = dt_d * 1e6
    sparse = np.arange(0, n_elem, 2)
    CH = CH[:, sparse, :]
    f0_mhz = f0 * 1e-6
    CH = bf.utils.bandpass(CH, 0.6 * f0_mhz, 1.4 * f0_mhz, dt=dt_us, axis=-1)
    inp = hilbert(CH).astype(np.complex64)

    # --- geometry ---------------------------------------------------------------------
    x_el_m = np.arange(n_elem) * p["pitch"]
    x_rx_m = x_el_m[sparse]
    x_rx_mm = x_rx_m * 1e3
    receiver_pos = np.vstack((x_rx_m, np.zeros_like(x_rx_m)))
    r_id_mm, r_od_mm = p["r_id"] * 1e3, p["r_od"] * 1e3
    x_cm, z_cm = p["x_c"] * 1e3, p["z_c"] * 1e3
    standoff_mm = r_id_mm + z_cm
    thickness_mm = r_od_mm - r_id_mm
    if verbose:
        print(f"    decimate {decimate}, fs {1/dt_d/1e6:.2f} MHz, rx {sparse.size}, "
              f"standoff {standoff_mm:.3f} mm")

    # --- imaging grid: their wavelength-driven TT grid ---------------------------------
    max_tx_f = f0 + 0.5 * (0.6 * f0)                       # 5.2 MHz
    safety = 1.2
    L = x_rx_mm.max() - x_rx_mm.min()
    x_ax_mm = bf.utils.imaging_grid(1e3 * p["c_S"], max_tx_f,
                                    x_rx_mm[0] - 0.1 * L, x_rx_mm[-1] + 0.1 * L,
                                    resolution=0.5 / safety)
    z_ax_mm = bf.utils.imaging_grid(1e3 * p["c_S"], max_tx_f, 0.0,
                                    1.5 * (standoff_mm + thickness_mm),
                                    resolution=0.25 / safety)
    x_ax_m, z_ax_m = 1e-3 * x_ax_mm, 1e-3 * z_ax_mm
    fnum = bf.utils.optimal_fnumber(velocity=p["c_f"], frequency=max_tx_f,
                                    transducer_width=p["pitch"] - p["kerf"], dB_drop=3)
    ang = np.deg2rad(np.array([angle_deg]))

    # --- travel times -----------------------------------------------------------------
    trav_src, angs_src = bf.tof.polar_wave_ray_pipe_two_walls_reflect(
        p["c_f"], p["c_S"], p["c_S"], x_ax_m, z_ax_m, ang,
        p["x_c"], p["z_c"], p["r_id"], p["r_od"],
        critical_thresh=0.8, interp_step=20, positions=receiver_pos,
        model_edges=False, output_rays=False, output_rayangles=True, verbose=False)
    trav_rec, angs_rec = bf.tof.omnidirectional_ray_pipe_one_wall(
        p["c_f"], p["c_S"], x_ax_m, z_ax_m, receiver_pos,
        p["x_c"], p["z_c"], p["r_id"],
        fnumber=fnum, critical_thresh=0.98,
        output_rays=False, output_rayangles=True, verbose=False)

    angle_filter, _ = bf.utils.angle_filter_migration(361, angle_pass=60, angle_cut=80)
    img = bf.mig.kirchhoff_from_tof(
        inp, np.asarray(trav_src) * 1e6, np.asarray(trav_rec) * 1e6, dt=dt_us,
        angles_src=np.asarray(angs_src), angles_rec=np.asarray(angs_rec),
        angle_filter=angle_filter, engine="numpy")
    return np.abs(np.squeeze(np.asarray(img))), x_ax_mm, z_ax_mm


def image_metrics(img, x_ax_mm, z_ax_mm, params: dict | None = None,
                  notch_x_mm: float | None = None, notch_depth_mm: float | None = None,
                  guard_mm: float = 6.0):
    """Crack response vs defect-free-wall clutter - the quantitative comparison.

    A defect-free region of homogeneous steel must return NOTHING. Whatever it does return
    is numerical, so crack-peak / wall-clutter is the metric that separates a solver whose
    geometry is exact from one that rasterises it.

    Wall pixels are selected by RADIUS from the pipe centre (the wall is curved, so a
    rectangular band would leak fluid and outside-the-OD pixels into the statistics).
    """
    p = dict(FROZEN if params is None else params)
    # Notch truth comes from the scenario unless the caller overrides it (the robustness
    # sweeps do). Falling back to FROZEN keeps a params dict that predates these keys working.
    if notch_x_mm is None:
        notch_x_mm = p.get("notch_x", FROZEN["notch_x"]) * 1e3
    if notch_depth_mm is None:
        notch_depth_mm = p.get("notch_depth", FROZEN["notch_depth"]) * 1e3
    X, Z = np.meshgrid(x_ax_mm, z_ax_mm, indexing="ij")
    R = np.hypot(X - p["x_c"] * 1e3, Z - p["z_c"] * 1e3)
    r_id_mm, r_od_mm = p["r_id"] * 1e3, p["r_od"] * 1e3

    in_wall = (R >= r_id_mm) & (R <= r_od_mm)
    near_notch = np.abs(X - notch_x_mm) <= 1.5          # +-1.5 mm around the slot
    deep_enough = R >= (r_od_mm - notch_depth_mm - 1.0)  # the notch spans this radial band
    crack_roi = in_wall & near_notch & deep_enough
    clutter_roi = in_wall & (np.abs(X - notch_x_mm) > guard_mm)

    a = np.nan_to_num(img)
    pk = a[crack_roi].max() if crack_roi.any() else np.nan
    i = np.unravel_index(np.where(crack_roi, a, -np.inf).argmax(), a.shape)
    cl = a[clutter_roi]
    rms = float(np.sqrt((cl ** 2).mean()))
    p95 = float(np.percentile(cl, 95))
    mx = float(cl.max())
    db = lambda x: 20 * np.log10(pk / x) if x > 0 else np.inf

    # notch vertical extent: contiguous run above 25% of peak in the notch column
    col = a[np.abs(x_ax_mm - notch_x_mm) <= 0.4, :].max(axis=0)
    hit = np.where(col > 0.25 * pk)[0]
    extent = (z_ax_mm[hit.max()] - z_ax_mm[hit.min()]) if hit.size else np.nan

    return dict(crack_peak=float(pk),
                crack_x_mm=float(x_ax_mm[i[0]]), crack_z_mm=float(z_ax_mm[i[1]]),
                clutter_rms=rms, clutter_p95=p95, clutter_max=mx,
                cnr_rms_db=float(db(rms)), cnr_p95_db=float(db(p95)),
                cnr_worst_db=float(db(mx)), notch_extent_mm=float(extent),
                n_wall_px=int(in_wall.sum()), n_clutter_px=int(clutter_roi.sum()))
