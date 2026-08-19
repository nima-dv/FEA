r"""
Form the TT-T image from channel data, using the research team's OWN beamformer.

Shared by the round-trip test and the FEM-vs-k-Wave comparison, so that BOTH datasets go
through byte-identical image formation. That is what makes an image difference
attributable to the forward solver rather than to imaging choices.

TT-T = transmit takes a HALF-SKIP off the OD, receive is DIRECT; both legs are shear,
mode-converted at the ID. This is the mode their published `imgz_TT_T` figures use.

Pipeline (the `legacy` chain, which every published figure uses). NOTE: the claim that
this is "exactly as their beamformingscript does it" was checked on 2026-08-19 and is
WRONG on three counts - see CHAINS below, and the `faithfulbf` preset that closes them:
    decimate to ~2x the max frequency (factor 23 -> 16.52 MHz; THEIRS is 3x -> 15)
    sparse receive aperture: every 2nd element (128 of 256)
    bandpass 0.6-1.4 f0, with dt in MICROSECONDS and cutoffs in MHz (THEIRS has none)
    Hilbert -> complex64, so the delay-and-sum is coherent-complex
    transmit TOF : polar_wave_ray_pipe_two_walls_reflect (Snell ray shooting)
    receive TOF  : omnidirectional_ray_pipe_one_wall
    Kirchhoff migration with the 60/80 deg angle filter that TT-T uses (TT and LL do not),
        and NO operator antialias, where THEIRS passes antialias=0.5

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


# --- imaging-chain presets ------------------------------------------------------------
# `legacy` IS THE DEFAULT AND MUST STAY BIT-IDENTICAL. Every published figure and number
# on disk was made with it, so it is frozen: stride decimation from 2*f0 (factor 23),
# engine="numpy", no operator antialias, bandpass on.
#
# `faithfulbf` is what the research team's OWN beamforming_script_simulation.py passes,
# which this module had drifted from on three counts (measured 2026-08-19):
#   dec_from  their line 197-199 sets max_freq = 3*f0, so their factor is 15, not our 23.
#   antialias their line 698/709 (the imgz_TT_T calls) pass antialias=0.5. We passed
#             nothing. antialias is the Lumley-Claerbout-Bevc migration-OPERATOR
#             anti-alias, and mig/_kirchhoff.py:290-291 raises NotImplementedError for
#             engine="numpy", so the engine has to move with it.
#   bandpass  their simulation script has NO bandpass. Ours came from their
#             experimental-data script (beamforming_script_experimental.py:233).
# `nobandpass` = faithfulbf with that last difference closed too. Split out because
# removing a bandpass can move sizing, so it needs its own threshold sweep.
CHAINS = dict(
    legacy=dict(dec_from=2.0, engine="numpy", antialias=0.0, bandpass=True),
    faithfulbf=dict(dec_from=3.0, engine="numba", antialias=0.5, bandpass=True),
    nobandpass=dict(dec_from=3.0, engine="numba", antialias=0.5, bandpass=False),
    # null check for the engine swap: numba must equal numpy when antialias is off.
    numbanull=dict(dec_from=2.0, engine="numba", antialias=0.0, bandpass=True),
)


def tt_t_image(bf, channel_data: np.ndarray, dt: float, angle_deg: float,
               params: dict | None = None, verbose: bool = False,
               chain: str | dict = "legacy"):
    """channel_data (n_t, n_elem) time-major -> (|image|, x_ax_mm, z_ax_mm).

    `chain` selects an entry of CHAINS (or is a dict of the same shape). It defaults to
    "legacy", which is the published chain, unchanged.
    """
    p = dict(FROZEN if params is None else params)
    f0 = p["f0"]
    n_elem = p["n_elem"]
    c = dict(CHAINS[chain] if isinstance(chain, str) else chain)

    # --- decimate + sparse receive + bandpass + analytic signal ------------------------
    decimate = max(1, int(np.floor((1 / (2 * c["dec_from"] * f0)) / dt)))
    dt_d = decimate * dt
    CH = channel_data[::decimate, :n_elem].T.copy()[np.newaxis, ...]
    dt_us = dt_d * 1e6
    sparse = np.arange(0, n_elem, 2)
    CH = CH[:, sparse, :]
    f0_mhz = f0 * 1e-6
    if c["bandpass"]:
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
        print(f"    chain {chain if isinstance(chain, str) else 'custom'}: decimate "
              f"{decimate}, fs {1/dt_d/1e6:.2f} MHz, rx {sparse.size}, "
              f"bandpass {'on' if c['bandpass'] else 'OFF'}, engine {c['engine']}, "
              f"antialias {c['antialias']}, standoff {standoff_mm:.3f} mm")

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
    # antialias is only passed when non-zero: `legacy` must reach their function with the
    # exact same argument list it always did.
    extra = {"antialias": c["antialias"]} if c["antialias"] else {}
    img = bf.mig.kirchhoff_from_tof(
        inp, np.asarray(trav_src) * 1e6, np.asarray(trav_rec) * 1e6, dt=dt_us,
        angles_src=np.asarray(angs_src), angles_rec=np.asarray(angs_rec),
        angle_filter=angle_filter, engine=c["engine"], **extra)
    return np.abs(np.squeeze(np.asarray(img))), x_ax_mm, z_ax_mm


EDGE_X_BAND = (70.0, 85.0)      # the right-edge band of the imaged wall
EDGE_Z_BAND = (25.0, 30.0)      # the OUTER half of the wall depth, where the notch lives


def edge_clutter(img, x_ax_mm, z_ax_mm, crack_peak: float,
                 params: dict | None = None, angle_deg: float = 20.0,
                 x_band=EDGE_X_BAND, z_band=EDGE_Z_BAND):
    """The edge-clutter level, in TWO PINNED definitions. Both are dB re the crack peak.

    WHY THIS FUNCTION EXISTS
        The "+1.8 dB edge-clutter excess" was chased through three experiments while
        living in no script at all. Re-derived from the saved images on 2026-08-19 it
        ranges from +3.11 dB to -0.76 dB depending on choices nobody had written down.
        Both definitions below are now computed here and nowhere else, so the number
        cannot drift again.

    THE PINNED DEFINITION (`edge_p95_db`)
        in-wall pixels with x in `x_band` AND z in `z_band`, p95, dB re that image's own
        crack peak. This is the one that reproduces the value the report quotes: on the
        committed images_20.npz it gives FEM -13.75, k-Wave -15.50, excess +1.76 dB.

    THE VARIANT (`edge_rms_db`)
        in-wall pixels with x in `x_band`, NO z restriction, RMS, same normalisation.
        FEM -21.40, k-Wave -20.65, excess -0.76 dB, i.e. FEM is BETTER under this one.
        It is reported alongside on purpose: our published claim IS sensitive to the
        choice and the variant that flatters us must not be the only one on show.

    THE FULL SET MEASURED, FEM-minus-k-Wave excess on the committed +20 deg images:
        x 70-85, z 25-30, p95   +1.76 dB   <- PINNED, matches the reported +1.8
        x 70-80, z 25-30, p95   +1.76 dB
        x 70-85, z 25-30, RMS   +3.11 dB
        x 70-80, z 25-30, RMS   +3.13 dB
        x 70-85, all z,   p95   -0.25 dB
        x 70-85, all z,   RMS   -0.76 dB   <- VARIANT
        x 70-80, all z,   RMS   -0.85 dB
        x 65-85, all z,   RMS   -1.17 dB
        per-column dB-domain mean over x 70-85, re crack peak       -2.06 dB
        same, normalised to each image's own whole-wall clutter RMS  -0.83 dB
    Normalising to each image's own whole-wall clutter RMS instead of its crack peak
    shifts every row by roughly +1.2 dB. NONE of this is derivable from first
    principles: the definition is a CHOICE, and `edge_p95_db` is the choice the report
    quotes. Positive excess means FEM is dirtier than k-Wave.

    THE BAND MIRRORS WITH THE STEERING SIGN, and it must. x 70-85 mm is the DOWN-STEER
    edge at +20 deg only. Measured 2026-08-19 on images_-20*.npz: at -20 deg that band is
    identically ZERO in both solvers' images (band RMS exactly 0, so the level is -inf) -
    the TOF tables are NaN there because the beam is steered the other way and nothing is
    insonified. Reading "x 70-85" literally at -20 deg therefore does not measure a
    smaller artefact, it measures empty grid. For `angle_deg` < 0 the band is reflected
    about the aperture midpoint, which is where the artefact actually appears.
    """
    p = dict(FROZEN if params is None else params)
    if angle_deg < 0:                       # reflect about the aperture midpoint
        x_mid = 0.5 * (p["n_elem"] - 1) * p["pitch"] * 1e3
        x_band = (2 * x_mid - x_band[1], 2 * x_mid - x_band[0])
    X, Z = np.meshgrid(x_ax_mm, z_ax_mm, indexing="ij")
    R = np.hypot(X - p["x_c"] * 1e3, Z - p["z_c"] * 1e3)
    in_wall = (R >= p["r_id"] * 1e3) & (R <= p["r_od"] * 1e3)
    a = np.nan_to_num(img)
    band = in_wall & (X >= x_band[0]) & (X <= x_band[1])
    core = band & (Z >= z_band[0]) & (Z <= z_band[1])
    db = lambda v: float(20 * np.log10(v / crack_peak)) if v > 0 else -np.inf
    return dict(edge_p95_db=db(float(np.percentile(a[core], 95))) if core.any() else np.nan,
                edge_rms_db=db(float(np.sqrt((a[band] ** 2).mean()))) if band.any() else np.nan,
                edge_x_band=(float(x_band[0]), float(x_band[1])),
                edge_core_px=int(core.sum()), edge_band_px=int(band.sum()))


def image_metrics(img, x_ax_mm, z_ax_mm, params: dict | None = None,
                  notch_x_mm: float | None = None, notch_depth_mm: float | None = None,
                  guard_mm: float = 6.0, angle_deg: float = 20.0):
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

    out = dict(crack_peak=float(pk),
               crack_x_mm=float(x_ax_mm[i[0]]), crack_z_mm=float(z_ax_mm[i[1]]),
               clutter_rms=rms, clutter_p95=p95, clutter_max=mx,
               cnr_rms_db=float(db(rms)), cnr_p95_db=float(db(p95)),
               cnr_worst_db=float(db(mx)), notch_extent_mm=float(extent),
               n_wall_px=int(in_wall.sum()), n_clutter_px=int(clutter_roi.sum()))
    # Pinned edge-clutter numbers travel with every metrics dict, so no caller can quote
    # an unpinned one. See edge_clutter's docstring for why both are reported.
    out.update(edge_clutter(img, x_ax_mm, z_ax_mm, float(pk), p, angle_deg))
    return out


if __name__ == "__main__":
    # Self-check. Needs only numpy and the committed +20 deg images, no beamformer.
    #   ./run.ps1 python3 lib/tt_t_image.py
    # Guards the two things that must not drift: the legacy chain's settings, and the
    # pinned edge-clutter number on the published image.
    from pathlib import Path

    assert CHAINS["legacy"] == dict(dec_from=2.0, engine="numpy", antialias=0.0,
                                    bandpass=True), "legacy chain has been altered"
    p = Path(__file__).resolve().parents[1] / "results" / "compare" / "images_20.npz"
    d = np.load(p)
    m = {lab: image_metrics(np.nan_to_num(d[f"{lab}_img"]), d["x"], d["z"])
         for lab in ("FEM", "k-Wave")}
    pinned = m["FEM"]["edge_p95_db"] - m["k-Wave"]["edge_p95_db"]
    variant = m["FEM"]["edge_rms_db"] - m["k-Wave"]["edge_rms_db"]
    print(f"pinned edge excess  {pinned:+.2f} dB (expect +1.76)")
    print(f"variant edge excess {variant:+.2f} dB (expect -0.76)")
    assert abs(pinned - 1.76) < 0.01, pinned
    assert abs(variant + 0.76) < 0.01, variant
    assert abs(m["FEM"]["notch_extent_mm"] - 3.73) < 0.01, m["FEM"]["notch_extent_mm"]
    assert abs(m["k-Wave"]["notch_extent_mm"] - 3.23) < 0.01
    print("OK: legacy chain frozen, pinned edge metric reproduces the published values")
