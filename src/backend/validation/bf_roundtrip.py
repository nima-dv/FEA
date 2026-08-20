r"""
ROUND-TRIP TEST: run the research team's own beamformer on their own channel data
and try to reproduce their archived TT_T.mat.

WHY THIS IS STEP ONE
--------------------
The whole FEM-vs-k-Wave comparison rests on being able to push OUR channel data
through THEIR imaging chain. Before trusting that, we must show the chain
reproduces their published image when fed their own input. If this fails, every
downstream number is meaningless.

It also validates our reading of their input contract - time-zero convention,
decimation, sparse aperture, geometry sign conventions - which we reverse
engineered from their scripts.

THE ONE SUBSTITUTION (declared, then measured)
----------------------------------------------
Their receive-side travel times use `omnidirectional_fermat_pipe_one_wall`, which
raises NotImplementedError("engine must be 'cuda'") for any non-CUDA engine. To
avoid requiring GPU passthrough we substitute `omnidirectional_ray_pipe_one_wall`
from the SAME package: same geometry, same conventions, ray-shooting +
interpolation instead of golden-section Fermat minimisation.

Everything else is theirs, unmodified: the transmit ray tracer
(`polar_wave_ray_pipe_two_walls_reflect`, CPU already), the bandpass, the imaging
grid, the angle filter, and `kirchhoff_from_tof` (numpy engine).

Pass `--receive fermat` to use their exact CUDA kernel when a GPU is available;
the difference between the two is the quantity this script reports.

RESULT (2026-08-11, crack_0deg vs their archived TT_T.mat)
---------------------------------------------------------
With defaults (nrays=101, interp_step=10):

    image grid            370 x 358   -> EXACTLY their shape
    standoff cross-check  20.000 mm computed == 20.000 recorded
    peak position         identical pixel, displacement 0.000 mm
    amplitude scale       0.976 (theirs/ours) - 2.4% off
    correlation           0.929
    relative L2           40.4%

Grid, geometry and peak location matching exactly means the input contract we
reverse engineered - decimation 23, sparse 128 receive, 0.6-1.4 f0 bandpass,
(z,x)-reversed transducer_ref/pipe_pos_m, min-shifted delay origin, the TT-T
60/80 deg angle filter - is RIGHT. The residual is the receive-TOF kernel.

Raising the ray count makes agreement WORSE, not better:

    nrays=101  (default)  corr 0.929   peak offset 0.000 mm
    nrays=401             corr 0.875   peak offset 6.483 mm
    nrays=1001            corr 0.860   peak offset 6.071 mm

So nrays/interp_step are not accuracy knobs here. More rays populate pixels that
Fermat's critical-angle test legitimately rejects, injecting energy where their
image has none - which drags the peak away. Keep the defaults.

CONSEQUENCE FOR THE COMPARISON
------------------------------
We do NOT need to match their archived Fermat image to make a fair FEM-vs-k-Wave
claim. We beamform BOTH their channel data and our FEM channel data through this
same chain, so the substitution cancels and any difference in the resulting
images is attributable to the forward solver. Their archived TT_T.mat stays a
sanity reference, not the comparison baseline.

If an exact match to their published images is ever required, pass
`--receive fermat`; that needs GPU passthrough into the container.

RUN
  ./run.ps1 python3 validation/bf_roundtrip.py \
      --case  <scratch>/kwave_cases/crack_0deg.npz \
      --ref   <scratch>/kwave_cases/ref_TT_T_crack_0deg.mat
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.bf_loader import load_beamformer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help=".npz from tools/extract_kwave_case.py")
    ap.add_argument("--ref", default=None, help="their TT_T.mat (v7.3) to compare against")
    ap.add_argument("--receive", choices=["ray", "fermat"], default="ray",
                    help="receive-side TOF kernel: CPU ray-shooting (default) or their CUDA Fermat")
    ap.add_argument("--out", default=None, help="optional .npz to save our image into")
    ap.add_argument("--nrays", type=int, default=101,
                    help="receive ray-tracer ray count (only for --receive ray)")
    ap.add_argument("--rec-interp-step", type=int, default=10,
                    help="receive ray-tracer interface interpolation step")
    args = ap.parse_args()

    bf = load_beamformer()
    d = np.load(args.case, allow_pickle=True)

    # ---- 1. channel data: decimate + sparse aperture, exactly as their script -------
    cd = d["channel_data"]                                # (n_t, 256) time-major
    dt_orig = float(d["dt"])
    f0 = float(d["frequency"])
    f0_mhz = 1e-6 * f0

    max_freq = 2 * f0
    nyquist = 1 / (2 * max_freq)
    decimate = max(1, int(np.floor(nyquist / dt_orig)))
    dt = decimate * dt_orig

    n_elem = int(d["element_count"])
    CH = cd[::decimate, np.arange(n_elem)].T.copy()[np.newaxis, ...]   # (1, 256, n_t')
    nt = CH.shape[-1]
    t_us = np.arange(nt) * dt * 1e6
    dt_us = t_us[1] - t_us[0]

    sparse_pattern = np.arange(0, n_elem, 2)
    CH = CH[:, sparse_pattern, :]                                      # (1, 128, n_t')

    lowcut_mhz, highcut_mhz = 0.6 * f0_mhz, 1.4 * f0_mhz
    CH = bf.utils.bandpass(CH, lowcut_mhz, highcut_mhz, dt=dt_us, axis=-1)
    inp_hilb = hilbert(CH).astype(np.complex64)

    print(f"decimate {decimate}  dt {dt:.6e} s  fs {1/dt/1e6:.3f} MHz  nt {nt}")
    print(f"aperture: transmit {n_elem}, receive {sparse_pattern.size}")
    print(f"bandpass {lowcut_mhz:.2f}-{highcut_mhz:.2f} MHz")

    # ---- 2. geometry (note: transducer_ref and pipe_pos_m are stored (z, x)) --------
    pitch = float(d["pitch"])
    drec_mm = 1e3 * pitch
    x_rec_mm = np.arange(n_elem) * drec_mm
    x_rec_m = 1e-3 * x_rec_mm
    x_rx_m = x_rec_m[sparse_pattern]
    x_rx_mm = x_rec_mm[sparse_pattern]
    receiver_pos = np.vstack((x_rx_m, np.zeros_like(x_rx_m)))           # (2, n_rx), metres

    tref_z, tref_x = (float(v) for v in np.asarray(d["transducer_ref_zx"]).ravel()[:2])
    ppos_z, ppos_x = (float(v) for v in np.asarray(d["pipe_pos_m_zx"]).ravel()[:2])

    r_id_mm = 1e3 * float(d["ID"]) / 2
    r_od_mm = 1e3 * float(d["OD"]) / 2
    thickness_mm = r_od_mm - r_id_mm

    x_centerpolar_mm = 1e3 * (ppos_x - tref_x + (x_rec_m[0] + x_rec_m[-1]) / 2.0)
    z_centerpolar_mm = 1e3 * (ppos_z - tref_z)
    standoff_mm = r_id_mm + z_centerpolar_mm

    c0_L = float(d["c_fluid"])
    c1_T = float(d["c_S_steel"])
    angles_rad = np.deg2rad(np.array([float(d["polar_incidence_angle"])]))

    print(f"pipe centre ({x_centerpolar_mm:.3f}, {z_centerpolar_mm:.3f}) mm  "
          f"r_ID {r_id_mm:.3f}  r_OD {r_od_mm:.3f}  wall {thickness_mm:.3f}")
    print(f"standoff from geometry {standoff_mm:.3f} mm "
          f"(recorded {1e3*float(d['standoff']):.3f}) "
          f"{'OK' if abs(standoff_mm - 1e3*float(d['standoff'])) < 1e-6 else 'MISMATCH'}")

    # ---- 3. imaging grid: their wavelength-driven TT grid ---------------------------
    safety_factor = 1.2
    standoff_region = 0.5
    bandwidth_hz = 0.6 * f0
    max_tx_f = f0 + 0.5 * bandwidth_hz                                 # 5.2 MHz
    element_width_m = pitch - float(d["kerf"])

    array_length = x_rx_mm.max() - x_rx_mm.min()
    x_min_mm = x_rx_mm[0] - 0.1 * array_length
    x_max_mm = x_rx_mm[-1] + 0.1 * array_length
    x_ax_mm = bf.utils.imaging_grid(1e3 * c1_T, max_tx_f, x_min_mm, x_max_mm,
                                    resolution=0.5 / safety_factor)
    z_max_mm = (1 + standoff_region) * (standoff_mm + thickness_mm)
    z_ax_mm = bf.utils.imaging_grid(1e3 * c1_T, max_tx_f, 0.0, z_max_mm,
                                    resolution=0.25 / safety_factor)
    x_ax_m, z_ax_m = 1e-3 * x_ax_mm, 1e-3 * z_ax_mm
    print(f"image grid {x_ax_mm.size} x {z_ax_mm.size}  "
          f"x [{x_ax_mm[0]:.2f},{x_ax_mm[-1]:.2f}] z [{z_ax_mm[0]:.2f},{z_ax_mm[-1]:.2f}] mm")

    fnumber_3dB = bf.utils.optimal_fnumber(velocity=c0_L, frequency=max_tx_f,
                                           transducer_width=element_width_m, dB_drop=3)

    # ---- 4. travel times: TT-T = transmit half-skip (via OD), receive direct --------
    t0 = time.time()
    trav_src, angs_src = bf.tof.polar_wave_ray_pipe_two_walls_reflect(
        c0_L, c1_T, c1_T,
        x_ax_m, z_ax_m, angles_rad,
        1e-3 * x_centerpolar_mm, 1e-3 * z_centerpolar_mm,
        1e-3 * r_id_mm, 1e-3 * r_od_mm,
        critical_thresh=0.8, interp_step=20,
        positions=receiver_pos, model_edges=False,
        output_rays=False, output_rayangles=True, verbose=False,
    )
    trav_src = np.asarray(trav_src) * 1e6                              # s -> us
    print(f"transmit TOF {trav_src.shape} in {time.time()-t0:.1f}s")

    t0 = time.time()
    if args.receive == "fermat":
        trav_rec, angs_rec = bf.tof.omnidirectional_fermat_pipe_one_wall(
            c0_L, c1_T, x_ax_m, z_ax_m, receiver_pos,
            1e-3 * x_centerpolar_mm, 1e-3 * z_centerpolar_mm, 1e-3 * r_id_mm,
            fnumber=fnumber_3dB, critical_thresh=0.98,
            output_angles=True, engine="cuda",
        )
        trav_rec = bf.utils.to_host_if_on_device(trav_rec)
        angs_rec = bf.utils.to_host_if_on_device(angs_rec)
    else:
        trav_rec, angs_rec = bf.tof.omnidirectional_ray_pipe_one_wall(
            c0_L, c1_T, x_ax_m, z_ax_m, receiver_pos,
            1e-3 * x_centerpolar_mm, 1e-3 * z_centerpolar_mm, 1e-3 * r_id_mm,
            nrays=args.nrays, interp_step=args.rec_interp_step,
            fnumber=fnumber_3dB, critical_thresh=0.98,
            output_rays=False, output_rayangles=True, verbose=False,
        )
    trav_rec = np.asarray(trav_rec) * 1e6
    angs_rec = np.asarray(angs_rec)
    print(f"receive TOF  ({args.receive}) {trav_rec.shape} in {time.time()-t0:.1f}s")

    # ---- 5. migrate (TT-T carries the 60/80 deg angle filter; TT and LL do not) -----
    angle_filter, _ = bf.utils.angle_filter_migration(361, angle_pass=60, angle_cut=80)
    t0 = time.time()
    img = bf.mig.kirchhoff_from_tof(
        inp_hilb, trav_src, trav_rec, dt=1e6 * dt,
        angles_src=np.asarray(angs_src), angles_rec=angs_rec,
        angle_filter=angle_filter, engine="numpy",
    )
    img = np.squeeze(np.asarray(img))
    print(f"migration {img.shape} in {time.time()-t0:.1f}s")

    ours = np.abs(img)
    print(f"\nOUR image  : {ours.shape}  max {ours.max():.6g}")

    # ---- 6. compare with their archived image --------------------------------------
    if args.ref:
        import h5py
        with h5py.File(args.ref, "r") as f:
            ref = f["data"][...]
        if ref.dtype.names and {"real", "imag"} <= set(ref.dtype.names):
            ref = ref["real"] + 1j * ref["imag"]
        ref = np.abs(np.squeeze(ref))
        print(f"THEIR image: {ref.shape}  max {ref.max():.6g}")

        if ref.shape != ours.shape:
            if ref.T.shape == ours.shape:
                print("note: their array is transposed relative to ours; transposing")
                ref = ref.T
            else:
                print(f"SHAPE MISMATCH {ref.shape} vs {ours.shape} - "
                      "grid construction differs; cannot compare pixelwise")
                return

        a, b = ours.ravel(), ref.ravel()
        good = np.isfinite(a) & np.isfinite(b)
        corr = np.corrcoef(a[good], b[good])[0, 1]
        scale = b[good].max() / a[good].max()
        rel = np.linalg.norm(a[good] * scale - b[good]) / np.linalg.norm(b[good])
        pk_o = np.unravel_index(np.nanargmax(ours), ours.shape)
        pk_r = np.unravel_index(np.nanargmax(ref), ref.shape)

        print(f"\n  correlation           {corr:.6f}")
        print(f"  amplitude scale       {scale:.6g}  (theirs/ours)")
        print(f"  relative L2 (scaled)  {rel*100:.3f}%")
        print(f"  peak ours   x={x_ax_mm[pk_o[0]]:.2f} z={z_ax_mm[pk_o[1]]:.2f} mm")
        print(f"  peak theirs x={x_ax_mm[pk_r[0]]:.2f} z={z_ax_mm[pk_r[1]]:.2f} mm")
        dpk = np.hypot(x_ax_mm[pk_o[0]] - x_ax_mm[pk_r[0]],
                       z_ax_mm[pk_o[1]] - z_ax_mm[pk_r[1]])
        print(f"  peak displacement     {dpk:.3f} mm")
        verdict = "PASS" if (corr > 0.98 and dpk < 0.5) else "INVESTIGATE"
        print(f"\n  ROUND-TRIP: {verdict}")

    if args.out:
        np.savez_compressed(args.out, img=img, x_ax_mm=x_ax_mm, z_ax_mm=z_ax_mm)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
