r"""
Extract one k-Wave run into a compact .npz for FEM comparison / beamformer round-trip.

The team's `*_workspace.mat` files are 40-230 MB MAT v7 dumps of an entire MATLAB
workspace. We need ~30 scalars plus the channel data. This pulls out exactly the
fields the beamformer contract requires and writes a few-MB .npz.

READ-ONLY over the source. The Dropbox tree
`F:\DarkVision Dropbox\RnD\Data\4_Simulations\pipe\polarWave\...`
is shared team data: we read it and never write, create, move or delete anything
there. `--out` must therefore be somewhere else (the scratchpad); the script
refuses to write onto the F: drive.

MAT format notes (learned the hard way):
  * `channel_data*.mat` and `*_workspace.mat` are MAT **v7** (zlib) -> scipy.io.
  * `scipy.io.whosmat()` THROWS on the workspace files (top-level MCOS opaque
    objects), so we must request named variables instead of introspecting.
  * The processed `TT_T.mat` / `RFdata.mat` are MAT **v7.3** (HDF5) -> h5py.

USAGE
  python3 tools/extract_kwave_case.py --run "<run folder>" --out <file.npz>
  optional: --tt-t "<processed folder>/TT_T.mat"   (ground-truth beamformed image)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# Fields we need, per the beamformer input contract.
_WANTED = ["sim_output", "TX", "target", "transducer", "simulation", "sim_object", "inc_angle"]


def _leaf(obj, *names, default=None):
    """Walk a chain of attribute names through scipy mat_struct objects."""
    cur = obj
    for n in names:
        if cur is None:
            return default
        cur = getattr(cur, n, None)
    return default if cur is None else cur


def _material(sim_object, name):
    """Pull (c_comp, c_shear, density) for a named entry of the 8-material table."""
    items = np.atleast_1d(sim_object)
    for it in items:
        nm = _leaf(it, "name")
        if isinstance(nm, np.ndarray):
            nm = nm.item() if nm.size == 1 else str(nm)
        if str(nm).strip() == name:
            return (float(_leaf(it, "sound_speed_compression", default=np.nan)),
                    float(_leaf(it, "sound_speed_shear", default=np.nan)),
                    float(_leaf(it, "density", default=np.nan)))
    raise KeyError(f"material {name!r} not found in sim_object table")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder containing *_workspace.mat")
    ap.add_argument("--out", required=True, help="output .npz (must NOT be on the F: drive)")
    ap.add_argument("--tt-t", default=None, help="optional TT_T.mat (v7.3) ground-truth image")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out)

    # Guard the read-only boundary explicitly rather than trusting the caller.
    if out.drive.upper().startswith("F:"):
        raise SystemExit("refusing to write to the F: drive - Dropbox is read-only")

    ws = sorted(run.glob("*_workspace.mat"))
    if not ws:
        raise SystemExit(f"no *_workspace.mat in {run}")
    ch = sorted(run.glob("channel_data*.mat"))
    if not ch:
        raise SystemExit(f"no channel_data*.mat in {run}")

    print(f"workspace : {ws[0].name}  ({ws[0].stat().st_size/1e6:.1f} MB)")
    print(f"channels  : {ch[0].name}  ({ch[0].stat().st_size/1e6:.1f} MB)")

    m = loadmat(str(ws[0]), variable_names=_WANTED,
                struct_as_record=False, squeeze_me=True)
    missing = [k for k in _WANTED if k not in m]
    if missing:
        print(f"note: absent from this workspace (older run?): {missing}")

    sim_output = m.get("sim_output")
    TX = m.get("TX")
    target = m.get("target")
    transducer = m.get("transducer")
    simulation = m.get("simulation")
    sim_object = m.get("sim_object")

    # channel_data from the dedicated (smaller) file; it duplicates sim_output.channel_data
    cd = loadmat(str(ch[0]), variable_names=["channel_data"])["channel_data"]
    if cd.dtype == object:            # the dual-probe cell [2,1] case
        raise SystemExit("channel_data is a cell (dual-transducer run); not supported here")
    cd = np.ascontiguousarray(cd, dtype=np.float64)

    c_f, _, rho_f = _material(sim_object, "fluid")
    c_L, c_S, rho_s = _material(sim_object, "target")

    # transducer_ref and pipe_pos_m are stored (z, x) - REVERSED. Keep raw and split.
    tref = np.atleast_1d(_leaf(sim_output, "transducer", "transducer_ref")).astype(float)
    ppos = np.atleast_1d(_leaf(sim_output, "pipe_pos_m")).astype(float)

    data = dict(
        channel_data=cd,                                        # (n_t, 256) time-major
        dt=float(_leaf(sim_output, "dt")),
        tx_delays=np.atleast_1d(_leaf(sim_output, "tx_delays")).astype(float),
        frequency=float(_leaf(TX, "frequency")),
        n_cycle=float(_leaf(TX, "n_cycle", default=1)),
        polar_incidence_angle=float(_leaf(TX, "polar_incidence_angle", default=np.nan)),
        c_ref=float(_leaf(TX, "c_ref", default=1500.0)),
        tx_mag=float(_leaf(TX, "mag", default=np.nan)),
        element_count=int(_leaf(transducer, "element_count", default=cd.shape[1])),
        pitch=float(_leaf(transducer, "target_pitch")),
        kerf=float(_leaf(transducer, "target_kerf")),
        ID=float(_leaf(target, "ID")),
        OD=float(_leaf(target, "OD")),
        transducer_ref_zx=tref,        # (z, x) as stored
        pipe_pos_m_zx=ppos,            # (z, x) as stored
        standoff=float(_leaf(sim_output, "standoff", default=np.nan)),
        dx=float(_leaf(sim_output, "dx", default=np.nan)),
        dy=float(_leaf(sim_output, "dy", default=np.nan)),
        c_fluid=c_f, rho_fluid=rho_f, c_L_steel=c_L, c_S_steel=c_S, rho_steel=rho_s,
        grid_res=float(_leaf(simulation, "grid_res", default=np.nan)),
        t_end=float(_leaf(simulation, "t_end", default=np.nan)),
        source_name=str(run.name),
    )

    # crack descriptor, when the run has one
    cracks = _leaf(target, "cracks")
    inc = _leaf(target, "include_crack", default=0)
    data["include_crack"] = int(np.atleast_1d(inc).ravel()[0]) if inc is not None else 0
    if cracks is not None and data["include_crack"]:
        for f in ("height", "width", "angle", "offset", "azimuth"):
            v = _leaf(np.atleast_1d(cracks)[0], f)
            if v is not None:
                data[f"crack_{f}"] = float(np.atleast_1d(v).ravel()[0])

    if args.tt_t:
        import h5py                                  # v7.3 = HDF5
        with h5py.File(args.tt_t, "r") as f:
            d = f["data"]
            arr = d[...]
            if arr.dtype.names and set(arr.dtype.names) >= {"real", "imag"}:
                arr = arr["real"] + 1j * arr["imag"]
            data["ref_TT_T"] = arr
            for k in ("coordinates",):
                if k in f:
                    try:
                        data[f"ref_{k}"] = np.array(
                            [f[r][...].squeeze() for r in np.atleast_1d(f[k][...]).ravel()],
                            dtype=object)
                    except Exception as exc:          # noqa: BLE001
                        print(f"note: could not resolve ref_{k}: {exc}")
        print(f"ref TT_T  : {data['ref_TT_T'].shape} {data['ref_TT_T'].dtype}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)

    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"  channel_data      {cd.shape} {cd.dtype}  |amp|max {np.abs(cd).max():.4g}")
    print(f"  dt                {data['dt']:.6e} s   -> fs {1/data['dt']/1e6:.3f} MHz")
    print(f"  t_end             {data['t_end']:.3e} s  ({cd.shape[0]} samples)")
    print(f"  steering          {data['polar_incidence_angle']:+.1f} deg")
    print(f"  tx_delays         min {data['tx_delays'].min():.3e}  "
          f"max {data['tx_delays'].max():.3e} s  (span {np.ptp(data['tx_delays'])*1e6:.3f} us)")
    print(f"  pipe ID/OD        {data['ID']*1e3:.3f} / {data['OD']*1e3:.3f} mm  "
          f"-> wall {(data['OD']-data['ID'])/2*1e3:.3f} mm")
    print(f"  standoff          {data['standoff']*1e3:.2f} mm")
    print(f"  speeds            fluid {c_f:.0f} | steel L {c_L:.0f} S {c_S:.0f} m/s")
    print(f"  include_crack     {data['include_crack']}")
    for k in sorted(k for k in data if k.startswith("crack_")):
        print(f"    {k:16s} {data[k]}")


if __name__ == "__main__":
    main()
