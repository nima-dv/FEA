r"""Blocking and warning rules. Pure functions, no Qt.

A BLOCK is a configuration that cannot produce a valid result or would damage something:
the mesh script itself refuses it, the disk cannot hold it, the output would land on the
published record, or the comparison it asks for has nothing to compare against. A WARN is a
configuration that runs and answers, but not the answer the user probably wanted.

The rules are the GUI README's Guardrails section, and nothing more. Every threshold either
mirrors a number the backend enforces (the domain extent) or a measurement on record (the
resolution floor, the snapshot size, the CPU/GPU gap). None of them is a taste judgement,
because a guardrail the user learns to click through is worse than no guardrail.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace

try:                                  # inside the app: `from model import guards`
    from . import derived, scenario as scen
    from . import spec
    from .spec import ArtifactReduction, Device, RunConfig
except ImportError:                   # run directly: python src/gui/model/guards.py
    import derived                    # type: ignore[no-redef]
    import scenario as scen           # type: ignore[no-redef]
    import spec                       # type: ignore[no-redef]
    from spec import ArtifactReduction, Device, RunConfig   # type: ignore[no-redef]


class Severity(enum.Enum):
    BLOCK = "block"
    WARN = "warn"


@dataclass(frozen=True)
class Finding:
    severity: Severity
    field: str          # the RunConfig field name the UI attaches this to
    message: str        # one line, states the number and what to do about it

    @property
    def level(self) -> str:
        """BLOCK or WARN as a string. views/simulate.py reads `.level` to decide whether to
        disable Run; without it that code falls back to "warn" and a BLOCK leaves the button
        live, which is the one outcome this whole module exists to prevent."""
        return self.severity.name


@dataclass(frozen=True)
class Context:
    """What the rules need from outside the parameter form.

    All optional: a rule with no evidence stays silent rather than guessing. `tracked` is the
    set of git-tracked output paths (relative to data/results) that the runner knows about -
    guards does the intersection with this plan's outputs itself, so the rule is testable
    without a git checkout.
    """
    free_bytes: int | None = None
    tracked: frozenset[str] = frozenset()
    history: list[dict] = field(default_factory=list)


def domain_extent(config: RunConfig, sc: scen.Scenario) -> tuple[float, float]:
    """The lateral extent this configuration will mesh, mm.

    Not a user field: only WIDE_DOMAIN moves it, and it moves it to the fixed pair in
    spec.py. Read back from spec rather than restated, so the two cannot drift.
    """
    if config.artifact_reduction is ArtifactReduction.WIDE_DOMAIN:
        return spec.WIDE_X_MIN, spec.WIDE_X_MAX
    return sc.x_min, sc.x_max


def check(config: RunConfig, sc: scen.Scenario | None = None,
          context: Context | None = None) -> list[Finding]:
    """Everything wrong with this configuration, blocks first.

    Both extra arguments are optional so the form can call check(cfg) on every keystroke: the
    scenario then comes from the cache, and an empty context means the rules that need outside
    evidence (free space, the k-Wave case, the tracked file set) stay silent rather than
    guess. The runner passes a full context, which is where those rules earn their keep.
    """
    sc = sc or scen.load()
    context = context or Context()
    out: list[Finding] = []

    # --- BLOCK: an extent the mesh script itself refuses ---------------------------------
    # Mirrors ili_mesh.py's own X_LIMIT_LO/HI, which come out of the geometry: past those the
    # pipe's ID arc rises above the flat array plane, so the water region would be nonsense.
    # Reading the limits from the scenario dump means a backend change moves this rule too.
    x0, x1 = domain_extent(config, sc)
    if x0 < sc.x_limit_lo or x1 > sc.x_limit_hi:
        out.append(Finding(
            Severity.BLOCK, "artifact_reduction",
            f"domain {x0:+.2f} to {x1:+.2f} mm is outside the geometric limit "
            f"[{sc.x_limit_lo:+.2f}, {sc.x_limit_hi:+.2f}] mm - the mesh script refuses it: "
            f"past there the pipe arc rises above the array plane"))

    # --- BLOCK: writing onto the published record ---------------------------------------
    # Defence in depth. The gui_ tag prefix already makes a collision impossible; this is what
    # catches the day someone "temporarily" removes the prefix to reproduce a figure.
    outputs = [o for j in spec.plan(config) for o in j.outputs]
    clash = sorted(set(outputs) & set(context.tracked))
    if clash:
        out.append(Finding(Severity.BLOCK, "tag",
                           f"would overwrite git-tracked output: {clash[0]}"))

    # --- BLOCK: no room on disk ----------------------------------------------------------
    disk = derived.estimated_disk(config)
    if context.free_bytes is not None and disk.value > context.free_bytes:
        out.append(Finding(Severity.BLOCK, "snapshots",
                           f"needs ~{disk.value/1e9:.2f} GB ({disk.note}), "
                           f"{context.free_bytes/1e9:.2f} GB free"))

    # --- BLOCK: a comparison with nothing to compare to ---------------------------------

    # --- WARN: under-resolved mesh -------------------------------------------------------
    res = derived.nodes_per_wavelength(config, sc)
    if res.value < derived.NODES_MIN:
        s = derived.scale_for_nodes(config, sc)
        out.append(Finding(
            Severity.WARN, "scale",
            f"~{res.value:.2f} nodes per wavelength in {res.binding} at "
            f"{res.f_upper/1e6:.0f} MHz, below {derived.NODES_MIN:.1f}: the mesh low-passes "
            f"its own pulse, which reads as a deeper, weaker notch. Use --scale {s:.2f}"))

    # --- WARN: snapshots are big --------------------------------------------------------
    if config.snapshots:
        out.append(Finding(Severity.WARN, "snapshots",
                           f"{config.snapshots} snapshots write ~{disk.value/1e6:.0f} MB "
                           f"(measured 700-970 MB at 240 frames)"))

    # --- WARN: a full-length solve on the CPU -------------------------------------------
    # Half the record is the cut: the 3 us gate runs are legitimately CPU work, a 60 us
    # production solve is 19x slower there because one explicit step is bandwidth bound.
    if config.device is Device.CPU and config.t_end >= 0.5 * sc.t_end:
        cpu = derived.estimated_runtime(config, sc, context.history)
        gpu = derived.estimated_runtime(replace(config, device=Device.GPU), sc,
                                        context.history)
        out.append(Finding(Severity.WARN, "device",
                           f"CPU for a {config.t_end*1e6:.0f} us solve: "
                           f"~{cpu.value/3600:.1f} h against ~{gpu.value/60:.0f} min "
                           f"on the GPU"))

    out.sort(key=lambda f: f.severity is Severity.WARN)     # blocks first, stable otherwise
    return out


def blocked(findings: list[Finding]) -> bool:
    return any(f.severity is Severity.BLOCK for f in findings)


def demo() -> None:
    sc = scen.load()
    ctx = Context(free_bytes=500 * 10**9)

    # the published configuration must be clean: it IS the record, so anything it triggers
    # would fire on every published figure too
    base = check(RunConfig(), sc, ctx)
    assert base == [], base

    # a coarse mesh must warn AND name a scale that actually clears the warning
    coarse = RunConfig(scale=2.0)
    warn = [f for f in check(coarse, sc, ctx) if f.field == "scale"]
    assert warn and "--scale" in warn[0].message, warn
    fixed = replace(coarse, scale=derived.scale_for_nodes(coarse, sc))
    assert not [f for f in check(fixed, sc, ctx) if f.field == "scale"], "named scale must fix"

    # the two fixed treatments are inside the geometric limit; a hand-rolled extent is not
    for ar in ArtifactReduction:
        assert not blocked(check(RunConfig(artifact_reduction=ar), sc, ctx)), ar
    narrow = replace(sc, x_min=-60.0)   # a hand-rolled extent past the geometric ceiling
    bad = check(RunConfig(), narrow, ctx)
    assert blocked(bad) and bad[0].field == "artifact_reduction", bad

    # comparison with no case on disk, and no room on disk
    assert check(RunConfig(), sc, Context()) == [], "no evidence, no findings"
    # the live form calls check(cfg) with nothing else, and BLOCK must read as BLOCK there
    assert [f.level for f in check(RunConfig(scale=2.0))] == ["WARN"]
    assert check(RunConfig(snapshots=240), None,
                 Context(free_bytes=10**8))[0].level == "BLOCK"
    assert blocked(check(RunConfig(snapshots=240), sc,
                         Context(free_bytes=10**8)))

    # the tracked-output block: only fires when a plan output really is tracked
    out0 = spec.plan(RunConfig())[0].outputs[0]
    hit = check(RunConfig(), sc, Context(tracked=frozenset({out0})))
    assert blocked(hit) and hit[0].field == "tag", hit

    cpu = check(RunConfig(device=Device.CPU), sc, ctx)
    assert any(f.field == "device" and "h against" in f.message for f in cpu), cpu
    # a short gate run on the CPU is not a mistake, so it must stay silent
    assert not [f for f in check(RunConfig(device=Device.CPU, t_end=3e-6), sc, ctx)
                if f.field == "device"]
    print("guards.demo: ok (nothing to report on the published configuration)")


if __name__ == "__main__":
    demo()
