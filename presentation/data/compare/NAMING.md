# Filename tags in results/compare/

Pattern: `compare_<angle>deg[_tag].png` and `images_<angle>[_tag].npz`, written by
`repro/compare_images.py --tag <tag>`. `p20` / `m20` mean +20 / -20 deg steering.
Nothing here is ever renamed in place; a new variant gets a new tag instead.

NO TAG        the current published baseline: the `faithfulbf` imaging chain (decimation
              factor 15, engine numba, migration antialias=0.5, 0.6-1.4*f0 bandpass) on
              the legacy absorbing boundary. Adopted 2026-08-19. Every quoted number and
              figure comes from these files.

_legacybf     the PREVIOUS published baseline, kept for reproducibility: the `legacy`
              chain (decimation 23, engine numpy, no operator antialias). These are the
              untagged files as they stood before 2026-08-19, renamed not regenerated -
              byte-identical, original mtimes. `--chain legacy` reproduces them exactly.

_boundary     absorbing-boundary experiment: shear-matched dashpot, and shear-matched
              + graded sponge, against legacy. Both came back negative.
_healthy_boundary   defect-free wall, so every pixel in the image is numerical.
_clean        superseded. It was the un-annotated twin of `_boundary`, named this way
              only because `--tag` and `--no-overlay` used to be mutually exclusive.
              Its data is bit-identical to `_boundary`. Use `_boundary_nooverlay`.
_p3s10        the older degree-3, --scale 1.0 mesh configuration. Legacy chain only:
              the degree-3 channel data is no longer on disk, so it cannot be regenerated.

_reprocheck   (transient) legacy chain re-run to prove bit-identity; deleted after use.
_numbanull    null check evidence: engine numba with antialias=0, which must equal
              `legacy`. Measured max relative deviation 1.2e-07 against a pre-registered
              1e-4 bound, so the engine swap is not a confound.

_nooverlay    no wall arcs and no true-notch marker, so detectability can be judged
              unaided. Every annotated comparison figure has one of these twins; the
              tags compose, e.g. `compare_p20deg_boundary_nooverlay.png`.
