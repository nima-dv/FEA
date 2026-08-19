# Filename tags in results/compare/

Pattern: `compare_<angle>deg[_tag].png` and `images_<angle>[_tag].npz`, written by
`repro/compare_images.py --tag <tag>`. `p20` / `m20` mean +20 / -20 deg steering.
Nothing here is ever renamed; a new variant gets a new tag instead.

NO TAG        the published record: the `legacy` imaging chain (stride decimation
              factor 23, engine numpy, no migration-operator antialias, 0.6-1.4*f0
              bandpass) on the `legacy` absorbing boundary (scalar c_P dashpot).
              These are the files every quoted number and figure comes from.

_boundary     absorbing-boundary experiment: shear-matched dashpot, and shear-matched
              + graded sponge, against legacy. Both came back negative.
_healthy      defect-free wall, so every pixel in the image is numerical.
_clean        same comparison redrawn without the extra variant panels.
_p3s10        the older degree-3, --scale 1.0 mesh configuration.

_reprocheck   `legacy` chain re-run to prove it still reproduces the untagged files
              bit-for-bit (2026-08-19: max relative deviation 0.000e+00).
_faithfulbf   the research team's own beamformer options: decimation factor 15,
              engine numba, migration antialias=0.5. See lib/tt_t_image.CHAINS.
_nobandpass   _faithfulbf with our 0.6-1.4*f0 bandpass removed, which their
              simulation beamforming script also does not have.
_numbanull    null check: engine numba with antialias=0, which must equal legacy.

_nooverlay    no wall arcs and no true-notch marker, so detectability can be judged
              unaided. Every annotated comparison figure has one of these twins; the
              tags compose, e.g. `compare_p20deg_faithfulbf_nooverlay.png`.
