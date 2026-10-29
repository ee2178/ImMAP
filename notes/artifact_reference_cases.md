# Artifact-detection reference cases (from the ISMRM MGLPDS recon work)

Two volumes whose reconstructions show visible artifacts and two that come out clean, picked
from the material behind the multigrid-recon ISMRM abstract (`docs/ismrm_mglpds/`, untracked
now; read from git history at `979811d^`).

## The four picks

| # | Dataset | Volume | Slices | Artifact | How judged | Confidence |
|---|---|---|---|---|---|---|
| 1 | fastMRI brain, multicoil AXT2, test | `file_brain_AXT2_202_2020522` | 0–7, worst 0 and 1 | Residual aliasing: thin bright and dark lines tracking the skull edge near the left of the FOV (array col ≈ 38–52) | Alias-scan report and PNGs (`results/alias_detect/brain/`, R8 and R12) | **High** that the recons show artifacts. **Medium** on the cause (see notes) |
| 2 | fastMRI knee, multicoil PD, test | `file1000264` | 13–18, worst 14–16 | Aliasing ghost: shifted copies of the fat/muscle boundary along the lateral edge (row ≈ 430–470, col ≈ 85–105) | Alias-scan report and PNG `R8_lpdsnet_06_file1000264_s14.png` | **High** |
| 3 | fastMRI brain, multicoil AXT2, test | `file_brain_AXT2_200_2000024` | 0–7 | Clean | Alias scan: score 0 on every slice for all 3 methods at R8 and R12 | **Medium**: scan numbers only, no image viewed |
| 4 | fastMRI knee, multicoil PD, test | `file1000635` | 12–24 | Clean (cleanest knee) | Alias scan: lowest knee total. R8: 4 of 39 method-slices flagged, sum 272. R12: sum 3947, the lowest of 47 knee volumes | **Medium–low**: no knee scores zero at R12 (the knee detector flags 98–99 % of R12 slices) |

"Methods" means E2E-VarNet (`varnetmaps`), LPDSNet (`lpdsnet`) and MGLPDSNet (`mg81v6`). "R"
is the undersampling factor.

Other clean brain volumes (score 0 everywhere): `2000133`, `2000270`, `2000300`, `2000309`,
`2000371`, `2000392`, `2000514` (all `file_brain_AXT2_200_…`).

## What the ISMRM MGLPDS figures used

- **Fig. 3 (residual aliasing)**, `docs/ismrm_mglpds/figures/fig3_alias.png`. Panel files are
  under `figures/aliasing/`.
  - brain R8: `file_brain_AXT2_200_2000182`, slice 0 (bright streak in LPDSNet)
  - brain R12: `file_brain_AXT2_200_2000182`, slice 6 (dark line)
  - knee R12: `file1000283`, slice 20 (oblique streak)
- **Fig. 2 (MG wins)**, `figures/rows/mgwins_*.json` (tracked):
  - brain R8: `2000182` s2
  - brain R12: `2000182` s5
  - knee R8: `file1000033` s22
  - knee R12: `file1000052` s19
- **Alias scan**: `scripts/alias_scan.py` and `scripts/alias_pick.py`. The output was
  `results/alias_detect/{brain,knee}/detect_R{8,12}.{txt,csv}`, now untracked and read from git
  history. It covered 68 brain volumes (slices 0–7) and 47 knee volumes (slices 12–24).

The Fig. 3 cases are good examples of the paper's point, but they are weak reference cases. They
were picked by eye, their artifacts are faint, and only LPDSNet shows them: `2000182` scores at
most 84 in the scan. `2020522` and `file1000264` show strong artifacts that all three networks
produce.

## Caveats

- **These are reconstruction artifacts**, from undersampled recons at R=8 and R=12. They are not
  judged as acquisition artifacts in the fully sampled reference. In the scan PNGs the reference
  images of picks 1 and 2 look normal at that crop. In `2020522`, every method has the artifact
  on every slice, which hints at something in the data itself (wrap-around or signal near the FOV
  edge, or an odd coil setup). Checking that needs the k-space.
- **The data isn't on this machine.** The only fastMRI file here is
  `~/Desktop/file_brain_AXT2_202_2020322.h5`, which is not one of the scanned volumes.
  `brain_dumps.tar` at the repo root is truncated and holds 3 LPDSNet R12 dumps (`2000019`,
  `2000021`, `2000022`). `trained_nets/` does not exist locally. No images were rendered for this
  note. The dumps and scan are on the cluster under `~/scratch/ee2178/ImMAP/trained_nets/mg_recon/`.
- The clean picks rely on a line detector ("new lines" in the recon that are not in the
  reference: z > 3, ≥ 8 px). That detector does not catch blur, noise or contrast errors.
