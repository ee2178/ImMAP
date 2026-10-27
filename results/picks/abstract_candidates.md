# ISMRM abstract: candidate slices (2026-10-27)

From `results/main/slices.csv` (every test volume, PSNR, measured k-space + AWGN), via
`scripts/rank_slices.py --from-csv`. MGLPDS = `mg81v6` (the net in the main table).
"viewer filter" = what to type into the web viewer's filter box.

## 1. Largest MGLPDS improvement (mg81v6 vs the better of E2E-VarNet and LPDSNet)

| task | volume | slice | VarNet | LPDSNet | MGLPDS | margin (dB) | viewer filter |
|---|---|---|---|---|---|---|---|
| brain R8  | file_brain_AXT2_207_2070217 | 5  | 21.43 | 21.06 | 22.79 | +1.35 | `2070217 5` |
| brain R8  | file_brain_AXT2_202_2020522 | 7  | 26.77 | 26.33 | 27.88 | +1.11 | `2020522 7` |
| brain R12 | file_brain_AXT2_207_2070745 | 4  | 21.38 | 21.31 | 22.91 | +1.53 | `2070745 4` |
| brain R12 | file_brain_AXT2_208_2080542 | 7  | 20.14 | 20.17 | 21.68 | +1.52 | `2080542 7` |
| knee R8   | file1002417 | 15 | 22.51 | 21.59 | 23.45 | +0.94 | `file1002417 15` |
| knee R8   | file1000735 | 17 | 26.68 | 26.47 | 27.52 | +0.84 | `file1000735 17` |
| knee R12  | file1000283 | 14 | 24.10 | 24.72 | 26.01 | +1.29 | `file1000283 14` |
| knee R12  | file1000925 | 24 | 25.48 | 26.40 | 27.59 | +1.19 | `file1000925 24` |

## 2. LPDSNet failures (LPDSNet alone drops; VarNet and MGLPDS do not): aliasing candidates

Whole runs of adjacent slices where LPDSNet trails by 2-3 dB while VarNet and MGLPDS agree --
the signature of a structured artifact rather than noise. PSNR cannot say it is aliasing:
confirm with `scripts/alias_scan.py` on the dumps (ghost / gerr / "x others") and by eye.

| task | volume | slices | VarNet | LPDSNet | MGLPDS | LPDSNet deficit (dB) | viewer filter |
|---|---|---|---|---|---|---|---|
| knee R12  | file1001122 | 16-19 | 28.3-29.3 | 26.8-27.2 | 28.7-29.6 | -1.8 to -2.7 | `file1001122 18` |
| brain R12 | file_brain_AXT2_202_2020522 | 0-3 | 24.6-25.7 | 22.1-24.0 | 24.7-26.2 | -2.2 to -2.9 | `2020522 2` |
| brain R8  | file_brain_AXT2_207_2070745 | 6-7 | 24.4-24.9 | 22.2-22.3 | 24.5 | -2.3 | `2070745 6` |
| brain R8  | file_brain_AXT2_202_2020088 | 6 | 26.43 | 24.84 | 27.16 | -2.32 | `2020088 6` |
| knee R8   | file1002417 | 15 | 22.51 | 21.59 | 23.45 | -1.86 | `file1002417 15` |
| knee R8   | file1002187 | 12 | 24.95 | 24.18 | 25.70 | -1.51 | `file1002187 12` |

Local check: the four knee volumes dumped on this laptop (file1000031/33/41/52) show NO
LPDSNet-specific aliasing (its ghost error is 0.9-1.3x the other nets'), so the candidates
above have to be checked on the cluster dumps.

## 3. Aliasing scan of the cluster dumps (2026-10-27, results/alias_scan_main on Torch)

`alias_scan.py --target lpdsnet --methods varnetmaps lpdsnet mg81v6` over the default dumps
(brain: 60 volumes, all site 200; knee: first 40). NONE of the section-2 candidates is in
those dumps. gerr = size of the strongest ghost as a fraction of the image; x others =
LPDSNet's gerr over the median of VarNet and MGLPDS.

Mean gerr, LPDSNet / MGLPDS: brain R8 0.0153 / 0.0105 (LPDSNet 1.46x); brain R12 0.0205 /
0.0234; knee R8 0.0144 / 0.0139; knee R12 0.0148 / 0.0149. Only brain R8 separates.

Best visible cases (large ghost AND above the others):

| task | volume | slice | gerr | x others | ghost share of error | viewer filter |
|---|---|---|---|---|---|---|
| knee R8  | file1000758 | 12 | 0.0383 | 1.2x | 0.35 | `file1000758 12` |
| knee R8  | file1000552 | 12 | 0.0358 | 1.2x | 0.34 | `file1000552 12` |
| knee R8  | file1000308 | 15 | 0.0295 | 1.3x | 0.30 | `file1000308 15` |
| knee R8  | file1000153 | 17 | 0.0265 | 1.3x | 0.28 | `file1000153 17` |
| brain R8 | file_brain_AXT2_200_2000130 | 7 | 0.0287 | 1.2x | 0.20 | `2000130 7` |
| brain R8 | file_brain_AXT2_200_2000337 | 5 | 0.0246 | 1.3x | 0.21 | `2000337 5` |
| brain R8 | file_brain_AXT2_200_2000414 | 7 | 0.0230 | 1.3x | 0.18 | `2000414 7` |

The 1.4-1.5x slices (file1000277 23-24, file1000537 22) have tiny ghosts (gerr < 0.01): not visible.
