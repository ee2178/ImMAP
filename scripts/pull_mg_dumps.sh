#!/usr/bin/env bash
# Pull the MG-recon eval dumps of the ISMRM figure volumes and the artifact reference cases from
# torch, as one tarball, and unpack it here.
#
#     bash scripts/pull_mg_dumps.sh                 # from the repo root, on the laptop
#     HOST=dtn.torch.hpc.nyu.edu bash scripts/pull_mg_dumps.sh
#
# Each dump (trained_nets/mg_recon/<anatomy>/<run>_R<R>/eval_dump/<volume>.h5, written by
# scripts/dump_eval.py) holds `reference` (fully sampled), `recon` (that run's output) and
# `zero_filled`, so the three runs below give all four images. torch takes a password + Duo, so
# expect two logins: one to build the tarball, one to copy it down.
#
# The tarball (mg_ref_dumps.tar) is kept beside the repo root, to move to another cluster as is:
#     scp mg_ref_dumps.tar <other>:<ImMAP>/ && ssh <other> 'cd <ImMAP> && tar xf mg_ref_dumps.tar'
# Unpacked, the dumps land under trained_nets/ (gitignored), where figures/configs/* look.

set -euo pipefail

HOST=${HOST:-torch}
TAR=mg_ref_dumps.tar
RUNS="varnetmaps lpdsnet mg81v6"
RS="8 12"
# fig3_alias (2000182, file1000283), mgwins / Fig. 2 (2000182, file1000033, file1000052),
# notes/artifact_reference_cases.md (2020522, 2000024, file1000264, file1000635)
BRAIN="file_brain_AXT2_200_2000182 file_brain_AXT2_202_2020522 file_brain_AXT2_200_2000024"
KNEE="file1000283 file1000033 file1000052 file1000264 file1000635"

# The settings go down as the script's first line: ssh joins its arguments into one command
# string, so passing them as `bash -s -- "$RUNS" ...` would split every list on the far side.
remote_tar=$({ printf 'TAR=%q RUNS=%q RS=%q BRAIN=%q KNEE=%q\n' "$TAR" "$RUNS" "$RS" "$BRAIN" "$KNEE"
               cat <<'REMOTE'
set -euo pipefail
USER=${USER:-$(whoami)}
for d in /scratch/$USER/ImMAP ~/scratch/$USER/ImMAP ~/ImMAP; do
    [ -d "$d/trained_nets/mg_recon" ] && { cd "$d"; break; }
done
[ -d trained_nets/mg_recon ] || { echo "no trained_nets/mg_recon found" >&2; exit 1; }
echo "repo: $PWD"
list=$(mktemp)
for anat in brain knee; do
    vols=$BRAIN; [ $anat = knee ] && vols=$KNEE
    for run in $RUNS; do for R in $RS; do for v in $vols; do
        f=trained_nets/mg_recon/$anat/${run}_R$R/eval_dump/$v.h5
        if [ -f "$f" ]; then echo "$f" >> "$list"; else echo "missing: $f" >&2; fi
    done; done; done
done
n=$(wc -l < "$list")
[ "$n" -gt 0 ] || { echo "none of the dumps found -- nothing to pack" >&2; rm -f "$list"; exit 1; }
echo "$n files, $(du -ch $(cat "$list") | tail -1 | cut -f1)"
tar cf "$TAR" -T "$list"
rm -f "$list"
echo "wrote $PWD/$TAR"
REMOTE
             } | ssh "$HOST" bash -s | tee /dev/stderr | sed -n "s/^wrote //p")

[ -n "$remote_tar" ] || { echo "tarball not built" >&2; exit 1; }
scp "$HOST:$remote_tar" .
tar tf "$TAR" | wc -l
tar xf "$TAR"
echo "unpacked under trained_nets/mg_recon; tarball kept at $PWD/$TAR"
