# Containers

## LibreOffice (headless document conversion)

`libreoffice.def` builds a small Debian image with headless LibreOffice. It is
used to render `.docx` files to PDF (and then to page images) for checking
layout. Build and run it only inside an `salloc` job, never on a login node.
Following the Alliance Apptainer page, the build uses node-local storage
(`$SLURM_TMPDIR`) for the cache and temporary files, not Lustre (`/project`,
`/scratch`). The finished `.sif` is copied to project space so it can be
reused.

```bash
# 1. On a login node, from the project directory: a small interactive job
cd /project/6116810/digital_twin/exp1
salloc --account=def-roudsari --time=1:00:00 --cpus-per-task=2 --mem=4G

# 2. Inside the job: build on node-local disk (~5-15 min, downloads ~300 MB)
module load apptainer/1.4.5
export APPTAINER_CACHEDIR=$SLURM_TMPDIR/apptainer-cache
export APPTAINER_TMPDIR=$SLURM_TMPDIR/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer build --fakeroot "$SLURM_TMPDIR/libreoffice.sif" containers/libreoffice.def

# 3. Keep the image in project space (outside the git repo)
mkdir -p /project/6116810/digital_twin/containers
cp "$SLURM_TMPDIR/libreoffice.sif" /project/6116810/digital_twin/containers/

# 4. Convert a document: PDF goes next to it, here in docs/
apptainer exec -C -B /project -W "$SLURM_TMPDIR" \
    /project/6116810/digital_twin/containers/libreoffice.sif \
    soffice --headless --convert-to pdf --outdir "$PWD/docs" "$PWD/docs/llm_selection.docx"

# 5. Leave the job
exit
```

Later conversions only need steps 1, `module load apptainer/1.4.5`, 4 and 5.
For those, 1 CPU and 2G of memory are enough.
