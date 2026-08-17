# GitHub upload guide

Recommended repository name:

```text
LEBRef
```

Recommended visibility before journal submission:

```text
Public
```

Recommended short description:

```text
Code and reproducibility materials for Linear Event-Bag Refinement in wrist-worn fall detection.
```

## Manuscript statement

After the GitHub repository is created, update the manuscript placeholder:

```text
https://github.com/xm2284/LEBRef
```

to the final repository URL.

## Manual upload route

If GitHub CLI is not installed, create an empty GitHub repository named
`LEBRef`, then upload the contents of this folder.

Do not upload:

- raw datasets;
- pretrained checkpoints;
- feature caches;
- local `artifacts/`, `outputs/`, or `data/` folders;
- private manuscript comments or review PDFs.

## Git command route

From this folder:

```bash
git init
git branch -M main
git add .
git commit -m "Release LEBRef reproducibility materials"
git remote add origin https://github.com/xm2284/LEBRef.git
git push -u origin main
```

If you use Zenodo later, create a GitHub release first and then archive the
release through Zenodo to obtain a DOI.
