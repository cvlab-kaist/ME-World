# Page maintenance notes (not part of the public README)

## Release checklist
- [ ] Enable GitHub Pages: Settings → Pages → Deploy from a branch → `main` / root
- [ ] arXiv: set the Paper button `href` and remove its `disabled` class in `index.html`; fill the arXiv id in the BibTeX (page + README); swap the arXiv badge in README
- [ ] Code & Weights: button now links to the repo; update once code is public
- [ ] Confirm the venue line in the hero (`arXiv Preprint 2026` now) and the README badge
- [ ] Add `assets/favicon.ico`
- [ ] Decide what to do with the mirror at `dhyun22.github.io/ME-World` (delete, or redirect here)
- [ ] Acknowledgements section (removed from the page for now)
- [ ] Optional: move `videos/` (~350 MB) to Git LFS or re-encode if the repo gets heavy

## Layout
```
index.html, style.css          page (single static page, no build step)
videos/teaser_video.mp4        145 s overview video embedded in the hero (built by tools/make_teaser.py)
videos/teaser/                 real-benchmark results            videos/synthetic_teaser/  synthetic results + TPV
videos/application/            three-agent and 221-frame AR      videos/comparison/        GT / ours / 6 baselines
videos/ablation/               the 8 variants of Table 2         videos/dataset/           data examples with pose conditions
videos/method/                 shared action explainer           videos/memory/            shared environment memory viz
assets/model.png               architecture figure               assets/teaser.jpg         README header image
```

## Rebuilding the videos
All scripts run in the `xmetric` env unless noted; caches land in `tools/cache/` (git-ignored).
- `make_teaser.py --out ../videos/teaser_video.mp4` — the overview video, cut from the clips in `videos/`.
  `--only <segment>` renders one segment (title, hook, real, synth, arch, action, memory, multi, compare, ablation).
  Needs `cache/history_58dd70af_069576.npz` and the Carlito fonts in `fonts/`. The shared-action segment uses Open3D
  offscreen (EGL) mesh rendering.
- `extract_history.py` (comind env, `COMIND_SHARED_WORLD=1`) — observation-history frames of a CoMind clip for the memory explainer.
- `extract_traj.py` (comind env, `COMIND_SHARED_WORLD=1`) — all 277 shared-world camera poses of a CoMind clip.
- `memory_viz.py` — the 3D shared-environment-memory visualization (DA3 depth with GT extrinsics, trajectories, warp source, anchors).

## Local preview
```bash
python -m http.server 8000   # then open http://localhost:8000
```
