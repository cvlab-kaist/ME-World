# ME-World: Multi-Agent Egocentric World Model with Fine-Grained Embodied Interaction

Project page for **ME-World**, a world model that generates synchronized first-person video for several agents
interacting in one shared world. Each agent's ego stream is denoised jointly with the others in a single token
sequence, conditioned on every agent's body motion projected into that agent's camera (shared action
conditioning), and grounded by a shared environment memory built from all agents' observation history
(warped history frames plus clean anchor references). The model is trained on real two-person recordings
(CoMind) and a synthetic set rendered from retargeted Inter-X / InterHuman interactions, and is evaluated
with new shared-world consistency metrics for the environment, interaction-induced updates and agent identity.

Dahyun Chung, Siyoon Jin, Hyunwook Choi, Honggyu An, Junyoung Seo, Hyunsung Kim, Seung Wook Kim, Seungryong Kim · KAIST AI

Live page: https://cvlab-kaist.github.io/ME-World/

## Release checklist
- [ ] Enable GitHub Pages: Settings → Pages → Deploy from a branch → `main` / root
- [ ] arXiv: set the Paper button `href` and remove its `disabled` class in `index.html`; fill the arXiv id in the BibTeX
- [ ] Code & Weights: set the button `href` once the code repo is public
- [ ] Confirm the venue line in the hero (`arXiv Preprint 2026` now)
- [ ] Add `assets/favicon.ico`
- [ ] Decide what to do with the mirror at `dhyun22.github.io/ME-World` (delete, or redirect here)
- [ ] Acknowledgements section (removed for now; add back before camera-ready if needed)
- [ ] Optional: move `videos/` (~350 MB) to Git LFS or re-encode if the repo gets heavy

## Layout
```
index.html, style.css          page (single static page, no build step)
videos/
  teaser_video.mp4             145 s overview video embedded in the hero (built by tools/make_teaser.py)
  teaser/                      real-benchmark results (Real Interactions section)
  synthetic_teaser/            synthetic results + third-person reference views
  application/                 three-agent and 221-frame autoregressive results
  comparison/                  GT / ours / 6 baselines on 3 examples (tab switcher)
  ablation/                    the 8 variants of Table 2
  dataset/                     real and synthetic data examples with pose conditions
  method/shared_action.mp4     shared action conditioning explainer
  memory/                      shared environment memory visualization (+ generated clip it uses)
assets/model.png               architecture figure (rasterized from the paper figure)
tools/                         scripts that produce the explainer videos (see below)
```

## Rebuilding the videos
All scripts run in the `xmetric` env unless noted; caches land in `tools/cache/` (git-ignored).

- `tools/make_teaser.py --out videos/teaser_video.mp4` — the overview video, cut from the clips in `videos/`.
  `--only <segment>` renders one segment (title, hook, real, synth, arch, action, memory, multi, compare, ablation).
  Needs `tools/cache/history_58dd70af_069576.npz` (below) and the Carlito fonts in `tools/fonts/`.
- `tools/extract_history.py` (comind env, `COMIND_SHARED_WORLD=1`) — observation-history frames of a CoMind clip for the memory explainer.
- `tools/extract_traj.py` (comind env, `COMIND_SHARED_WORLD=1`) — all 277 shared-world camera poses of a CoMind clip.
- `tools/memory_viz.py` — the 3D shared-environment-memory visualization (DA3 depth with GT extrinsics, trajectories, warp source, anchors).

## Local preview
```bash
python -m http.server 8000   # then open http://localhost:8000
```
