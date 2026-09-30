# Project page

Static GitHub Pages site. Layout follows https://seoul-world-model.github.io/ (fixed scroll-spy sidebar, centred hero, TL;DR box, lazy-loaded demo videos, BibTeX copy button), with its own palette (ink-blue for Agent 1, coral for Agent 2, green for Agent 3) and Pretendard type.

## Files
- `index.html` — page content. Search for `TODO` to find every placeholder.
- `style.css` — all styling. Colours and fonts are CSS variables at the top.
- `videos/` — demo clips grouped by section (teaser, synthetic_teaser, application, comparison, ablation, dataset), 292 MB total. Consider Git LFS or re-encoding if the repo gets too large.
- `assets/model.png` — architecture figure rasterised from `pdf/main_architecture.pdf`; `assets/teaser_poster.jpg` — poster for the first hero clip. Add `favicon.ico` here.

## Placeholders to fill (grep TODO)
1. Open Graph URL, favicon.
2. Homepage links for authors without one; confirm venue line (paper header says ICLR 2027, page says arXiv 2026).
3. Paper / Code buttons: set `href` and remove the `disabled` class.
4. arXiv id in the BibTeX; acknowledgements text.

## Deploy
```bash
cd project_page
git init && git add . && git commit -m "project page"
git branch -M main
git remote add origin git@github.com:USERNAME/REPO.git   # or USERNAME.github.io
git push -u origin main
```
Then GitHub → repo Settings → Pages → Source: `main` / root. For an org-style URL like `xxx.github.io`, name the repo `xxx.github.io` and push there.

## Local preview
```bash
python -m http.server 8000   # then open http://localhost:8000
```
