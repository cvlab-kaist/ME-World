"""Project-page teaser video for ME-World, cut from the clips already in videos/.

  /data2/envs/xmetric/bin/python make_teaser.py --out ../videos/teaser_video.mp4 [--fps 30]

Segments: title / hook / real results / synthetic results / architecture / shared environment memory /
3 agents + autoregressive / comparison / end card. Dark ground, Carlito type (metric-compatible Calibri).
"""
import argparse, os, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); V = os.path.join(HERE, "..", "videos"); A = os.path.join(HERE, "..", "assets")
W, H = 1920, 1080
BG = (14, 16, 22); FG = (240, 242, 246); MUTED = (160, 168, 184)
BLUE = (31, 79, 216); CORAL = (228, 87, 46); GREEN = (21, 128, 61); REF = (90, 220, 140)
FONT_R = os.path.join(HERE, "fonts", "Carlito-Regular.ttf"); FONT_B = os.path.join(HERE, "fonts", "Carlito-Bold.ttf")
_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT_R, size)
    return _fonts[k]


# ------------------------------------------------------------------ helpers
def read_video(path, step=1):
    cap = cv2.VideoCapture(path); fr = []; i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % step == 0: fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    assert fr, path
    return fr


def fit(img, w, h):
    """Resize keeping aspect, centred on a BG canvas of (h, w)."""
    ih, iw = img.shape[:2]; s = min(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), BG, np.uint8); y, x = (h - nh) // 2, (w - nw) // 2
    out[y:y + nh, x:x + nw] = r
    return out, (x, y, nw, nh)


def rounded(img, radius=14):
    """Round the corners of an RGB image by blending to BG."""
    h, w = img.shape[:2]
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius, fill=255)
    m = np.asarray(m)[..., None] / 255.0
    return (img * m + np.array(BG) * (1 - m)).astype(np.uint8)


def paste(canvas, img, x, y):
    h, w = img.shape[:2]; canvas[y:y + h, x:x + w] = img


def text(canvas, s, xy, size=40, color=FG, bold=False, anchor="la", maxw=None):
    """Draw text with PIL onto the numpy canvas (in place). Returns the text bbox."""
    im = Image.fromarray(canvas); d = ImageDraw.Draw(im); f = font(size, bold)
    if maxw:  # simple word wrap
        words, lines, cur = s.split(), [], ""
        for w_ in words:
            t = (cur + " " + w_).strip()
            if d.textlength(t, font=f) > maxw and cur: lines.append(cur); cur = w_
            else: cur = t
        lines.append(cur); s = "\n".join(lines)
    d.multiline_text(xy, s, font=f, fill=color, anchor=anchor if "\n" not in s else "la", spacing=size * 0.25)
    canvas[:] = np.asarray(im)


def tag(canvas, s, x, y, color, size=26):
    im = Image.fromarray(canvas); d = ImageDraw.Draw(im); f = font(size, True)
    tw = d.textlength(s, font=f); d.rounded_rectangle((x, y, x + tw + 20, y + size + 12), 6, fill=color)
    d.text((x + 10, y + 6), s, font=f, fill=(255, 255, 255)); canvas[:] = np.asarray(im)


def header(canvas, eyebrow, title, sub=None):
    text(canvas, eyebrow.upper(), (120, 64), 26, BLUE if eyebrow else FG, True)
    text(canvas, title, (120, 96), 58, FG, True)
    if sub: text(canvas, sub, (120, 170), 32, MUTED)


def fade(frames, n_in, n_out):
    """Generator applying fade from/to BG over n_in / n_out frames."""
    frames = list(frames); N = len(frames)
    for i, f in enumerate(frames):
        a = 1.0
        if i < n_in: a = i / n_in
        if i >= N - n_out: a = min(a, (N - 1 - i) / n_out)
        yield f if a >= 1 else (f * a + np.array(BG) * (1 - a)).astype(np.uint8)


def hold(frame, n):
    for _ in range(n): yield frame


def loop(frames, n):
    """n output frames cycling through `frames`."""
    for i in range(n): yield frames[i % len(frames)]


# ------------------------------------------------------------------ segments (all yield 1920x1080 RGB)
def seg_title(fps, sec=4.5):
    c = np.full((H, W, 3), BG, np.uint8)
    im = Image.fromarray(c); d = ImageDraw.Draw(im); f = font(150, True); s = "ME-World"
    tw = d.textlength(s, font=f); x = (W - tw) / 2; y = 300
    d.rounded_rectangle((x - 36, y - 10, x + tw + 36, y + 170), 18, fill=BLUE)
    d.text((x, y), s, font=f, fill=(255, 255, 255)); c = np.asarray(im).copy()
    text(c, "Multi-Agent Egocentric World Model", (W // 2, 560), 60, FG, True, "ma")
    text(c, "with Fine-Grained Embodied Interaction", (W // 2, 630), 60, FG, True, "ma")
    text(c, "Dahyun Chung · Siyoon Jin · Hyunwook Choi · Honggyu An · Junyoung Seo · Hyunsung Kim · Seung Wook Kim · Seungryong Kim",
         (W // 2, 760), 30, MUTED, False, "ma")
    text(c, "KAIST AI", (W // 2, 810), 32, MUTED, True, "ma")
    yield from hold(c, int(sec * fps))


def seg_hook(fps):
    clip = read_video(f"{V}/teaser/ours_1326e688_017304.mp4")
    n = len(clip) * 3
    for i in range(n):
        c = np.full((H, W, 3), BG, np.uint8)
        text(c, "Two agents.", (120, 330), 84, FG, True)
        text(c, "One shared world.", (120, 425), 84, BLUE, True)
        text(c, "ME-World generates every agent's first-person video jointly, so actions, the environment and interaction outcomes agree across views.",
             (120, 580), 34, MUTED, maxw=640)
        img = rounded(cv2.resize(clip[i // 3], (960, 480), interpolation=cv2.INTER_AREA))
        paste(c, img, 880, 300); tag(c, "Agent 1", 896, 316, BLUE); tag(c, "Agent 2", 896 + 480, 316, CORAL)
        text(c, "real benchmark · generated", (880, 796), 26, MUTED)
        yield c


def seg_real(fps, sec=15):
    clips = [read_video(f"{V}/teaser/{n}.mp4") for n in ["ours_1326e688_017304", "ours_1326e688_032712", "ours_1326e688_039048", "ours_1326e688_053160"]]
    cw, ch, gap = 820, 410, 24; x0 = (W - 2 * cw - gap) // 2; y0 = 220
    n = int(sec * fps)
    for i in range(n):
        c = np.full((H, W, 3), BG, np.uint8)
        header(c, "Results", "Jointly generated ego streams", "Real benchmark · two people cooking · 77 frames per clip")
        for k, clip in enumerate(clips):
            x = x0 + (k % 2) * (cw + gap); y = y0 + (k // 2) * (ch + gap)
            paste(c, rounded(cv2.resize(clip[(i // 3) % len(clip)], (cw, ch), interpolation=cv2.INTER_AREA)), x, y)
            tag(c, "Agent 1", x + 14, y + 14, BLUE); tag(c, "Agent 2", x + cw // 2 + 14, y + 14, CORAL)
        yield c


def seg_synth(fps, sec=15):
    ids = [("0000_gbw008_ixG058T006A021R004_w000_gen", "gbw008_ixG058T006A021R004_w000_thirdperson"),
           ("0001_gbw010_ixG057T000A001R006_w010_gen", "gbw010_ixG057T000A001R006_w010_thirdperson"),
           ("0003_g00703_ixG005T004A022R006_w009_gen", "g00703_ixG005T004A022R006_w009_thirdperson"),
           ("0004_g00166_ih1560_w000_gen", "g00166_ih1560_w000_thirdperson")]
    pairs = [(read_video(f"{V}/synthetic_teaser/{g}.mp4"), read_video(f"{V}/synthetic_teaser/{t}.mp4")) for g, t in ids]
    gh = 300; gw = 2 * gh; tw = gh; gap = 16; col = gw + gap + tw; cgap = 32
    x0 = (W - 2 * col - cgap) // 2; y0 = 290
    n = int(sec * fps)
    for i in range(n):
        c = np.full((H, W, 3), BG, np.uint8)
        header(c, "Results", "Distinct characters, shared scenes", "Synthetic benchmark · third-person view shown for reference only, never given to the model")
        for k, (g, t) in enumerate(pairs):
            x = x0 + (k % 2) * (col + cgap); y = y0 + (k // 2) * (gh + 24)
            paste(c, rounded(cv2.resize(g[(i // 3) % len(g)], (gw, gh), interpolation=cv2.INTER_AREA)), x, y)
            paste(c, rounded(cv2.resize(t[(i // 3) % len(t)], (tw, gh), interpolation=cv2.INTER_AREA)), x + gw + gap, y)
            tag(c, "Agent 1", x + 12, y + 12, BLUE); tag(c, "Agent 2", x + gw // 2 + 12, y + 12, CORAL)
            tag(c, "TPV", x + gw + gap + 12, y + 12, (70, 74, 84))
        yield c


def seg_arch(fps, sec=13):
    img = cv2.cvtColor(cv2.imread(f"{A}/model.png"), cv2.COLOR_BGR2RGB)
    fig, (fx, fy, fw, fh) = fit(img, 1680, 640)
    bullets = ["Joint multi-agent generation: all ego streams denoised in one token sequence",
               "Shared action conditioning: every agent's body motion rendered into each view",
               "Shared environment memory: warped history + clean anchor references shared across streams"]
    n = int(sec * fps)
    for i in range(n):
        c = np.full((H, W, 3), BG, np.uint8)
        header(c, "Method", "ME-World architecture")
        # white card behind the figure (the figure has a white ground)
        card = np.full((fh + 40, fw + 40, 3), 255, np.uint8); card[20:20 + fh, 20:20 + fw] = fig[fy:fy + fh, fx:fx + fw]
        paste(c, rounded(card, 18), (W - fw - 40) // 2, 200)
        t = i / fps
        for b, s in enumerate(bullets):
            if t > 1.0 + b * 2.5:
                a = min(1.0, (t - (1.0 + b * 2.5)) / 0.5)
                col = tuple(int(BG[j] * (1 - a) + FG[j] * a) for j in range(3))
                dot = tuple(int(BG[j] * (1 - a) + BLUE[j] * a) for j in range(3))
                cv2.circle(c, (140, 912 + b * 50), 7, dot, -1, cv2.LINE_AA)
                text(c, s, (165, 893 + b * 50), 32, col)
        yield c


def seg_memory(fps):
    clip = read_video(f"{V}/memory/58dd70af_069576_memory.mp4")
    n = len(clip) * 3
    for i in range(n):
        c = np.full((H, W, 3), BG, np.uint8)
        header(c, "Method", "Shared environment memory", "History frames warped into each view (yellow) and shared anchor references (green) ground every stream in one scene")
        img, _ = fit(clip[i // 3], 1760, 585)
        paste(c, img, (W - 1760) // 2, 270)
        yield c


def seg_multi(fps, sec3=7, secar=10):
    tp = [(read_video(f"{V}/application/tp00_iter1000_gen3.mp4"), read_video(f"{V}/application/tp00_thirdperson.mp4")),
          (read_video(f"{V}/application/tp01_iter200_gen3.mp4"), read_video(f"{V}/application/tp01_thirdperson.mp4"))]
    gh = 340; gw = 3 * gh; tw = gh; gap = 16; col = gw + gap + tw
    x0 = (W - col) // 2
    for i in range(int(sec3 * fps)):
        c = np.full((H, W, 3), BG, np.uint8)
        header(c, "Results", "Three agents, one joint sequence", "The same model generates three synchronized ego views")
        for k, (g, t) in enumerate(tp):
            y = 240 + k * (gh + 24)
            paste(c, rounded(cv2.resize(g[(i // 3) % len(g)], (gw, gh), interpolation=cv2.INTER_AREA)), x0, y)
            paste(c, rounded(cv2.resize(t[(i // 3) % len(t)], (tw, gh), interpolation=cv2.INTER_AREA)), x0 + gw + gap, y)
            tag(c, "Agent 1", x0 + 12, y + 12, BLUE); tag(c, "Agent 2", x0 + gw // 3 + 12, y + 12, CORAL)
            tag(c, "Agent 3", x0 + 2 * gw // 3 + 12, y + 12, GREEN); tag(c, "TPV", x0 + gw + gap + 12, y + 12, (70, 74, 84))
        yield c
    ar = [(read_video(f"{V}/application/AR_stitched_g00537.mp4", step=2), read_video(f"{V}/application/ar_g00537_thirdperson.mp4", step=2)),
          (read_video(f"{V}/application/AR_stitched_g01002.mp4", step=2), read_video(f"{V}/application/ar_g01002_thirdperson.mp4", step=2))]
    gh = 340; gw = 2 * gh; tw = gh; col = gw + gap + tw; x0 = (W - col) // 2
    for i in range(int(secar * fps)):
        c = np.full((H, W, 3), BG, np.uint8)
        header(c, "Results", "Autoregressive long-horizon generation", "221 frames per agent, chained chunk by chunk · shown at 2x speed")
        for k, (g, t) in enumerate(ar):
            y = 240 + k * (gh + 24)
            paste(c, rounded(cv2.resize(g[(i // 3) % len(g)], (gw, gh), interpolation=cv2.INTER_AREA)), x0, y)
            paste(c, rounded(cv2.resize(t[(i // 3) % len(t)], (tw, gh), interpolation=cv2.INTER_AREA)), x0 + gw + gap, y)
            tag(c, "Agent 1", x0 + 12, y + 12, BLUE); tag(c, "Agent 2", x0 + gw // 2 + 12, y + 12, CORAL); tag(c, "TPV", x0 + gw + gap + 12, y + 12, (70, 74, 84))
            tag(c, f"frame {(i // 3) % len(g) * 2 + 1:3d} / 221", x0 + gw - 190, y + gh - 52, (40, 44, 54), 22)
        yield c


def seg_compare(fps):
    ex = "1326e688_057336"
    methods = [("gt", "Ground truth"), ("ours", "ME-World (Ours)"), ("gen3c", "GEN3C"), ("anyview", "AnyView-DVS"),
               ("egosim", "EgoSim"), ("jointcontrol", "JointControlVideo"), ("dreamx", "DreamX-World"), ("lingbot", "LingBot-World")]
    clips = [read_video(f"{V}/comparison/{m}_{ex}.mp4") for m, _ in methods]
    cw, ch, gap = 440, 220, 22; x0 = (W - 4 * cw - 3 * gap) // 2; y0 = 260
    n = max(len(c_) for c_ in clips) * 3
    for i in range(n):
        c = np.full((H, W, 3), BG, np.uint8)
        header(c, "Comparison", "Against existing world models", "Multi-view baselines receive Agent 1's ground-truth stream; the others generate each view independently")
        for k, (clip, (m, name)) in enumerate(zip(clips, methods)):
            x = x0 + (k % 4) * (cw + gap); y = y0 + (k // 4) * (ch + 90)
            paste(c, rounded(cv2.resize(clip[(i // 3) % len(clip)], (cw, ch), interpolation=cv2.INTER_AREA), 10), x, y)
            text(c, name, (x, y + ch + 12), 30, BLUE if m == "ours" else FG, m in ("ours", "gt"))
        yield c


def seg_end(fps, sec=4.5):
    c = np.full((H, W, 3), BG, np.uint8)
    im = Image.fromarray(c); d = ImageDraw.Draw(im); f = font(120, True); s = "ME-World"
    tw = d.textlength(s, font=f); x = (W - tw) / 2; y = 360
    d.rounded_rectangle((x - 30, y - 8, x + tw + 30, y + 136), 16, fill=BLUE); d.text((x, y), s, font=f, fill=(255, 255, 255))
    c = np.asarray(im).copy()
    text(c, "Multi-Agent Egocentric World Model with Fine-Grained Embodied Interaction", (W // 2, 560), 40, FG, True, "ma")
    text(c, "dhyun22.github.io/ME-World", (W // 2, 660), 44, MUTED, False, "ma")   # TODO: switch to the cvlab-kaist URL
    yield from hold(c, int(sec * fps))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--only", default=None, help="render a single segment by name (debug)"); a = ap.parse_args()
    fps = a.fps; fi, fo = int(0.4 * fps), int(0.4 * fps)
    segs = [("title", seg_title), ("hook", seg_hook), ("real", seg_real), ("synth", seg_synth), ("arch", seg_arch),
            ("memory", seg_memory), ("multi", seg_multi), ("compare", seg_compare), ("end", seg_end)]
    if a.only: segs = [s for s in segs if s[0] == a.only]
    pp = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
                           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out],
                          stdin=subprocess.PIPE)
    total = 0
    for name, fn in segs:
        n = 0
        for fr in fade(fn(fps), fi, fo):
            pp.stdin.write(np.ascontiguousarray(fr).tobytes()); n += 1
        total += n; print(f"{name:8s} {n / fps:5.1f}s", flush=True)
    pp.stdin.close(); pp.wait(); print(f"total {total / fps:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
