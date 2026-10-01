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




CLIPS = "/data4/comind_dataset/clips"
CID = "58dd70af_069576"


def jdec(b): return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def border(img, color, t=4):
    out = img.copy(); cv2.rectangle(out, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, t); return out


def ease(a):
    a = min(1.0, max(0.0, a)); return a * a * (3 - 2 * a)


def lerp_rect(r0, r1, a):
    return tuple(int(round(r0[i] + (r1[i] - r0[i]) * a)) for i in range(4))


def seg_memory(fps):
    """Shared environment memory explainer on one real clip: pooled history -> geometric memory -> anchor memory -> 3D."""
    z = np.load(f"{CLIPS}/{CID[:8]}/{CID}.npz", allow_pickle=True)
    rf = np.load(f"{CLIPS}/{CID[:8]}/{CID}.refs.npz", allow_pickle=True)
    hs = np.load(os.path.join(HERE, "cache", f"history_{CID}.npz"))
    gen = read_video(f"{V}/memory/{CID}_gen.mp4"); viz = read_video(f"{V}/memory/{CID}_memory.mp4")
    T = 77
    A = {"L": dict(name="Agent 1", col=BLUE), "H": dict(name="Agent 2", col=CORAL)}
    for s_, d in A.items():
        d["warped"] = [jdec(b) for b in z[f"{s_}_warped"]]; d["srcf"] = z[f"{s_}_srcf"]; d["srcview"] = z[f"{s_}_srcview"]
        d["hist"] = hs[f"{s_}_img"]; d["hist_vrs"] = list(hs[f"{s_}_img_vrs"]); d["pool"] = hs[f"{s_}_hist_vrs"]
        d["gen"] = [g[:, :480] if s_ == "L" else g[:, 480:] for g in gen]
    def hist_img(s_, vrs):
        d = A[s_]; return d["hist"][d["hist_vrs"].index(int(vrs))]
    refs = rf["reference_frames"]; ref_view = rf["reference_src_view"]
    TH = 80; gapx = 8; pool_x = 120; pool_y = {"L": 330, "H": 440}
    def draw_pool(c, a=1.0, n_show=None):
        for s_, d in A.items():
            text(c, f"{d['name']} history · 200 frames", (pool_x, pool_y[s_] - 30), 24, d["col"], True)
            vs = list(d["pool"])[::2]
            for k, v in enumerate(vs):
                if n_show is not None and k >= n_show: break
                im = cv2.resize(hist_img(s_, v), (TH, TH), interpolation=cv2.INTER_AREA)
                if a < 1: im = (im * a + np.array(BG) * (1 - a)).astype(np.uint8)
                paste(c, rounded(im, 8), pool_x + k * (TH + gapx), pool_y[s_])
        return pool_x + 10 * (TH + gapx) - gapx
    def head(c, step, title, sub):
        header(c, "Method", "Shared environment memory", None)
        text(c, step, (120, 170), 30, BLUE, True); text(c, title, (120 + 36, 170), 30, FG, True)
        text(c, sub, (120, 214), 26, MUTED)
    # ---- P1: the pool appears (4 s)
    n1 = int(4 * fps)
    for i in range(n1):
        c = np.full((H, W, 3), BG, np.uint8)
        head(c, "", "Observation history, pooled across agents", "Every agent's past frames go into one shared memory; each agent sees only part of the room.")
        k = int(ease(i / (n1 * 0.7)) * 10 + 0.999)
        draw_pool(c, 1.0, k)
        yield c
    # ---- P2: stream-aligned geometric memory (77 frames)
    SX, SW = 1040, 200; RX1, RX2, RW = 1300, 1590, 280; RY = {"L": 300, "H": 630}
    for i in range(T * 3):
        t = i // 3; c = np.full((H, W, 3), BG, np.uint8)
        head(c, "1", "Stream-aligned geometric memory", "The history frame that best covers each target frame is warped into that agent's view.")
        right = draw_pool(c)
        for s_, d in A.items():
            sv = int(d["srcview"][t]); src_s = s_ if sv == 0 else ("H" if s_ == "L" else "L")
            src = hist_img(src_s, d["srcf"][t]); y = RY[s_]
            # source box, linked to the pool row it came from
            sx, sy = SX, y + 40
            cv2.line(c, (right + 6, pool_y[src_s] + TH // 2), (sx, sy + SW // 2), (255, 220, 80), 2, cv2.LINE_AA)
            paste(c, rounded(border(cv2.resize(src, (SW, SW), interpolation=cv2.INTER_AREA), (255, 220, 80), 5), 8), sx, sy)
            text(c, "source", (sx, sy - 30), 22, (255, 220, 80), True)
            tag(c, A[src_s]["name"], sx + 8, sy + 8, A[src_s]["col"], 20)
            # arrow -> warped -> generated
            cv2.arrowedLine(c, (sx + SW + 10, sy + SW // 2), (RX1 - 10, sy + SW // 2), (255, 220, 80), 2, cv2.LINE_AA, tipLength=0.2)
            paste(c, rounded(cv2.resize(d["warped"][t], (RW, RW), interpolation=cv2.INTER_AREA), 8), RX1, y)
            paste(c, rounded(cv2.resize(d["gen"][t], (RW, RW), interpolation=cv2.INTER_AREA), 8), RX2, y)
            text(c, f"{d['name']} · warped", (RX1, y - 30), 22, d["col"], True)
            text(c, "generated", (RX2, y - 30), 22, FG, True)
            tag(c, f"t = {t + 1:02d}", RX2 + RW - 92, y + 8, (40, 44, 54), 20)
        yield c
    # ---- P3: cross-stream anchor memory (6 s)
    n3 = int(6 * fps); RT = 150; ry = 600; rx0 = 120
    for i in range(n3):
        c = np.full((H, W, 3), BG, np.uint8)
        head(c, "2", "Cross-stream anchor memory", "Greedy set cover over the pooled history picks eight clean, unwarped anchors that every stream attends to.")
        draw_pool(c, 0.45)
        k = int(ease(i / (n3 * 0.5)) * 8 + 0.999)
        for j in range(min(k, 8)):
            im = cv2.resize(refs[j], (RT, RT), interpolation=cv2.INTER_AREA)
            paste(c, rounded(border(im, REF, 5), 8), rx0 + j * (RT + 12), ry)
            tag(c, A["L" if ref_view[j] == 0 else "H"]["name"], rx0 + j * (RT + 12) + 8, ry + 8, A["L" if ref_view[j] == 0 else "H"]["col"], 18)
        text(c, "8 shared anchors · appended to the joint token sequence", (rx0, ry - 32), 24, REF, True)
        # token strip
        if i > n3 * 0.55:
            a = ease((i - n3 * 0.55) / (n3 * 0.2)); yb = 830
            text(c, "joint sequence", (rx0, yb - 36), 24, MUTED, True)
            bx = rx0
            for lab, col, n in (("Agent 1 tokens", BLUE, 6), ("Agent 2 tokens", CORAL, 6), ("anchor tokens", REF, 8)):
                for q in range(n):
                    cc = tuple(int(BG[j] * (1 - a) + col[j] * a) for j in range(3))
                    cv2.rectangle(c, (bx, yb), (bx + 44, yb + 44), cc, -1, cv2.LINE_AA); bx += 52
                text(c, lab, (bx - n * 52, yb + 56), 22, col, True); bx += 40
            text(c, "self-attention runs over all of it: streams see each other and the same anchors", (rx0, yb + 110), 24, MUTED)
        yield c
    # ---- P4: 3D view (7.7 s)
    for fr in viz:
        c = np.full((H, W, 3), BG, np.uint8)
        head(c, "3", "Both memories in the shared 3D world", "Yellow: history frame warped into the current view · green: the eight anchors at their camera poses")
        img, _ = fit(fr, 1760, 585); paste(c, img, (W - 1760) // 2, 300)
        for _ in range(3): yield c


# architecture figure regions in model.png pixel coords (x0, y0, x1, y1), 2400 x 1113
ARCH_STEPS = [
    ((28, 90, 724, 750), "Inputs", "Each agent's head motion and body motion, plus the observation history shared by all agents"),
    ((840, 105, 1470, 757), "Per-agent conditions", "Viewing rays, every agent's pose, the warped history frame and its visibility mask, concatenated per stream"),
    ((1455, 85, 1740, 757), "Cross-stream anchor memory", "Clean reference frames with their own rays and poses, shared by all streams"),
    ((850, 820, 1740, 1075), "Tokenization + agent embedding", "Streams share one positional frame and are told apart by a learned agent embedding"),
    ((1755, 75, 2390, 1085), "Joint denoising", "Self-attention across all streams and anchors; per-agent text cross-attention; synchronized videos out"),
]


def seg_arch(fps, step_sec=4.0, trans=0.6):
    img = cv2.cvtColor(cv2.imread(f"{A}/model.png"), cv2.COLOR_BGR2RGB)
    fh = 700; sc = fh / img.shape[0]; fw = int(img.shape[1] * sc)
    fig = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA)
    fx0, fy0 = (W - fw) // 2 + 20, 190
    card = np.full((fh + 40, fw + 40, 3), 255, np.uint8); card[20:20 + fh, 20:20 + fw] = fig; card = rounded(card, 18)
    def rect(r): return tuple(int(v * sc) for v in r)
    rects = [rect(r) for r, _, _ in ARCH_STEPS]
    n_step = int(step_sec * fps); n_tr = int(trans * fps); intro = int(1.5 * fps)
    total = intro + len(ARCH_STEPS) * n_step
    for i in range(total):
        c = np.full((H, W, 3), BG, np.uint8)
        header(c, "Method", "ME-World architecture")
        paste(c, card, fx0 - 20, fy0 - 20)
        if i >= intro:
            k = (i - intro) // n_step; j = (i - intro) % n_step
            r1 = rects[k]; r0 = rects[k - 1] if k > 0 else r1
            r = lerp_rect(r0, r1, ease(j / n_tr)) if j < n_tr else r1
            # dim everything but the spotlight
            x0, y0, x1, y1 = r; ov = c.copy()
            cv2.rectangle(ov, (fx0 - 20, fy0 - 20), (fx0 + fw + 20, fy0 + fh + 20), (255, 255, 255), -1)
            c[:] = (c * 0.55 + ov * 0.0 + np.array(BG) * 0.45).astype(np.uint8)
            c[fy0 + y0:fy0 + y1, fx0 + x0:fx0 + x1] = fig[y0:y1, x0:x1]
            cv2.rectangle(c, (fx0 + x0 - 3, fy0 + y0 - 3), (fx0 + x1 + 3, fy0 + y1 + 3), BLUE, 4, cv2.LINE_AA)
            # step pills along the bottom + callout
            px = 120
            for q, (_, name, _) in enumerate(ARCH_STEPS):
                on = q == k
                im = Image.fromarray(c); d = ImageDraw.Draw(im); f = font(24, on); tw = d.textlength(name, font=f)
                d.rounded_rectangle((px, 995, px + tw + 28, 1035), 20, fill=BLUE if on else (30, 34, 44)); d.text((px + 14, 1002), name, font=f, fill=(255, 255, 255) if on else MUTED); c[:] = np.asarray(im)
                px += int(tw) + 44
            a = ease(j / n_tr) if j < n_tr else 1.0
            col = tuple(int(BG[q] * (1 - a) + FG[q] * a) for q in range(3))
            text(c, ARCH_STEPS[k][2], (120, 940), 30, col)
        yield c


def seg_compare(fps):
    ex = "1326e688_057336"
    groups = [("", [("gt", "Ground truth"), ("ours", "ME-World (Ours)")]),
              ("Multi-view video generation", [("gen3c", "GEN3C"), ("anyview", "AnyView-DVS")]),
              ("Egocentric world models", [("egosim", "EgoSim"), ("jointcontrol", "JointControlVideo")]),
              ("General world models", [("dreamx", "DreamX-World"), ("lingbot", "LingBot-World")])]
    clips = {m: read_video(f"{V}/comparison/{m}_{ex}.mp4") for _, ms in groups for m, _ in ms}
    cw, ch, gap = 440, 220, 22; x0 = (W - 4 * cw - 3 * gap) // 2; y0 = 300
    n = 77 * 3
    for i in range(n):
        c = np.full((H, W, 3), BG, np.uint8)
        header(c, "Comparison", "Against three families of baselines", "Multi-view models receive Agent 1's ground-truth stream; egocentric and general world models generate each view independently")
        for g, (gname, ms) in enumerate(groups):
            gx = x0 + (g % 2) * 2 * (cw + gap); gy = y0 + (g // 2) * (ch + 120)
            if gname:
                text(c, gname.upper(), (gx, gy - 34), 22, BLUE, True)
                cv2.line(c, (gx, gy - 8), (gx + 2 * cw + gap, gy - 8), BLUE, 2)
            for k, (m, name) in enumerate(ms):
                clip = clips[m]; x = gx + k * (cw + gap)
                paste(c, rounded(cv2.resize(clip[(i // 3) % len(clip)], (cw, ch), interpolation=cv2.INTER_AREA), 10), x, gy)
                text(c, name, (x, gy + ch + 10), 28, BLUE if m == "ours" else FG, m in ("ours", "gt"))
        yield c


def seg_ablation(fps):
    items = [("cosmos_zeroshot", "(1) Cosmos-Predict2.5, zero-shot"), ("cosmos_finetune", "(2) fine-tuned"),
             ("single_indep", "(3) + self hand pose + history warp"), ("jointgen_noshared", "(4) (3) + joint generation"),
             ("wo_sharedenv", "(5) (4) + shared action"), ("wo_sharedaction", "(6) (4) + shared environment"),
             ("wo_jointgen", "(7) (3) + shared action + env, no joint"), ("ours_full", "(8) full model (Ours)")]
    clips = [read_video(f"{V}/ablation/{m}.mp4") for m, _ in items]
    cw, ch, gap = 440, 220, 22; x0 = (W - 4 * cw - 3 * gap) // 2; y0 = 260
    for i in range(77 * 3):
        c = np.full((H, W, 3), BG, np.uint8)
        header(c, "Ablation", "Each component removes a failure mode", "Joint generation couples the streams · shared action aligns the interaction · shared environment anchors the room")
        for k, (clip, (m, name)) in enumerate(zip(clips, items)):
            x = x0 + (k % 4) * (cw + gap); y = y0 + (k // 4) * (ch + 90)
            paste(c, rounded(cv2.resize(clip[(i // 3) % len(clip)], (cw, ch), interpolation=cv2.INTER_AREA), 10), x, y)
            text(c, name, (x, y + ch + 10), 26, BLUE if m == "ours_full" else FG, m == "ours_full")
        yield c



# ------------------------------------------------------------------ shared action conditioning (synthetic g00537)
SEQ_DIR = "/data4/vroid_batch/groups/g00537/render_root/ixG005T000A001R007"
CLIP_DIR = "/data4/vroid_batch/groups/g00537/train/clips/g00537_ixG005T000A001R007_w000"


class LookAt:
    def __init__(self, eye, target, up, fov_deg, w, h):
        f = target - eye; f /= np.linalg.norm(f); r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
        Rw = np.stack([r, -u, f], 0); self.w2c = np.eye(4); self.w2c[:3, :3] = Rw; self.w2c[:3, 3] = -Rw @ eye
        fl = 0.5 * w / np.tan(np.radians(fov_deg) / 2); self.K = np.array([[fl, 0, w / 2], [0, fl, h / 2], [0, 0, 1]]); self.w, self.h = w, h
    def project(self, pts):
        c = pts @ self.w2c[:3, :3].T + self.w2c[:3, 3]; z = np.maximum(c[:, 2], 1e-6)
        return self.K[0, 0] * c[:, 0] / z + self.K[0, 2], self.K[1, 1] * c[:, 1] / z + self.K[1, 2], c[:, 2]
    def pt(self, p):
        u, v, _ = self.project(np.asarray(p, np.float64)[None])
        return int(np.clip(round(u[0]), -1e5, 1e5)), int(np.clip(round(v[0]), -1e5, 1e5))


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = np.zeros_like(V)
    for k in range(3): np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def splat(img, vc, pts, color, size=3, normals=None, eye=None, light=None):
    """Splat body vertices. With normals: back-face culling + Lambert shading, so only the visible side shows."""
    keep = np.ones(len(pts), bool); sh = None
    if normals is not None:
        view = pts - eye; view /= np.linalg.norm(view, axis=1, keepdims=True)
        keep = (normals * view).sum(1) < 0.05
        lam = np.clip((normals * light).sum(1), 0, 1); sh = 0.4 + 0.6 * lam
    u, v, z = vc.project(pts[keep]); sh = sh[keep] if sh is not None else None
    ok = (z > 0.05) & (u >= 0) & (u < vc.w - size) & (v >= 0) & (v < vc.h - size)
    u, v, z = u[ok].astype(int), v[ok].astype(int), z[ok]
    order = np.argsort(-z); u, v = u[order], v[order]
    if sh is not None:
        cols = (np.array(color)[None] * sh[ok][order][:, None]).astype(np.uint8)
    else:
        cols = np.tile(np.array(color, np.uint8), (len(u), 1))
    for dy in range(size):
        for dx in range(size):
            img[v + dy, u + dx] = cols


def frustum3d(c2w, K, size):
    corners = np.array([[0, 0, 1], [504, 0, 1], [504, 504, 1], [0, 504, 1]], float)
    cam = (np.linalg.inv(K) @ corners.T).T * size
    pts = np.vstack([np.zeros(3), cam]); return pts @ c2w[:3, :3].T + c2w[:3, 3]


def draw_frustum3d(img, vc, c2w, K, color, size=0.25, thick=2):
    ps = [vc.pt(p) for p in frustum3d(c2w, K, size)]
    for i in range(1, 5):
        cv2.line(img, ps[0], ps[i], color, thick, cv2.LINE_AA); cv2.line(img, ps[i], ps[1 + i % 4], color, thick, cv2.LINE_AA)
    cv2.circle(img, ps[0], 5, color, -1, cv2.LINE_AA); return ps[0]


def seg_action(fps):
    import json
    seq = np.load(f"{SEQ_DIR}/cache/sequence.npz"); meta = json.load(open(f"{SEQ_DIR}/cache/clip_meta.json"))
    K = np.array(meta["camera_intrinsics"]["K"], np.float64); F = np.diag([1.0, -1.0, -1.0, 1.0])
    T = 77
    V1 = seq["p1_vertices"][:T]; V2 = seq["p2_vertices"][:T]; F1 = seq["p1_faces"]; F2 = seq["p2_faces"]           # (T, ~5k, 3) world, z up
    w2c = {"a": np.array([F @ m for m in seq["ego_a_t_camera_world"][:T]]), "b": np.array([F @ m for m in seq["ego_b_t_camera_world"][:T]])}
    c2w = {k: np.linalg.inv(v) for k, v in w2c.items()}
    pose = {k: read_video(f"{CLIP_DIR}__ego_{k}/pose_person.mp4") for k in "ab"}
    gen = read_video(f"{V}/application/AR_stitched_g00537.mp4")[:T]
    gen = {"a": [g[:, :480] for g in gen], "b": [g[:, 480:] for g in gen]}
    # virtual camera: look at the pair from the side, slightly above
    allv = np.concatenate([V1.reshape(-1, 3), V2.reshape(-1, 3)]); centre = allv.mean(0); centre[2] = 0.9
    d = (V2[:, :, :2].mean((0, 1)) - V1[:, :, :2].mean((0, 1))); d /= np.linalg.norm(d); side = np.array([-d[1], d[0], 0.0])
    eye = centre + side * 2.3 + np.array([0, 0, 0.9]); PW, PH = 980, 700; px, py = 120, 250
    vc = LookAt(eye, centre, np.array([0, 0, 1.0]), 50, PW, PH)
    # floor grid for grounding
    grid = np.full((PH, PW, 3), BG, np.uint8); g0 = centre.copy(); g0[2] = 0
    for k in range(-6, 7):
        for axis in (0, 1):
            a_, b_ = g0.copy(), g0.copy(); a_[axis] += k * 0.5; b_[axis] += k * 0.5; a_[1 - axis] -= 3; b_[1 - axis] += 3
            u, v, z = vc.project(np.stack([a_, b_]))
            if (z > 0.1).all() and np.abs(u).max() < 1e5 and np.abs(v).max() < 1e5:
                cv2.line(grid, (int(u[0]), int(v[0])), (int(u[1]), int(v[1])), (34, 38, 50), 1, cv2.LINE_AA)
    AG = {"a": ("Agent 1", BLUE), "b": ("Agent 2", CORAL)}
    RX1, RX2, RW = 1200, 1510, 290; RY = {"a": 250, "b": 610}
    phases = [(4.0, None), (8.0, "a"), (8.0, "b"), (4.0, "both")]
    total = int(sum(p[0] for p in phases) * fps); n = 0
    for i in range(total):
        t = i // 3 % T; tt = i / fps
        # phase lookup
        acc = 0; ph = None; local = 0
        for dur, key in phases:
            if tt < acc + dur: ph = key; local = tt - acc; break
            acc += dur
        c = np.full((H, W, 3), BG, np.uint8)
        header(c, "Method", "Shared action conditioning")
        panel = grid.copy()
        light = np.array([-0.4, -0.5, 1.0]); light /= np.linalg.norm(light)
        splat(panel, vc, V1[t], BLUE, 3, vertex_normals(V1[t], F1), eye, light)
        splat(panel, vc, V2[t], CORAL, 3, vertex_normals(V2[t], F2), eye, light)
        heads = {k: draw_frustum3d(panel, vc, c2w[k][t], K, AG[k][1], 0.16, 3 if ph in (k, "both") else 2) for k in "ab"}
        for k in "ab":
            tag(panel, AG[k][0], heads[k][0] - 40, heads[k][1] - 62, AG[k][1], 20)
        text(panel, "shared world · both agents' body motion + head cameras", (16, PH - 40), 22, MUTED)
        paste(c, rounded(panel, 14), px, py)
        if ph is None:
            text(c, "Both agents' body motion lives in one shared world, together with their head cameras.", (120, 200), 30, MUTED)
        else:
            text(c, "All agents' motion is projected into each agent's own camera.", (120, 200), 30, MUTED)
        show = {"a": ph in ("a", "both") or ph == "b", "b": ph in ("b", "both")}
        for k in "ab":
            if not show[k]: continue
            y = RY[k]; name, col = AG[k]
            a_in = ease(local / 0.6) if ph == k else 1.0
            # projection link from that agent's head camera to its row
            hx, hy = heads[k]; hx += px; hy += py
            cv2.line(c, (hx, hy), (RX1 - 14, y + RW // 2), col, 2, cv2.LINE_AA)
            pimg = cv2.resize(pose[k][t], (RW, RW), interpolation=cv2.INTER_AREA); gimg = cv2.resize(gen[k][t], (RW, RW), interpolation=cv2.INTER_AREA)
            if a_in < 1:
                pimg = (pimg * a_in + np.array(BG) * (1 - a_in)).astype(np.uint8); gimg = (gimg * a_in + np.array(BG) * (1 - a_in)).astype(np.uint8)
            paste(c, rounded(border(pimg, col, 4), 10), RX1, y); paste(c, rounded(gimg, 10), RX2, y)
            cv2.arrowedLine(c, (RX1 + RW + 6, y + RW // 2), (RX2 - 6, y + RW // 2), col, 2, cv2.LINE_AA, tipLength=0.3)
            text(c, f"{name} view · pose condition", (RX1, y - 30), 22, col, True)
            text(c, "generated", (RX2, y - 30), 22, FG, True)
            text(c, "own hands + partner body · a fixed palette per identity", (RX1, y + RW + 10), 20, MUTED)
        if ph == "both":
            text(c, "One motion, two views: the hand reaching into Agent 1's frame is the hand leaving Agent 2's.", (120, 985), 28, FG)
        yield c

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--only", default=None, help="render a single segment by name (debug)"); a = ap.parse_args()
    fps = a.fps; fi, fo = int(0.4 * fps), int(0.4 * fps)
    segs = [("title", seg_title), ("hook", seg_hook), ("real", seg_real), ("synth", seg_synth), ("arch", seg_arch), ("action", seg_action),
            ("memory", seg_memory), ("multi", seg_multi), ("compare", seg_compare), ("ablation", seg_ablation)]
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
