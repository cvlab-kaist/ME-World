"""Build a PowerPoint deck from the teaser segments: one headless segment video per slide, titles as editable text.

  /data2/envs/xmetric/bin/python make_slides.py --out ../ME-World_slides.pptx
Expects pptx_build/raw_<segment>.mp4 rendered by `make_teaser.py --headless --only <segment>`.
"""
import argparse, os, subprocess
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(HERE, "pptx_build")
BG = RGBColor(0x0E, 0x10, 0x16); FG = RGBColor(0xF0, 0xF2, 0xF6); MUTED = RGBColor(0xA0, 0xA8, 0xB8); BLUE = RGBColor(0x1F, 0x4F, 0xD8)
FONT = "Calibri"
CROP_TOP = 160                       # px of the 1920x1080 frame given up to the slide title

SLIDES = [
    ("hook", "Overview", "Two agents, one shared world",
     "ME-World generates every agent's first-person video jointly, so actions, the environment and interaction outcomes agree across views."),
    ("real", "Results", "Jointly generated ego streams on the real benchmark",
     "Two people cooking together, 77 frames per clip. Each tile shows Agent 1 (left) and Agent 2 (right) at the same instant."),
    ("synth", "Results", "Distinct characters, shared scenes (synthetic benchmark)",
     "Each agent is a different rigged avatar. The third-person view is shown for reference only and is never given to the model."),
    ("arch", "Method", "ME-World architecture",
     "Inputs, per-agent conditions, cross-stream anchor memory, tokenization with agent embeddings, and joint denoising."),
    ("action", "Method", "Shared action conditioning",
     "All agents' body motion lives in one shared world and is projected into each agent's own camera as that stream's pose condition."),
    ("memory", "Method", "Shared environment memory",
     "Pooled observation history; the best-covering frame is warped into each view (geometric memory); eight clean anchors are appended for every stream (anchor memory)."),
    ("multi", "Results", "Three agents and long-horizon generation",
     "The same model generates three synchronized ego views, and chains chunks for long-horizon two-agent sequences (trained with teacher forcing)."),
    ("compare", "Comparison", "Against three families of baselines",
     "Multi-view video generation (GEN3C, AnyView-DVS), egocentric world models (EgoSim, JointControlVideo), general world models (DreamX-World, LingBot-World)."),
    ("compare_multi", "Comparison", "Against adapted multi-agent world models",
     "MultiWorld, Gamma-World, MetaWorld and Solaris, each re-implemented with its own mechanism on our backbone, data and action conditions."),
    ("ablation", "Ablation", "Each component removes a failure mode",
     "Variants (1)–(8) of Table 2: joint generation couples the streams, shared action aligns the interaction, shared environment anchors the room."),
]


def prep_video(seg):
    raw = f"{B}/raw_{seg}.mp4"; out = f"{B}/seg_{seg}.mp4"; poster = f"{B}/seg_{seg}.jpg"
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-vf", f"crop=1920:{1080 - CROP_TOP}:0:{CROP_TOP},scale=1600:-2",
                        "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "2", "-i", out, "-frames:v", "1", "-q:v", "3", poster], check=True)
    return out, poster


def textbox(slide, x, y, w, h, s, size, color, bold=False, align=PP_ALIGN.LEFT, caps=False, spacing=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = align; r = p.add_run(); r.text = s.upper() if caps else s
    f = r.font; f.name = FONT; f.size = Pt(size); f.bold = bold; f.color.rgb = color
    if spacing is not None:
        rPr = r._r.get_or_add_rPr(); rPr.set("spc", str(spacing))
    return tb


def dark(slide):
    fill = slide.background.fill; fill.solid(); fill.fore_color.rgb = BG


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---- title slide
    s = prs.slides.add_slide(blank); dark(s)
    box = s.shapes.add_shape(1, Inches(4.4), Inches(1.7), Inches(4.5), Inches(1.15))   # rounded rect wordmark
    box.fill.solid(); box.fill.fore_color.rgb = BLUE; box.line.fill.background()
    tf = box.text_frame; tf.margin_left = tf.margin_right = 0; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = "ME-World"; r.font.name = FONT; r.font.size = Pt(48); r.font.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)
    textbox(s, 1, 3.15, 11.333, 1.3, "Multi-Agent Egocentric World Model\nwith Fine-Grained Embodied Interaction", 30, FG, True, PP_ALIGN.CENTER)
    textbox(s, 1, 4.75, 11.333, 0.5, "Dahyun Chung · Siyoon Jin · Hyunwook Choi · Honggyu An · Junyoung Seo · Hyunsung Kim · Seung Wook Kim · Seungryong Kim", 15, MUTED, False, PP_ALIGN.CENTER)
    textbox(s, 1, 5.2, 11.333, 0.4, "KAIST AI", 16, MUTED, True, PP_ALIGN.CENTER)
    textbox(s, 1, 6.2, 11.333, 0.4, "arXiv 2610.12299  ·  cvlab-kaist.github.io/ME-World", 14, MUTED, False, PP_ALIGN.CENTER)
    s.notes_slide.notes_text_frame.text = "Title. Project page: https://cvlab-kaist.github.io/ME-World/  Paper: https://arxiv.org/abs/2610.12299"

    # ---- one video slide per segment
    VW = 12.133; VH = VW * (1080 - CROP_TOP) / 1920
    for seg, eyebrow, title, note in SLIDES:
        video, poster = prep_video(seg)
        s = prs.slides.add_slide(blank); dark(s)
        textbox(s, 0.6, 0.32, 12, 0.3, eyebrow, 12, BLUE, True, caps=True, spacing=200)
        textbox(s, 0.6, 0.58, 12, 0.6, title, 26, FG, True)
        s.shapes.add_movie(video, Inches(0.6), Inches(1.35), Inches(VW), Inches(VH), poster_frame_image=poster, mime_type="video/mp4")
        s.notes_slide.notes_text_frame.text = note

    # ---- closing slide
    s = prs.slides.add_slide(blank); dark(s)
    textbox(s, 1, 2.6, 11.333, 0.8, "ME-World", 40, FG, True, PP_ALIGN.CENTER)
    textbox(s, 1, 3.5, 11.333, 0.5, "Project page  cvlab-kaist.github.io/ME-World", 18, MUTED, False, PP_ALIGN.CENTER)
    textbox(s, 1, 3.95, 11.333, 0.5, "Paper  arxiv.org/abs/2610.12299", 18, MUTED, False, PP_ALIGN.CENTER)
    textbox(s, 1, 4.4, 11.333, 0.5, "Code  github.com/cvlab-kaist/ME-World", 18, MUTED, False, PP_ALIGN.CENTER)

    prs.save(a.out); print("saved", a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
