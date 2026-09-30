"""Shared environment memory visualization for one CoMind clip.

Left  : per agent, [warped (stream-aligned geometric memory) | pose (shared action) | generated]
Right : 3D lifting of the scene (DA3 depth with GT extrinsics), both agents' camera trajectories,
        the current warp source camera (geometric memory) and the 8 shared anchor references
        (cross-stream anchor memory) placed at their camera poses.

  CUDA_VISIBLE_DEVICES=0 /data2/envs/xmetric/bin/python memory_viz.py --cid 58dd70af_069576 \
      --gen /data/model_output/our_comind_refpose_14000_infer_subset500/1212_58dd70af_069576_gen.mp4 \
      --traj traj.npz --out out.mp4 [--still 30]
"""
import argparse, os, sys, subprocess
import numpy as np, cv2

sys.path.insert(0, "/home/ubuntu/dahyun/comind_eval")
CLIPS = "/data4/comind_dataset/clips"
R = 504; P = 200; T = 77
CELL = 240                        # left panel cell size -> 3 x 2 cells = 720 x 480
W3, H3 = 720, 480                 # right panel
BG = (14, 16, 22)                 # dark ground (RGB)
COL_A = (70, 130, 255); COL_B = (255, 110, 70); COL_REF = (90, 220, 140); COL_SRC = (255, 220, 80)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def jdec(b): return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
def ddec(b): return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0


def read_video(path):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(fr)


# ---------------------------------------------------------------- DA3 depth (cached)
def da3_depths(cache, frames, w2c, K):
    if os.path.isfile(cache):
        return np.load(cache)["depth"]
    import torch
    from da3_util import load_da3, da3_depth_frames
    da3 = load_da3()
    d = da3_depth_frames(da3, frames, w2c, K, chunk=16)
    np.savez(cache, depth=d); del da3; torch.cuda.empty_cache()
    return d


# ---------------------------------------------------------------- geometry
def backproject(depth, rgb, K, w2c, valid, stride=2, zmax=6.0):
    ys, xs = np.mgrid[0:R:stride, 0:R:stride]
    z = depth[ys, xs]; ok = (z > 0.15) & (z < zmax) & valid[ys, xs]
    x = (xs + 0.5 - K[0, 2]) / K[0, 0] * z; y = (ys + 0.5 - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], -1)[ok]; c2w = np.linalg.inv(w2c)
    return cam @ c2w[:3, :3].T + c2w[:3, 3], rgb[ys, xs][ok]


def clean_cloud(pts, cols, voxel=0.008):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    pc = pc.voxel_down_sample(voxel)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)
    return np.asarray(pc.points), (np.asarray(pc.colors) * 255).astype(np.uint8)


class VirtualCam:
    """Look-at pinhole camera used to draw the 3D panel."""
    def __init__(self, eye, target, up, fov_deg=58.0, w=W3, h=H3):
        f = target - eye; f /= np.linalg.norm(f)
        r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
        Rw = np.stack([r, -u, f], 0)              # rows: cam x (right), cam y (down), cam z (forward)
        self.w2c = np.eye(4); self.w2c[:3, :3] = Rw; self.w2c[:3, 3] = -Rw @ eye
        fl = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
        self.K = np.array([[fl, 0, w / 2], [0, fl, h / 2], [0, 0, 1]]); self.w, self.h = w, h

    def project(self, pts):
        c = pts @ self.w2c[:3, :3].T + self.w2c[:3, 3]
        z = c[:, 2]; u = self.K[0, 0] * c[:, 0] / np.maximum(z, 1e-6) + self.K[0, 2]
        v = self.K[1, 1] * c[:, 1] / np.maximum(z, 1e-6) + self.K[1, 2]
        return u, v, z

    def render_cloud(self, pts, cols, size=2):
        u, v, z = self.project(pts)
        ok = (z > 0.05) & (u >= 0) & (u < self.w - size) & (v >= 0) & (v < self.h - size)
        u, v, z, cols = u[ok].astype(int), v[ok].astype(int), z[ok], cols[ok]
        order = np.argsort(-z)                    # far -> near, near overwrites
        img = np.full((self.h, self.w, 3), BG, np.uint8)
        for dy in range(size):
            for dx in range(size):
                img[v[order] + dy, u[order] + dx] = cols[order]
        return img

    def pt(self, p):
        u, v, z = self.project(np.asarray(p, np.float64)[None]); return (int(round(u[0])), int(round(v[0]))), z[0]


def frustum_pts(c2w, K, size=0.10):
    """5 world points: centre + 4 image corners at depth `size`."""
    corners = np.array([[0, 0, 1], [R, 0, 1], [R, R, 1], [0, R, 1]], float)
    cam = (np.linalg.inv(K) @ corners.T).T * size
    pts = np.vstack([np.zeros(3), cam])
    return pts @ c2w[:3, :3].T + c2w[:3, 3]


def draw_frustum(img, vc, c2w, K, color, size=0.10, thick=1, fill=None):
    ps = [vc.pt(p)[0] for p in frustum_pts(c2w, K, size)]
    if fill is not None:
        ov = img.copy(); cv2.fillPoly(ov, [np.array(ps[1:], np.int32)], fill); cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)
    for i in range(1, 5):
        cv2.line(img, ps[0], ps[i], color, thick, cv2.LINE_AA)
        cv2.line(img, ps[i], ps[1 + i % 4], color, thick, cv2.LINE_AA)
    cv2.circle(img, ps[0], 3, color, -1, cv2.LINE_AA)
    return ps[0]


def draw_path(img, vc, c2ws, color, thick=1, alpha=1.0, dashed=False):
    ps = [vc.pt(c[:3, 3])[0] for c in c2ws]
    ov = img.copy() if alpha < 1 else img
    for a, b in zip(ps[:-1], ps[1:]):
        if dashed and (hash(a) % 2): continue
        cv2.line(ov, a, b, color, thick, cv2.LINE_AA)
    if alpha < 1: cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    return ps


def label(img, text, org, color=(235, 235, 235), scale=0.5, thick=1, bg=None):
    if bg is not None:
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
        cv2.rectangle(img, (org[0] - 4, org[1] - th - 5), (org[0] + tw + 4, org[1] + 4), bg, -1)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def rounded_thumb(img, size, border):
    t = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (size - 1, size - 1), border, 2)
    return t


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", required=True); ap.add_argument("--gen", required=True)
    ap.add_argument("--traj", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--still", type=int, default=-1, help="write one PNG at this frame instead of the video")
    ap.add_argument("--cloud_stride", type=int, default=2, help="use every k-th target frame for the cloud")
    ap.add_argument("--eye", type=float, nargs=3, default=None, help="override virtual camera eye (world)")
    ap.add_argument("--look", type=float, nargs=3, default=None)
    a = ap.parse_args()

    rec8 = a.cid[:8]
    z = np.load(f"{CLIPS}/{rec8}/{a.cid}.npz", allow_pickle=True)
    cp = np.load(f"{CLIPS}/{rec8}/{a.cid}.campose.npz", allow_pickle=True)
    rf = np.load(f"{CLIPS}/{rec8}/{a.cid}.refs.npz", allow_pickle=True)
    tj = np.load(a.traj, allow_pickle=True)
    gen = read_video(a.gen)                                   # (77,480,960,3): [agent1 | agent2]
    assert len(gen) >= T, gen.shape

    views = {"L": dict(K=np.asarray(cp["K_leader"], np.float64), w2c_all=tj["leader_w2c"], vrs=tj["leader_vrs"],
                       col=COL_A, name="Agent 1"),
             "H": dict(K=np.asarray(cp["K_helper"], np.float64), w2c_all=tj["helper_w2c"], vrs=tj["helper_vrs"],
                       col=COL_B, name="Agent 2")}
    for s, V in views.items():
        V["target"] = np.stack([jdec(b) for b in z[f"{s}_target"]])
        V["warped"] = np.stack([jdec(b) for b in z[f"{s}_warped"]])
        V["pose"] = np.stack([jdec(b) for b in z[f"{s}_pose_person"]])
        V["human"] = np.stack([ddec(b) > 0 for b in z[f"{s}_human_depth"]])
        V["w2c"] = V["w2c_all"][P:P + T]; V["c2w"] = np.linalg.inv(V["w2c"]); V["c2w_all"] = np.linalg.inv(V["w2c_all"])
        V["srcf"] = z[f"{s}_srcf"]; V["srcview"] = z[f"{s}_srcview"]

    # ---- 3D lifting: DA3 depth with GT extrinsics on subsampled target frames of both views
    sel = list(range(0, T, a.cloud_stride))
    pts, cols = [], []
    for s, V in views.items():
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", f".{a.cid}_{s}_da3_s{a.cloud_stride}.npz")
        d = da3_depths(cache, V["target"][sel], V["w2c"][sel], V["K"])
        for k, t in enumerate(sel):
            valid = ~cv2.dilate(V["human"][t].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
            p, c = backproject(d[k], V["target"][t], V["K"], V["w2c"][t], valid, stride=1)
            pts.append(p); cols.append(c)
    pts = np.concatenate(pts); cols = np.concatenate(cols)
    pts, cols = clean_cloud(pts, cols)
    print(f"cloud: {len(pts)} points", flush=True)

    # ---- virtual camera: behind/above the two agents, looking where they look (world z up)
    up = np.array([0, 0, 1.0])
    cams = np.concatenate([views["L"]["c2w"][:, :3, 3], views["H"]["c2w"][:, :3, 3]])
    fwd = np.concatenate([views["L"]["c2w"][:, :3, 2], views["H"]["c2w"][:, :3, 2]]).mean(0); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
    centre = cams.mean(0) + fwd * 0.9
    eye = np.array(a.eye) if a.eye else cams.mean(0) - fwd * 2.6 + up * 1.7
    look = np.array(a.look) if a.look else centre + np.array([0, 0, -0.35])
    vc = VirtualCam(eye, look, up)
    base = vc.render_cloud(pts, cols)
    # vignette-free: darken slightly so overlays pop
    base = (base.astype(np.float32) * 0.92).astype(np.uint8)

    ref_c2w = np.linalg.inv(np.asarray(rf["reference_w2c"], np.float64)); ref_K = np.asarray(rf["reference_K"], np.float64)
    ref_img = rf["reference_frames"]; ref_view = rf["reference_src_view"]
    TH = 50; thumbs = [rounded_thumb(ref_img[i], TH, COL_A if ref_view[i] == 0 else COL_B) for i in range(len(ref_img))]

    def frame(t):
        # ------------------------------------------------ right: 3D panel
        img = base.copy()
        # history trajectories (200 past frames) faint, target trajectory up to t bright
        for s, V in views.items():
            draw_path(img, vc, V["c2w_all"][:P + 1], V["col"], 1, alpha=0.45)
            if t > 0: draw_path(img, vc, V["c2w"][:t + 1], V["col"], 2)
        # anchor memory: 8 shared references at their poses + thumbnails along the right edge
        ref_pts = []
        for i in range(len(ref_img)):
            c = COL_A if ref_view[i] == 0 else COL_B
            ref_pts.append(draw_frustum(img, vc, ref_c2w[i], ref_K[i], COL_REF, size=0.07, thick=1))
        x0 = W3 - TH - 10; y0 = 96; gap = (H3 - y0 - 8 - TH) // max(1, len(ref_img) - 1)
        gap = min(gap, TH + 4)
        order = np.argsort([p[1] for p in ref_pts])           # top-to-bottom by frustum position: fewer crossings
        ov = img.copy()
        for slot, i in enumerate(order):
            y = y0 + slot * gap
            cv2.line(ov, (x0, y + TH // 2), ref_pts[i], COL_REF, 1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
        for slot, i in enumerate(order):
            y = y0 + slot * gap; img[y:y + TH, x0:x0 + TH] = thumbs[i]
        # geometric memory: the frame warped into each current target view
        for s, V in views.items():
            sv = int(V["srcview"][t]); src_key = s if sv == 0 else ("H" if s == "L" else "L")
            SV = views[src_key]; idx = int(np.where(SV["vrs"] == V["srcf"][t])[0][0])
            src_c2w = SV["c2w_all"][idx]
            ps = draw_frustum(img, vc, src_c2w, SV["K"], COL_SRC, size=0.11, thick=2)
            pt = draw_frustum(img, vc, V["c2w"][t], V["K"], V["col"], size=0.13, thick=2, fill=V["col"])
            cv2.arrowedLine(img, ps, pt, COL_SRC, 2, cv2.LINE_AA, tipLength=0.25)
            label(img, V["name"], (pt[0] + 8, pt[1] - 6), V["col"], 0.48, 1, bg=BG)
        # legend
        label(img, "Shared environment memory", (12, 22), (240, 240, 240), 0.55, 1)
        cv2.line(img, (12, 40), (34, 40), COL_SRC, 2); label(img, "warp source -> target  (stream-aligned geometric memory)", (40, 44), (200, 200, 200), 0.42)
        cv2.rectangle(img, (12, 52), (34, 62), COL_REF, 1); label(img, "shared anchor references  (cross-stream anchor memory)", (40, 61), (200, 200, 200), 0.42)
        label(img, "past 200 frames, faint / generated 77 frames, bold", (40, 78), (170, 170, 170), 0.4)
        label(img, f"frame {t + 1:02d}/{T}", (12, H3 - 12), (170, 170, 170), 0.45)

        # ------------------------------------------------ left: conditions + generated
        left = np.full((2 * CELL, 3 * CELL, 3), BG, np.uint8)
        for r, (s, V) in enumerate(views.items()):
            g = gen[t][:, :480] if s == "L" else gen[t][:, 480:]
            cells = [V["warped"][t], V["pose"][t], g]
            for c, im in enumerate(cells):
                left[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = cv2.resize(im, (CELL, CELL), interpolation=cv2.INTER_AREA)
            cv2.rectangle(left, (0, r * CELL), (3 * CELL - 1, (r + 1) * CELL - 1), V["col"], 2)
            label(left, V["name"], (8, r * CELL + 22), V["col"], 0.55, 1, bg=BG)
        for c, txt in enumerate(["warped (geometric memory)", "pose (shared action)", "generated"]):
            label(left, txt, (c * CELL + 8, 2 * CELL - 10), (235, 235, 235), 0.45, 1, bg=BG)
        # thin divider and compose
        out = np.full((H3, 3 * CELL + 6 + W3, 3), BG, np.uint8)
        out[:, :3 * CELL] = left; out[:, 3 * CELL + 6:] = img
        return out

    if a.still >= 0:
        cv2.imwrite(a.out, cv2.cvtColor(frame(a.still), cv2.COLOR_RGB2BGR)); print("still ->", a.out); return
    fr0 = frame(0); hh, ww = fr0.shape[:2]
    pp = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ww}x{hh}", "-r", "10",
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", a.out],
                          stdin=subprocess.PIPE)
    for t in range(T):
        pp.stdin.write(np.ascontiguousarray(frame(t)).tobytes())
    pp.stdin.close(); pp.wait(); print("video ->", a.out)


if __name__ == "__main__":
    main()
