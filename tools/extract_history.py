"""Pull rectified RGB frames of a CoMind clip's observation history (200 past frames per wearer) plus every
warp-source frame, for the teaser's shared-environment-memory explainer.
  COMIND_SHARED_WORLD=1 /data2/envs/comind/bin/python extract_history.py --cid 58dd70af_069576 --out cache/history_58dd70af_069576.npz
"""
import argparse, sys, numpy as np, cv2
sys.path.insert(0, "/home/ubuntu/dahyun/comind_data_preprocess")
from comind_pool_mat import load_rec, rectify

ap = argparse.ArgumentParser(); ap.add_argument("--cid", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--stride", type=int, default=10); ap.add_argument("--res", type=int, default=256); a = ap.parse_args()
rec8 = a.cid[:8]
DP, SID, CAL, LIN, K, TDC, CAPTS, TR = load_rec(rec8)
z = np.load(f"/data4/comind_dataset/clips/{rec8}/{a.cid}.npz", allow_pickle=True)
out = {}
for w, key, s in (("leader", "leader_vrs", "L"), ("helper", "helper_vrs", "H")):
    vrs = z[key]; hist = list(vrs[:200:a.stride]); src = sorted(set(int(v) for v in z[f"{s}_srcf"]))
    # srcf may point into the partner's pool (srcview==1); gather those from the partner below
    need = sorted(set(int(v) for v in hist) | set(src))
    imgs = {}
    for v in need:
        if v in set(int(x) for x in vrs):   # only frames of this wearer
            imgs[v] = cv2.resize(rectify(DP, SID, LIN, CAL, w, v), (a.res, a.res), interpolation=cv2.INTER_AREA)
    out[f"{s}_hist_vrs"] = np.array(hist, np.int64)
    out[f"{s}_img_vrs"] = np.array(sorted(imgs), np.int64)
    out[f"{s}_img"] = np.stack([imgs[v] for v in sorted(imgs)]).astype(np.uint8)
    print(w, "history", len(hist), "images", len(imgs), flush=True)
# partner-pool sources
for s, o in (("L", "H"), ("H", "L")):
    sv = z[f"{s}_srcview"]; sf = z[f"{s}_srcf"]
    miss = sorted(set(int(f) for f, v in zip(sf, sv) if v == 1) - set(int(x) for x in out[f"{o}_img_vrs"]))
    if miss:
        w = "helper" if o == "H" else "leader"
        add = np.stack([cv2.resize(rectify(DP, SID, LIN, CAL, w, v), (a.res, a.res), interpolation=cv2.INTER_AREA) for v in miss])
        out[f"{o}_img"] = np.concatenate([out[f"{o}_img"], add]); out[f"{o}_img_vrs"] = np.concatenate([out[f"{o}_img_vrs"], np.array(miss)])
        print("partner sources added to", o, len(miss))
np.savez_compressed(a.out, **out); print("saved", a.out)
