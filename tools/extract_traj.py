"""Recover shared-world (multislam) camera poses for all 277 frames of a CoMind clip (200 history + 77 target).
Needs projectaria_tools:  COMIND_SHARED_WORLD=1 /data2/envs/comind/bin/python extract_traj.py --cid 58dd70af_069576 --out cache/traj_58dd70af_069576.npz
"""
import argparse, sys, numpy as np
sys.path.insert(0, "/home/ubuntu/dahyun/comind_data_preprocess")
from comind_pool_mat import load_rec, w2c_at   # reads COMIND_SHARED_WORLD=1 -> multislam trajectory

ap = argparse.ArgumentParser(); ap.add_argument("--cid", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
rec8 = a.cid[:8]
DP, SID, CAL, LIN, K, TDC, CAPTS, TR = load_rec(rec8)
z = np.load(f"/data4/comind_dataset/clips/{rec8}/{a.cid}.npz", allow_pickle=True)
out = {}
for w, key in (("leader", "leader_vrs"), ("helper", "helper_vrs")):
    idx = z[key]
    out[f"{w}_w2c"] = np.stack([w2c_at(TR, TDC, w, CAPTS[w][int(i)]) for i in idx]).astype(np.float64)
    out[f"{w}_vrs"] = idx; out[f"K_{w}"] = K[w]
np.savez(a.out, **out); print("saved", a.out)
