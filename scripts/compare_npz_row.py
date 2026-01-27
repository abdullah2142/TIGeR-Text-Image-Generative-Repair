import argparse, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--row_id", required=True)
    args = ap.parse_args()

    A = np.load(args.a, allow_pickle=True)
    B = np.load(args.b, allow_pickle=True)

    rid = str(args.row_id)
    a_ids = A["row_id"].astype(str)
    b_ids = B["row_id"].astype(str)

    if rid not in set(a_ids):
        raise SystemExit(f"row_id {rid} not found in A")
    if rid not in set(b_ids):
        raise SystemExit(f"row_id {rid} not found in B")

    ia = int(np.where(a_ids == rid)[0][0])
    ib = int(np.where(b_ids == rid)[0][0])

    print("A keys:", A.files)
    print("B keys:", B.files)
    print("row_id:", rid, "idxA:", ia, "idxB:", ib)

    common = sorted(set(A.files) & set(B.files))
    for k in common:
        va, vb = A[k], B[k]

        # print text inputs if any
        if va.dtype == object and len(va.shape) == 1:
            try:
                ta, tb = va[ia], vb[ib]
                if isinstance(ta, str) or isinstance(tb, str):
                    if ta != tb:
                        print(f"\n{k} (TEXT) changed:")
                        print(" A:", ta)
                        print(" B:", tb)
                    else:
                        print(f"\n{k} (TEXT) same.")
            except Exception:
                pass
            continue

        # compare vector embeddings (N,D)
        if hasattr(va, "shape") and len(va.shape) == 2 and va.shape[0] > ia and vb.shape[0] > ib:
            xa, xb = va[ia], vb[ib]
            if np.issubdtype(xa.dtype, np.floating) and np.issubdtype(xb.dtype, np.floating):
                l2 = float(np.linalg.norm(xa - xb))
                same = bool(np.allclose(xa, xb))
                print(f"{k:25s}  L2={l2:.6f}  same={same}")
            continue

    print("\nDone.")

if __name__ == "__main__":
    main()
