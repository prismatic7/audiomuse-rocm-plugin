"""Validate the torch/ROCm clustering backend against sklearn CPU on real data.

Runs inside the audiomuse-ai-rocm worker image with GPU passthrough.
Checks: (1) KMeans/PCA correctness on 60 real CLAP embeddings from the prior
local-test DB, (2) the core clustering seam (_apply_clustering_model) using the
GPU backend without core code changes, (3) CPU-vs-GPU speedup at realistic
library scale.
"""
import csv
import os
import sys
import time

import numpy as np

import clustering  # the plugin backend module (mounted at /p)

DATA = "/p/am_clap_data.csv"
MOOD_LABELS = [
    "rock", "pop", "alternative", "indie", "electronic", "female vocalists", "dance", "00s", "alternative rock", "jazz",
    "beautiful", "metal", "chillout", "male vocalists", "classic rock", "soul", "indie rock", "Mellow", "electronica", "80s",
    "folk", "90s", "chill", "instrumental", "punk", "oldies", "blues", "hard rock", "ambient", "acoustic", "experimental",
    "female vocalist", "guitar", "Hip-Hop", "70s", "party", "country", "easy listening", "sexy", "catchy", "funk", "electro",
    "heavy metal", "Progressive rock", "60s", "rnb", "indie pop", "sad", "House", "happy",
]
OTHER_FEATURE_LABELS = ["danceable", "aggressive", "happy", "party", "relaxed", "sad"]


def load_real_data():
    rows = []
    with open(DATA, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    embs = np.zeros((len(rows), 512), dtype=np.float32)
    feats = np.zeros((len(rows), 2 + len(MOOD_LABELS) + len(OTHER_FEATURE_LABELS)), dtype=np.float64)
    for i, r in enumerate(rows):
        embs[i] = np.frombuffer(bytes.fromhex(r["emb_hex"]), dtype="<f4").copy()
        tempo = float(r["tempo"] or 0.0)
        energy = float(r["energy"] or 0.0)
        tempo_norm = np.clip((tempo - 40.0) / 160.0, 0.0, 1.0)
        energy_norm = np.clip((energy - 0.01) / 0.14, 0.0, 1.0)
        vec = [tempo_norm, energy_norm]
        mood = np.zeros(len(MOOD_LABELS))
        for pair in (r["mood_vector"] or "").split(","):
            if ":" in pair:
                label, score = pair.split(":")
                if label in MOOD_LABELS:
                    mood[MOOD_LABELS.index(label)] = float(score)
        vec += mood.tolist()
        other = np.zeros(len(OTHER_FEATURE_LABELS))
        for pair in (r["other_features"] or "").split(","):
            if ":" in pair:
                label, score = pair.split(":")
                if label in OTHER_FEATURE_LABELS:
                    other[OTHER_FEATURE_LABELS.index(label)] = float(score)
        vec += other.tolist()
        feats[i] = vec
    return rows, embs, feats


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(np.asarray(a), np.asarray(b))


def bench(fn, n=1):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def section(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70, flush=True)


def main():
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans as skKMeans
    from sklearn.decomposition import PCA as skPCA
    from sklearn.metrics import silhouette_score

    section("0. environment")
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0))
    print("gpu available (plugin):", clustering.check_gpu_available())

    rows, embs, feats = load_real_data()
    print(f"real data: {len(rows)} tracks, embeddings {embs.shape}, features {feats.shape}")

    emb_s = StandardScaler().fit_transform(embs)          # float64, matches core
    feat_s = StandardScaler().fit_transform(feats)
    emb_s32 = emb_s.astype(np.float32)
    print(f"scaled: emb_s {emb_s.shape} {emb_s.dtype}, feat_s {feat_s.shape}")

    section("1. KMeans correctness on real CLAP embeddings (same init centers)")
    for seed in [0, 7]:
        sk = skKMeans(n_clusters=6, init="random", n_init=1, random_state=seed)
        sk.fit(emb_s)
        tk = clustering.TorchKMeans(
            n_clusters=6, init=sk.cluster_centers_, n_init=1, random_state=seed
        )
        tk.fit(emb_s)
        exact = float(np.mean(sk.labels_ == tk.labels_))
        print(f"  seed={seed}: same-init exact-match {exact:.4f}  ARI {ari(sk.labels_, tk.labels_):.4f}  "
              f"inertia CPU {sk.inertia_:.3e} vs GPU {tk.inertia_:.3e}")

    section("1b. KMeans best-inertia parity across seeds (k-means++, n_init=5)")
    for seed in range(5):
        sk = skKMeans(n_clusters=8, init="k-means++", n_init=5, random_state=seed)
        tk = clustering.TorchKMeans(n_clusters=8, init="k-means++", n_init=5, random_state=seed)
        sk.fit(emb_s)
        tk.fit(emb_s)
        print(f"  seed={seed}: inertia CPU {sk.inertia_:.3e} vs GPU {tk.inertia_:.3e} "
              f"ratio {tk.inertia_/sk.inertia_:.4f}  ARI {ari(sk.labels_, tk.labels_):.4f}")

    section("2. KMeans correctness (k-means++, core default n_init=10)")
    for seed in [1, 42]:
        sk = skKMeans(n_clusters=8, init="k-means++", n_init=10, random_state=seed)
        tk = clustering.TorchKMeans(n_clusters=8, init="k-means++", n_init=10, random_state=seed)
        sk.fit(emb_s)
        tk.fit(emb_s)
        sil_g = silhouette_score(emb_s, tk.labels_)
        sil_c = silhouette_score(emb_s, sk.labels_)
        print(f"  seed={seed}: ARI {ari(sk.labels_, tk.labels_):.4f}  "
              f"inertia CPU {sk.inertia_:.3e} vs GPU {tk.inertia_:.3e}  "
              f"silhouette CPU {sil_c:.3f} vs GPU {sil_g:.3f}")

    section("2b. is GPU-vs-CPU variance normal? CPU-vs-CPU baseline")
    def cpu_vs_cpu(data, k, pairs):
        for a, b in pairs:
            la = skKMeans(n_clusters=k, init="k-means++", n_init=10, random_state=a).fit_predict(data)
            lb = skKMeans(n_clusters=k, init="k-means++", n_init=10, random_state=b).fit_predict(data)
            print(f"  CPU seed {a} vs {b}: ARI {ari(la, lb):.4f}")
    def gpu_vs_cpu(data, k, pairs):
        for a, b in pairs:
            la = skKMeans(n_clusters=k, init="k-means++", n_init=10, random_state=a).fit_predict(data)
            lb = clustering.TorchKMeans(n_clusters=k, init="k-means++", n_init=10, random_state=b).fit_predict(data)
            print(f"  GPU seed {b} vs CPU seed {a}: ARI {ari(la, lb):.4f}")
    pairs = [(1, 2), (1, 42), (42, 2)]
    print("  embeddings (d=512, k=8):")
    cpu_vs_cpu(emb_s, 8, pairs)
    gpu_vs_cpu(emb_s, 8, pairs)
    print("  features (d=58, k=8):")
    cpu_vs_cpu(feat_s, 8, pairs)
    gpu_vs_cpu(feat_s, 8, pairs)

    section("2c. KMeans best-of-seeds parity (k-means++, n_init=10, 5 seeds)")
    sk_best = min(
        skKMeans(n_clusters=8, init="k-means++", n_init=10, random_state=s).fit(emb_s).inertia_
        for s in range(5)
    )
    tk_best = min(
        clustering.TorchKMeans(n_clusters=8, init="k-means++", n_init=10, random_state=s).fit(emb_s).inertia_
        for s in range(5)
    )
    print(f"  embeddings: CPU best {sk_best:.3e} vs GPU best {tk_best:.3e} ratio {tk_best/sk_best:.4f}")

    section("3. PCA correctness on real CLAP embeddings")
    for nc in [4, 16]:
        sp = skPCA(n_components=nc)
        tp = clustering.TorchPCA(n_components=nc)
        sp.fit(emb_s)
        tp.fit(emb_s)
        cos = np.abs(np.diag(np.corrcoef(sp.components_, tp.components_)[:nc, nc:]))
        t_ratio = tp.explained_variance_ratio_
        s_ratio = sp.explained_variance_ratio_
        err = np.max(np.abs(s_ratio - t_ratio[:len(s_ratio)]))
        recon_cpu = sp.inverse_transform(sp.transform(emb_s))
        recon_gpu = tp.inverse_transform(tp.transform(emb_s))
        rmse = float(np.sqrt(np.mean((recon_cpu - recon_gpu) ** 2)))
        print(f"  n_comp={nc}: components cos {np.round(cos, 3)}  evr max-abs-diff {err:.2e}  "
              f"inverse_transform rmse {rmse:.2e}")

    section("4. full pipeline on real data (StandardScaler -> PCA -> KMeans)")
    sk = skKMeans(n_clusters=6, init="k-means++", n_init=10, random_state=5)
    tk = clustering.TorchKMeans(n_clusters=6, init="k-means++", n_init=10, random_state=5)
    sp = skPCA(n_components=12)
    tp = clustering.TorchPCA(n_components=12)
    Xc = sp.fit_transform(emb_s)
    Xg = tp.fit_transform(emb_s.astype(np.float32))
    sk.fit(Xc)
    tk.fit(Xg)
    print(f"  ARI(CPU vs GPU labels) {ari(sk.labels_, tk.labels_):.4f}  "
          f"silhouette CPU {silhouette_score(Xc, sk.labels_):.3f} GPU {silhouette_score(Xg, tk.labels_):.3f}")

    section("4b. feature-mode clustering on real data (tempo/energy/moods, d=58)")
    for seed in [1, 7]:
        sk = skKMeans(n_clusters=8, init="k-means++", n_init=10, random_state=seed)
        tk = clustering.TorchKMeans(n_clusters=8, init="k-means++", n_init=10, random_state=seed)
        sk.fit(feat_s)
        tk.fit(feat_s)
        print(f"  seed={seed}: ARI {ari(sk.labels_, tk.labels_):.4f}  "
              f"inertia CPU {sk.inertia_:.3e} vs GPU {tk.inertia_:.3e}  "
              f"silhouette CPU {silhouette_score(feat_s, sk.labels_):.3f} GPU {silhouette_score(feat_s, tk.labels_):.3f}")

    section("5. core seam: tasks.clustering_helper._apply_clustering_model")
    sys.path.insert(0, "/app")
    import tasks.clustering_helper as ch
    print("  before install: USE_GPU_CLUSTERING", ch.USE_GPU_CLUSTERING,
          "| get_clustering_model:", ch.get_clustering_model.__module__)
    cfg = {"method": "kmeans", "params": {"n_clusters": 6}}
    labels_cpu, centers_cpu, model_cpu = ch._apply_clustering_model(emb_s, cfg, "[seam]", 1)
    print(f"  CPU baseline via core: labels {len(np.unique(labels_cpu))} clusters, "
          f"model {type(model_cpu).__name__}, using_gpu={getattr(model_cpu, 'using_gpu', None)}")
    ok = clustering.install()
    print("  install() ->", ok)
    print("  after install: USE_GPU_CLUSTERING", ch.USE_GPU_CLUSTERING,
          "| get_clustering_model:", ch.get_clustering_model.__module__)
    labels_gpu, centers_gpu, model_gpu = ch._apply_clustering_model(emb_s, cfg, "[seam]", 2)
    print(f"  GPU via core seam: {len(np.unique(labels_gpu))} clusters, "
          f"model {type(model_gpu).__name__}, using_gpu={getattr(model_gpu, 'using_gpu', None)}")
    print(f"  seam ARI(CPU vs GPU labels): {ari(labels_cpu, labels_gpu):.4f}")

    section("6. speed: CPU sklearn vs GPU torch at realistic library scale")
    rng = np.random.default_rng(0)
    for n, d, k in [(2000, 512, 20), (10000, 512, 50), (20000, 58, 30)]:
        X = rng.standard_normal((n, d)).astype(np.float32)
        sk = skKMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0)
        tk = clustering.TorchKMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0)
        tc = bench(lambda: sk.fit_predict(X), n=3)
        tg = bench(lambda: tk.fit_predict(X), n=3)
        print(f"  KMeans n={n} d={d} k={k}: CPU {tc*1e3:.0f} ms | GPU {tg*1e3:.0f} ms "
              f"| x{tc/tg:.1f} faster  ARI {ari(sk.labels_, tk.labels_):.4f}")
        sp = skPCA(n_components=min(40, d))
        tp = clustering.TorchPCA(n_components=min(40, d))
        tc2 = bench(lambda: sp.fit_transform(X), n=3)
        tg2 = bench(lambda: tp.fit_transform(X), n=3)
        print(f"  PCA  n={n} d={d} nc={min(40, d)}: CPU {tc2*1e3:.0f} ms | GPU {tg2*1e3:.0f} ms "
              f"| x{tc2/tg2:.1f} faster")
        tc3 = tc + tc2
        tg3 = tg + tg2
        print(f"  ITER n={n} d={d} (PCA+KMeans): CPU {tc3*1e3:.0f} ms | GPU {tg3*1e3:.0f} ms "
              f"| x{tc3/tg3:.1f} faster net")

    print("\nVALIDATION DONE")


if __name__ == "__main__":
    main()