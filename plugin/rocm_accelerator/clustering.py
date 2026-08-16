# AudioMuse-AI ROCm Accelerator plugin - native ROCm clustering backend.
#
# Replaces the cuML-backed clustering_gpu path on AMD GPUs without touching
# core: the module mirrors the public surface of tasks.clustering_gpu
# (check_gpu_available / get_clustering_model / get_pca_model) and is installed
# over the core functions at worker start. KMeans and PCA run on the GPU via
# PyTorch ROCm; DBSCAN / GMM / spectral keep the scikit-learn CPU path, same as
# the NVIDIA build already does for GMM and spectral.

import logging

import numpy as np

logger = logging.getLogger(__name__)

_GPU_AVAILABLE = None
_GPU_CHECK_DONE = False


def check_gpu_available():
    global _GPU_AVAILABLE, _GPU_CHECK_DONE
    if _GPU_CHECK_DONE:
        return _GPU_AVAILABLE
    try:
        import torch

        _GPU_AVAILABLE = bool(torch.cuda.is_available())
        if _GPU_AVAILABLE:
            logger.info("ROCm clustering backend: HIP device visible to torch")
    except Exception as e:
        _GPU_AVAILABLE = False
        logger.info("ROCm clustering backend unavailable: %s", e)
    _GPU_CHECK_DONE = True
    return _GPU_AVAILABLE


def _to_tensor(X):
    import torch

    arr = np.asarray(X)
    dtype = torch.float32
    if arr.dtype == torch.float64:
        dtype = torch.float64
    return torch.as_tensor(arr, device="cuda", dtype=dtype)


def _to_numpy(t):
    return t.detach().cpu().numpy()


def _sq_dist_all(X, C):
    """Squared Euclidean distance matrix X (n,d) vs C (k,d) via GEMM."""
    import torch

    x2 = torch.sum(X * X, dim=1, keepdim=True)
    c2 = torch.sum(C * C, dim=1, keepdim=True)
    return x2 - 2.0 * (X @ C.T) + c2.T


def _kmeans_pp_centers(X, k, seed):
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, device=X.device)
    centers = [X[perm[0]].clone().unsqueeze(0)]
    min_d2 = torch.sum((X - centers[0]) ** 2, dim=1)
    for _ in range(1, k):
        sel = torch.multinomial(min_d2, 1)
        c = X[sel]
        centers.append(c)
        d2 = torch.sum((X - c) ** 2, dim=1)
        min_d2 = torch.minimum(min_d2, d2)
    return torch.cat(centers, dim=0)


def _kmeans_lloyd(X, k, init_centers, seed, max_iter=300, tol=1e-4):
    import torch

    centers = init_centers.clone()
    labels = torch.empty(X.shape[0], dtype=torch.int64, device=X.device)
    for _ in range(max_iter):
        dist = _sq_dist_all(X, centers)
        labels = torch.argmin(dist, dim=1)
        counts = torch.bincount(labels, minlength=k).to(X.dtype)
        sums = torch.zeros((k, X.shape[1]), dtype=X.dtype, device=X.device)
        sums.index_add_(0, labels, X)
        new_centers = sums.clone()
        non_empty = counts > 0
        new_centers[non_empty] /= counts[non_empty].unsqueeze(1)
        new_centers[~non_empty] = centers[~non_empty]
        shift = torch.sum((new_centers - centers) ** 2)
        centers = new_centers
        if shift <= tol:
            break
    dist = _sq_dist_all(X, centers)
    labels = torch.argmin(dist, dim=1)
    inertia = torch.sum(dist[torch.arange(X.shape[0]), labels])
    return centers, labels, inertia


class TorchKMeans:
    """sklearn-compatible KMeans running on the ROCm GPU."""

    def __init__(self, n_clusters=8, init="k-means++", n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.using_gpu = False

    def fit(self, X, y=None):
        import torch

        if not check_gpu_available():
            return self._fit_cpu(X)
        Xt = _to_tensor(X)
        k = int(self.n_clusters)
        n = Xt.shape[0]
        best = None
        rng_state = self.random_state
        for run in range(int(self.n_init)):
            run_seed = rng_state if rng_state is not None else None
            if run > 0 and rng_state is not None:
                run_seed = rng_state + run
            if isinstance(self.init, str) and self.init == "k-means++":
                init_centers = _kmeans_pp_centers(Xt, k, run_seed)
            elif isinstance(self.init, str) and self.init == "random":
                if run_seed is not None:
                    torch.manual_seed(run_seed)
                perm = torch.randperm(n, device=Xt.device)
                init_centers = Xt[perm[:k]].clone()
            else:
                init_centers = torch.as_tensor(self.init, device="cuda", dtype=Xt.dtype)
            centers, labels, inertia = _kmeans_lloyd(Xt, k, init_centers, run_seed)
            if best is None or float(inertia) < best[0]:
                best = (float(inertia), centers, labels)
        self.inertia_, self.cluster_centers_, self.labels_ = best
        self.cluster_centers_ = _to_numpy(self.cluster_centers_)
        self.labels_ = _to_numpy(self.labels_).astype(np.int32)
        self.using_gpu = True
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before predict")
        import torch

        Xt = _to_tensor(X)
        C = torch.as_tensor(self.cluster_centers_, device="cuda", dtype=Xt.dtype)
        dist = _sq_dist_all(Xt, C)
        return _to_numpy(torch.argmin(dist, dim=1)).astype(np.int32)

    def _fit_cpu(self, X):
        from sklearn.cluster import KMeans

        model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        model.fit(X)
        self.cluster_centers_ = model.cluster_centers_
        self.labels_ = model.labels_
        self.inertia_ = model.inertia_
        self.n_iter_ = model.n_iter_
        self.using_gpu = False
        return self


class TorchPCA:
    """sklearn-compatible PCA running on the ROCm GPU (full SVD)."""

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None
        self.mean_ = None
        self.using_gpu = False

    def fit(self, X, y=None):
        import torch

        if not check_gpu_available():
            return self._fit_cpu(X)
        Xt = _to_tensor(X)
        n, d = Xt.shape
        n_comp = self.n_components
        if n_comp is None:
            n_comp = min(n, d)
        n_comp = max(1, min(int(n_comp), n, d))
        mean = Xt.mean(dim=0)
        Xc = Xt - mean
        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        components = Vh[:n_comp].contiguous()
        # Match sklearn's sign convention (largest absolute entry positive).
        max_idx = torch.argmax(torch.abs(components), dim=1)
        signs = torch.sign(components[torch.arange(n_comp), max_idx])
        components *= signs.unsqueeze(1)
        explained_var = (S[:n_comp] ** 2) / (n - 1)
        total_var = (S ** 2).sum() / (n - 1)
        self.components_ = _to_numpy(components)
        self.explained_variance_ratio_ = _to_numpy(explained_var / total_var)
        self.n_components_ = n_comp
        self.mean_ = _to_numpy(mean)
        self.using_gpu = True
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        if self.components_ is None:
            raise ValueError("Model must be fitted before transform")
        import torch

        Xt = _to_tensor(X)
        mean = torch.as_tensor(self.mean_, device="cuda", dtype=Xt.dtype)
        comp = torch.as_tensor(self.components_, device="cuda", dtype=Xt.dtype)
        return _to_numpy((Xt - mean) @ comp.T)

    def inverse_transform(self, X):
        if self.components_ is None:
            raise ValueError("Model must be fitted before inverse_transform")
        import torch

        Xt = _to_tensor(X)
        mean = torch.as_tensor(self.mean_, device="cuda", dtype=Xt.dtype)
        comp = torch.as_tensor(self.components_, device="cuda", dtype=Xt.dtype)
        return _to_numpy(Xt @ comp + mean)

    def _fit_cpu(self, X):
        from sklearn.decomposition import PCA

        model = PCA(n_components=self.n_components)
        model.fit(X)
        self.components_ = model.components_
        self.explained_variance_ratio_ = model.explained_variance_ratio_
        self.n_components_ = model.n_components_
        self.mean_ = model.mean_
        self.using_gpu = False
        return self


def get_clustering_model(method, params, use_gpu=False):
    """Mirror of tasks.clustering_gpu.get_clustering_model."""
    if method == "kmeans":
        if use_gpu and check_gpu_available():
            return TorchKMeans(n_clusters=params["n_clusters"], init="k-means++", n_init=10)
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=params["n_clusters"], init="k-means++", n_init=10)
    if method == "dbscan":
        from sklearn.cluster import DBSCAN

        return DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    if method == "gmm":
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=params["n_components"],
            covariance_type="diag",
            init_params="k-means++",
            n_init=10,
            random_state=None,
            reg_covar=1e-4,
        )
        return model
    if method == "spectral":
        from sklearn.cluster import SpectralClustering

        return SpectralClustering(
            n_clusters=params["n_clusters"],
            assign_labels="kmeans",
            affinity="nearest_neighbors",
            n_neighbors=20,
            random_state=params.get("random_state"),
            n_init=10,
            verbose=False,
        )
    raise ValueError(f"Unsupported clustering method: {method}")


def get_pca_model(n_components, use_gpu=False):
    """Mirror of tasks.clustering_gpu.get_pca_model."""
    if use_gpu and check_gpu_available():
        return TorchPCA(n_components=n_components)
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components)


def install():
    """Swap core's clustering factory for this GPU backend, in place.

    Runs at worker start, before any clustering job imports the helper, and
    patches both namespaces core resolves at call time: tasks.clustering_gpu
    (the source) and tasks.clustering_helper (where the functions are bound by
    `from .clustering_gpu import ...` at module import).
    """
    if not check_gpu_available():
        logger.warning("ROCm clustering backend not installed: no CUDA-capable torch")
        return False

    try:
        import tasks.clustering_helper as helper
        import tasks.clustering_gpu as gpu
    except ImportError as e:
        logger.warning("ROCm clustering backend not installed: %s", e)
        return False

    if not helper.USE_GPU_CLUSTERING:
        logger.info(
            "ROCm clustering backend available but USE_GPU_CLUSTERING is off - "
            "leaving core's clustering on the CPU scikit-learn path."
        )
        return False

    gpu.get_clustering_model = get_clustering_model
    gpu.get_pca_model = get_pca_model
    gpu.check_gpu_available = check_gpu_available
    helper.get_clustering_model = get_clustering_model
    helper.get_pca_model = get_pca_model
    helper.GPU_CLUSTERING_AVAILABLE = True
    logger.info(
        "ROCm clustering backend installed: KMeans + PCA on GPU via torch, "
        "DBSCAN/GMM/spectral on CPU"
    )
    return True