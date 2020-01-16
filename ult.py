from pandas import unique, Index
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal
from matplotlib import rcParams
import matplotlib.pyplot as pl
import numpy as np
def default_basis(adata):
    keys = [key for key in ['pca', 'tsne', 'umap'] if 'X_' + key in adata.obsm.keys()]
    if not keys:
        raise ValueError('No basis specified.')
    return keys[-1] if len(keys) > 0 else None


def default_size(adata,rcParams_style):
    adjusted, classic = 1.2e5 / adata.n_obs, 20
    return np.mean([adjusted, classic]) if rcParams_style == 'scvelo' else adjusted


def default_color(adata):
    return 'clusters' if 'clusters' in adata.obs.keys() else 'louvain' if 'louvain' in adata.obs.keys() else 'grey'


def get_components(components=None, basis=None, projection=None):
    if components is None: components = '1,2,3' if projection == '3d' else '1,2'
    if isinstance(components, str): components = components.split(',')
    components = np.array(components).astype(int) - 1
    if 'diffmap' in basis or 'vmap' in basis: components += 1
    return components

def default_arrow(size):
    if isinstance(size, (list, tuple)) and len(size) == 3:
        head_l, head_w, ax_l = size
    elif type(size) == int or type(size) == float:
        head_l, head_w, ax_l = 12 * size, 10 * size, 8 * size
    else:
        head_l, head_w, ax_l = 12, 10, 8
    return head_l, head_w, ax_l

def make_unique_list(key, allow_array=False):
    if isinstance(key, Index): key = key.tolist()
    is_list = isinstance(key, (list, tuple, np.record)) if allow_array else isinstance(key, (list, tuple, np.ndarray, np.record))
    is_list_of_str = is_list and all(isinstance(item, str) for item in key)
    return key if is_list_of_str else key if is_list and len(key) < 20 else [key]

def get_basis(adata, basis):
    if isinstance(basis, str) and basis.startswith('X_'):
        basis = basis[2:]
    check_basis(adata, basis)
    return basis

def velocity_embedding_changed(adata, basis, vkey):
    if 'X_' + basis not in adata.obsm.keys(): changed = False
    else:
        changed = vkey + '_' + basis not in adata.obsm_keys()
        if vkey + '_settings' in adata.uns.keys():
            sett = adata.uns[vkey + '_settings']
            changed |= 'embeddings' not in sett or basis not in sett['embeddings']
    return changed

def quiver_autoscale(X_emb, V_emb):
    import matplotlib.pyplot as pl
    scale_factor = np.abs(X_emb).max()  # just so that it handles very large values
    fig, ax = pl.subplots()
    Q = ax.quiver(X_emb[:, 0] / scale_factor, X_emb[:, 1] / scale_factor,
                  V_emb[:, 0], V_emb[:, 1], angles='xy', scale_units='xy', scale=None)
    Q._init()
    fig.clf()
    pl.close(fig)
    return Q.scale / scale_factor


def compute_velocity_on_grid(X_emb, V_emb, density=None, smooth=None, n_neighbors=None, min_mass=None, autoscale=True,
                             adjust_for_stream=False, cutoff_perc=None):
    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = .5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - .01 * np.abs(M - m)
        M = M + .01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None: n_neighbors = int(n_obs/50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1) / np.maximum(1, p_mass)[:, None]
    if min_mass is None: min_mass = 1

    if adjust_for_stream:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid ** 2).sum(0))
        min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.max(mass) * .9)
        cutoff = mass.reshape(V_grid[0].shape) < min_mass

        if cutoff_perc is None: cutoff_perc = 5
        length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)

        V_grid[0][cutoff] = np.nan
    else:
        min_mass *= np.percentile(p_mass, 99) / 100
        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

        if autoscale: V_grid /= 3 * quiver_autoscale(X_grid, V_grid)

    return X_grid, V_grid


def check_basis(adata, basis):
    if basis in adata.obsm.keys() and 'X_' + basis not in adata.obsm.keys():
        adata.obsm['X_' + basis] = adata.obsm[basis]
        logg.info('Renamed', '\'' + basis + '\'', 'to convention', '\'X_' + basis + '\' (adata.obsm).')
