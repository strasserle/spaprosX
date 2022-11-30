import pytest
import pandas as pd
from spapros.util.util import sample_cells


def test_sample_cells(tiny_adata):
    n_out = 100
    adata = sample_cells(tiny_adata,
                 n_out=n_out,
                 obs_key="celltype",
                 sampling_seed=0,
                 copy=True)
    celltypes = tiny_adata.obs["celltype"].unique()
    group_counts = tiny_adata.obs["celltype"].value_counts()
    sample_counts = adata.obs["celltype"].value_counts()
    assert adata.shape[0] == n_out
    for ct in celltypes:
        assert (sample_counts[ct] == group_counts[ct]) or (sample_counts[ct] > n_out/len(celltypes))

