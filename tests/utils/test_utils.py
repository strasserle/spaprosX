import pytest
import pandas as pd
from spapros.util.util import sample_cells


@pytest.mark.parametrize("obs_key", ["celltype", ["celltype"], ["tissue", "celltype"]])
@pytest.mark.parametrize("drop_rare_groups", [True, False])
def test_sample_cells_one_key(tiny_adata, obs_key, drop_rare_groups):
    n_out = 50
    adata = sample_cells(tiny_adata,
                 n_out=n_out,
                 obs_key=obs_key,
                 sampling_seed=0,
                 drop_rare_groups=drop_rare_groups,
                 copy=True)
    tiny_adata = tiny_adata.copy()
    if isinstance(obs_key, list):
        obs_keys_replaced = [x.replace("_", "-") for x in obs_key]
        key_added = "_".join(obs_keys_replaced)
        adata.obs[key_added] = adata.obs[obs_key[0]]
        tiny_adata.obs[key_added] = tiny_adata.obs[obs_key[0]]
        for k in obs_key[1:]:
            adata.obs[key_added] = adata.obs[key_added].astype(str) + "_" + \
                                   adata.obs[k].astype(str).str.replace("_", "-")
            tiny_adata.obs[key_added] = tiny_adata.obs[key_added].astype(str) + "_" + \
                                   tiny_adata.obs[k].astype(str).str.replace("_", "-")
    else:
        key_added = obs_key
    groups = tiny_adata.obs[key_added].unique()
    group_counts = tiny_adata.obs[key_added].value_counts()
    sample_counts = adata.obs[key_added].value_counts()
    sampled_groups = adata.obs[key_added].unique()
    assert adata.shape[0] == n_out
    assert sample_counts.sum() == n_out
    for group in groups:
        if group in sampled_groups:
            assert (sample_counts[group] == group_counts[group]) or \
                   (sample_counts[group] >= (n_out // len(groups)))
            if drop_rare_groups:
                assert sample_counts[group] in [n_out // len(sampled_groups), n_out // len(sampled_groups) + 1]
        elif group not in sampled_groups:
            assert drop_rare_groups

