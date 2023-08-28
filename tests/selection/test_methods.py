import random
import pandas as pd
import pytest

from spapros import se
from spapros.selection import select_pca_genes
import scanpy as sc
from spapros.selection.selection_methods import van_elteren_test
import numpy as np


@pytest.mark.parametrize("genes_key", ["highly_variable", None])
@pytest.mark.parametrize("n", [10, 50])
@pytest.mark.parametrize("preselected_genes", [["IL7R", "CD14", "NKG7"], ["IL7R", "CD14"]])
@pytest.mark.parametrize("prior_genes", [["LST1", "CST3"], ["CD14", "NKG7"]])
@pytest.mark.parametrize("n_pca_genes", [100])
def test_selection_params(
    adata_pbmc3k,
    genes_key,
    n,
    preselected_genes,
    prior_genes,
    n_pca_genes,
):
    random.seed(0)
    selector = se.ProbesetSelector(
        adata_pbmc3k[random.sample(range(adata_pbmc3k.n_obs), 100), :],
        genes_key=genes_key,
        n=n,
        celltype_key="celltype",
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
        preselected_genes=preselected_genes,
        prior_genes=prior_genes,
        n_pca_genes=n_pca_genes,
        save_dir=None,
    )
    selector.select_probeset()


@pytest.mark.parametrize("celltypes", (["CD4 T cells"], ["Megakaryocytes"]))
@pytest.mark.parametrize(
    "marker_list",
    [
        "evaluation/selection/test_data/pbmc3k_marker_list.csv",
        {
            "CD4 T cells": ["IL7R"],
            "CD14+ Monocytes": ["CD14", "LYZ"],
            "B cells": ["MS4A1"],
            "CD8 T cells": ["CD8A"],
            "NK cells": ["GNLY", "NKG7"],
            "FCGR3A+ Monocytes": ["FCGR3A", "MS4A7"],
            "3	Dendritic Cells": ["FCER1A", "CST3"],
            "Megakaryocytes": ["NAPA-AS1", "PPBP"],
        },
    ],
)
def test_selection_celltypes(adata_pbmc3k, celltypes, marker_list):
    selector = se.ProbesetSelector(
        adata_pbmc3k,  # [random.sample(range(adata_pbmc3k.n_obs), 100), :],
        genes_key="highly_variable",
        n=50,
        celltype_key="celltype",
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
        save_dir=None,
        celltypes=celltypes,
    )
    selector.select_probeset()


@pytest.mark.parametrize("save_dir", [None, "tmp_path"])
@pytest.mark.parametrize("verbosity", [0, 1, 2])
def test_selection_verbosity(
    adata_pbmc3k,
    verbosity,
    save_dir,
    request,
):
    random.seed(0)
    selector = se.ProbesetSelector(
        adata_pbmc3k[random.sample(range(adata_pbmc3k.n_obs), 100), :],
        celltype_key="celltype",
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
        verbosity=verbosity,
        save_dir=None if not save_dir else request.getfixturevalue(save_dir),
    )
    selector.select_probeset()


def test_selection_stable(adata_pbmc3k):
    random.seed(0)
    idx = random.sample(range(adata_pbmc3k.n_obs), 100)
    selector_a = se.ProbesetSelector(
        adata_pbmc3k[idx, :],
        verbosity=2,
        n=50,
        celltype_key="celltype",
        seed=0,
        save_dir=None,
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
    )
    selector_a.select_probeset()
    selection1 = selector_a.probeset.copy()
    selector_a.select_probeset()
    selection2 = selector_a.probeset.copy()
    selector_b = se.ProbesetSelector(
        adata_pbmc3k[idx, :],
        verbosity=2,
        n=50,
        celltype_key="celltype",
        seed=0,
        save_dir=None,
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
    )
    selector_b.select_probeset()
    selection3 = selector_b.probeset.copy()
    assert pd.testing.assert_frame_equal(selection1, selection2) is None
    assert pd.testing.assert_frame_equal(selection2, selection3) is None


@pytest.mark.parametrize(
    "n, " "genes_key, " "seeds, " "verbosity, " "save_dir, " "methods",
    [
        (
            10,
            "highly_variable",
            [0, 202],
            0,
            None,
            {
                "hvg_selection": {"flavor": "cell_ranger"},
                "random_selection": {},
                "pca_selection": {
                    "variance_scaled": False,
                    "absolute": True,
                    "n_pcs": 20,
                    "penalty_keys": [],
                    "corr_penalty": None,
                },
                "DE_selection": {"per_group": "True"},
            },
        ),
        (
            50,
            "highly_variable",
            [],
            1,
            None,
            {
                "hvg_selection": {"flavor": "seurat"},
                "pca_selection": {
                    "variance_scaled": True,
                    "absolute": False,
                    "n_pcs": 10,
                },
                "DE_selection": {"per_group": "False"},
            },
        ),
        (100, "highly_variable", [], 2, "tmp_path", ["PCA", "DE", "HVG", "random"]),
    ],
)
def test_select_reference_probesets(
    adata_pbmc3k, n, genes_key, seeds, verbosity, save_dir, request, methods
):
    se.select_reference_probesets(
        adata_pbmc3k,
        n=n,
        genes_key=genes_key,
        seeds=seeds,
        verbosity=verbosity,
        save_dir=None if not save_dir else request.getfixturevalue(save_dir),
        methods=methods,
    )

##################################
## test for batch aware methods ##
##################################

def test_select_pca_genes_per_batch(tiny_adata_w_penalties):
    a = tiny_adata_w_penalties
    a.obs["batch"] = ["batch_1", "batch_2"] * (a.shape[0]//2)
    a.obs["one_batch"] = ["one_batch"] * a.shape[0]

    selection_df_X_batch = select_pca_genes(
        a,
        n=50,
        penalty_keys=["expression_penalty_upper", "expression_penalty_lower"],
        batch_aware=True,
        batch_key="batch",
        corr_penalty=None,
        inplace=False,
    )

    selection_df_one_batch = select_pca_genes(
        a,
        n=50,
        penalty_keys=["expression_penalty_upper", "expression_penalty_lower"],
        batch_aware=True,
        batch_key="one_batch",
        corr_penalty=None,
        inplace=False,
    )

    selection_df = select_pca_genes(
        a,
        n=50,
        penalty_keys=["expression_penalty_upper", "expression_penalty_lower"],
        batch_aware=False,
        batch_key=None,
        corr_penalty=None,
        inplace=False,
    )

    assert selection_df.shape == selection_df_X_batch.shape
    assert pd.testing.assert_frame_equal(selection_df_one_batch, selection_df) is None


def test_van_elteren_test_equals_rank_genes_groups(adata):
    # for only one batch, both tests should be equal
    adata = adata.copy()
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.log1p(adata)
    adata.obs["batch"] = "single_batch"
    sc.tl.rank_genes_groups(adata, groupby="cell_type", method="wilcoxon")
    van_elteren_test(adata, ct_key="cell_type", batch_key="batch", groups="all", reference="rest")
    for key in list(adata.uns["rank_genes_groups"].keys())[1:]:
        print(key)
        for results_a, results_b in zip(adata.uns["rank_genes_groups"][key], adata.uns["rank_genes_groups_stratified"][
            key]):
            for value_a, value_b in zip(results_a, results_b):
                if type(value_a) in [np.float32, np.float64] and type(value_b) in [np.float32, np.float64]:
                    np.testing.assert_almost_equal(value_a, value_b)
                    print(value_a, " equals ", value_b)
                elif type(value_a) == str and type(value_b) == str:
                    assert value_a == value_b
                    print(value_a, " equals ", value_b)
                else:
                    raise ValueError("Unexpected dtype")


def test_van_elteren_with_missing_cts():

    # ct3 missing in batch2 --> no problem for one vs all (because no ct lost)
    adata = sc.AnnData(np.random.randint(0, 100, (100, 30)))
    sc.pp.log1p(adata)
    adata.obs["batch"] = ["batch1"] * 50 + ["batch2"] * 50
    adata.obs["cell_type"] = ["ct1"] * 25 + ["ct2"] * 20 + ["ct3"] * 5 + ["ct1"] * 25 + ["ct2"] * 25
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    van_elteren_test(adata, ct_key="cell_type", batch_key="batch", groups="all", reference="rest")
    res = adata.uns["rank_genes_groups_stratified"]
    assert all(res["n_batches"].ct1 == 2)
    assert all(res["n_batches"].ct2 == 2)
    assert all(res["n_batches"].ct3 == 1)

    # ct3 missing in batch2 --> still no problem for one vs ct3 (because batch 2 lost but no ct)
    adata = sc.AnnData(np.random.randint(0, 100, (100, 30)))
    sc.pp.log1p(adata)
    adata.obs["batch"] = ["batch1"] * 50 + ["batch2"] * 50
    adata.obs["cell_type"] = ["ct1"] * 25 + ["ct2"] * 20 + ["ct3"] * 5 + ["ct1"] * 25 + ["ct2"] * 25
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    van_elteren_test(adata, ct_key="cell_type", batch_key="batch", groups="all", reference="ct3")
    res = adata.uns["rank_genes_groups_stratified"]
    assert all(res["n_batches"].ct1 == 1)
    assert all(res["n_batches"].ct2 == 1)

    # ct3 missing in batch2 and ct1 missing in batch1 --> problem for ct1 vs ct3 --> additional
    # ct1 vs ct3 in batch 1 skipped (no ct1), in batch 2 skipped (no ct3) --> w_stats[ct1] empty --> case 1
    # ct2 vs ct3 in batch 1 okay (vs 3), in batch 2 skipped (no ct3) --> w_stats[ct2] 1 batch
    # ct3 vs ct3 in batch 1 --> skipped, not asked
    # --> c1 never target --> additional non-batch-aware DE test (case 1)
    adata = sc.AnnData(np.random.randint(0, 100, (100, 30)))
    sc.pp.log1p(adata)
    adata.obs["batch"] = ["batch1"] * 50 + ["batch2"] * 50
    adata.obs["cell_type"] = ["ct1"] * 0 + ["ct2"] * 45 + ["ct3"] * 5 + ["ct1"] * 25 + ["ct2"] * 25
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    van_elteren_test(adata, ct_key="cell_type", batch_key="batch", groups="all", reference="ct3")
    res = adata.uns["rank_genes_groups_stratified"]
    assert all(res["n_batches"].ct1 == "non-batch-aware_case-1")
    assert all(res["n_batches"].ct2 == 1)

    # ct3 missing in batch2 and ct3 is the target --> no problem because all reference cts also in batch 1
    # ct3 vs rest in batch 1 okay (vs 1, 2), in batch 2 skipped (no ct3) --> w_stats[ct3] 1 batch
    # no other target cts requested
    adata = sc.AnnData(np.random.randint(0, 100, (100, 30)))
    sc.pp.log1p(adata)
    adata.obs["batch"] = ["batch1"] * 50 + ["batch2"] * 50
    adata.obs["cell_type"] = ["ct1"] * 5 + ["ct2"] * 40 + ["ct3"] * 5 + ["ct1"] * 25 + ["ct2"] * 25
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    van_elteren_test(adata, ct_key="cell_type", batch_key="batch", groups=["ct3"], reference="rest")
    res = adata.uns["rank_genes_groups_stratified"]
    assert all(res["n_batches"].ct3 == 1)

    # ct3 is the target, only present in batch 1, ct1 only present in batch 2 --> problem for ct3 vs ct1 --> case 2
    # ct3 vs rest in batch 1 okay (vs 2), in batch 2 skipped (no ct3) --> w_stats[ct3] 1 batch
    # ct 1 never reference --> additional non-batch-aware DE test (case 2)
    adata = sc.AnnData(np.random.randint(0, 100, (100, 30)))
    sc.pp.log1p(adata)
    adata.obs["batch"] = ["batch1"] * 50 + ["batch2"] * 50
    adata.obs["cell_type"] = ["ct1"] * 0 + ["ct2"] * 40 + ["ct3"] * 10 + ["ct1"] * 25 + ["ct2"] * 25
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    van_elteren_test(adata, ct_key="cell_type", batch_key="batch", groups=["ct3"], reference="rest")
    res = adata.uns["rank_genes_groups_stratified"]
    assert all(res["n_batches"].ct3[[x for x in range(0, 30, 2)]] == 1)
    assert all(res["n_batches"].ct3[[x for x in range(1, 30, 2)]] == "non-batch-aware_case-2")

    # multiple target cts
    # never ct1 vs ct4 pos --> additionally test [ct1, ct2, ct3, ct4] vs [ct1, ct 4] --> merge for each cell type
    adata = sc.AnnData(np.random.randint(0, 100, (100, 30)))
    sc.pp.log1p(adata)
    adata.obs["batch"] = ["batch1"] * 50 + ["batch2"] * 50
    adata.obs["cell_type"] = ["ct1"] *  0 + ["ct2"] * 30 + ["ct3"] * 10 + ["ct4"] * 10 + \
                             ["ct1"] * 25 + ["ct2"] * 15 + ["ct3"] * 10 + ["ct4"] * 0
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    van_elteren_test(adata, ct_key="cell_type", batch_key="batch", groups='all', reference="rest")
    res = adata.uns["rank_genes_groups_stratified"]
    assert all(res["n_batches"].ct1[[x for x in range(0, 30, 2)]] == 1)
    assert all(res["n_batches"].ct1[[x for x in range(1, 30, 2)]] == "non-batch-aware_case-2")
    assert all(res["n_batches"].ct4[[x for x in range(0, 30, 2)]] == 1)
    assert all(res["n_batches"].ct4[[x for x in range(1, 30, 2)]] == "non-batch-aware_case-2")
    assert all(res["n_batches"].ct2 == 2)
    assert all(res["n_batches"].ct3 == 2)

#
# def test_van_elteren_test_liver_real():
#     adata_path = "/big/st/strasserl/MA/benchmarking/data/test_data/liver-real.h5ad"
#     batch_key = "development_stage"
#     adata = sc.read_h5ad(os.path.join("..", adata_path)) # real dataset
#     van_elteren_test(adata, ct_key="cell_type", batch_key=batch_key, groups="all", reference="rest")










