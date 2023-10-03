import pytest
from matplotlib.testing.compare import compare_images
from PIL import Image


#############
# selection #
#############

@pytest.mark.skip(reason="plotting")
@pytest.mark.parametrize(
    "fun, kwargs, fig_id", [

        # plot_histogram -> selection_histogram:
        ("plot_histogram", {
            "x_axis_keys": None,
            "selections": None,
            "penalty_keys": None,
            "unapplied_penalty_keys": None,
            "background_key": None,
        }, "default"),
        ("plot_histogram", {
            "x_axis_keys": {"expression_penalty_upper": "quantile_0.99",
                            "expression_penalty_lower": "quantile_0.9 expr > 0",
                            "marker": "quantile_0.99"},
            "selections": ["marker"],
            "penalty_keys": {"marker": []},
            "unapplied_penalty_keys": {"marker": []},
            "background_key": True
        }, "marker_w_bg"),
        ("plot_histogram", {
            "x_axis_keys": {"expression_penalty_upper": "quantile_0.99",
                            "expression_penalty_lower": "quantile_0.9 expr > 0",
                            "marker": "quantile_0.99"},
            "selections": ["marker"],
            "penalty_keys": {"marker": None},
            "unapplied_penalty_keys": {"marker": None},
            "background_key": None
        }, "marker_w_penal"),
        ("plot_histogram", {
            "x_axis_keys": {"expression_penalty_upper": "quantile_0.99",
                            "expression_penalty_lower": "quantile_0.9 expr > 0",
                            "marker": "quantile_0.99"},
            "selections": ["marker"],
            "penalty_keys": {"marker": []},
            "unapplied_penalty_keys": {"marker": []},
            "background_key": "all"
        }, "marker_w_bg_all"),

        # plot_coexpression -> correlation_matrix
        ("plot_coexpression", {
            "selections": None,
            "n_cols": 3,
            "scale": True
        }, "n_cols_3_scaled"),
        ("plot_coexpression", {
            "selections": None,
            "n_cols": 1,
            "scale": False
        }, "n_cols_1_unscaled"),
        ("plot_coexpression", {
            "selections": ["marker"],
            "colorbar": False
        }, "marker_wo_cbar"),

        # plot_clf_genes -> clf_genes_umaps
        ("plot_clf_genes", {
            "till_rank": 5,
            "importance_th": 0.3,
            "size_factor": 0.5,
            "fontsize": 10,
        }, "till_rank_5_imp_th_03"),

        ("plot_clf_genes", {
            "till_rank": 3,
            "importance_th": 0.3,
            "n_cols": 3,
        }, "till_rank_3_imp_th_03_n_cols_3"),

        # overlap:
        ("plot_gene_overlap", {"style": "venn"}, "venn"),
        ("plot_gene_overlap", {"style": "upset"}, "upset"),

        # dotplot:
        ("plot_masked_dotplot", {}, "default"),
        ("plot_masked_dotplot", {"comb_markers_only": True,
                                 "markers_only": True,
                                 "n_genes": 10}, "top10_markers"),
    ]
)
def test_selection_plots(selector_with_penalties, fun, tmp_path, kwargs, fig_id):
    ref_name = f"tests/plotting/test_data/selection_{fun}_{fig_id}.png"
    fig_name = f"{tmp_path}/selection_{fun}_{fig_id}.png"
    getattr(selector_with_penalties, fun)(save=fig_name, show=True, **kwargs)
    # getattr(selector_with_penalties, fun)(save=ref_name, show=False, **kwargs)
    ref_image = Image.open(ref_name)
    fig_image = Image.open(fig_name)
    # sometimes the image sizes change by 1 pixel, so we resize
    if ref_image.size != fig_image.size:
        fig_image = fig_image.resize(ref_image.size)
        fig_image.save(fig_name)
    assert compare_images(ref_name, fig_name, 0.01) is None


##############
# evaluation #
##############

@pytest.mark.skip(reason="plotting")
@pytest.mark.parametrize("ev, ref_name", [
                         # ("evaluator", "tests/plotting/test_data/plot_summary.png"),
                          ("evaluator_X", "tests/plotting/test_data/plot_summary_X.png")])
def test_plot_summary(ev, ref_name, tmp_path, request):
    fig_name = f"{tmp_path}/tmp_plot_summary.png"
    ev = request.getfixturevalue(ev)
    probesets = ev.summary_results.index.sort_values()
    ev.plot_summary(show=False, save=fig_name, set_ids=probesets)
    # ev.plot_summary(show=False, save=ref_name, set_ids=probesets)
    ref_image = Image.open(ref_name)
    fig_image = Image.open(fig_name)
    # sometimes the image sizes change by 1 pixel, so we resize
    if ref_image.size != fig_image.size:
        fig_image = fig_image.resize(ref_image.size)
        fig_image.save(fig_name)
    assert compare_images(ref_name, fig_name, 0.01) is None


@pytest.mark.skip(reason="plotting")
@pytest.mark.parametrize(
    "fun, kwargs",
    [
        ("plot_confusion_matrix", {"kwargs_name": "kwargs1"}),
        ("plot_coexpression", {"kwargs_name": "kwargs1"}),
        ("plot_cluster_similarity", {"kwargs_name": "kwargs1"}),

        # ev.plot_knn_overlap --> pl.knn_overlap
        ("plot_knn_overlap", {
            "kwargs_name": "kwargs1",
            "set_ids": ["ref_DE", "ref_PCA", "spapros_selection"],
            "selections_info": None
        }),
        ("plot_knn_overlap", {
            "kwargs_name": "kwargs2",
            "set_ids": None,
            "selections_info": "selections_info_1"
        }),
        ("plot_knn_overlap", {
            "kwargs_name": "kwargs3",
            "set_ids": None,
            "selections_info": "selections_info_2"
        }),
    ],
)
@pytest.mark.skip(reason="plotting")
def test_evaluation_plots(evaluator_4_sets, fun, tmp_path, kwargs, request):
    ref_name = f"tests/plotting/test_data/evaluation_{fun}_{kwargs['kwargs_name']}.png"
    fig_name = f"{tmp_path}/evaluations_{fun}_{kwargs['kwargs_name']}.png"
    if "selections_info" in kwargs:
        if kwargs["selections_info"] is not None:
            kwargs["selections_info"] = request.getfixturevalue(kwargs["selections_info"])
    del kwargs["kwargs_name"]
    getattr(evaluator_4_sets, fun)(save=fig_name, show=False, **kwargs)
    # getattr(evaluator_4_sets, fun)(save=ref_name, show=False, **kwargs)
    ref_image = Image.open(ref_name)
    fig_image = Image.open(fig_name)
    # sometimes the image sizes change by 1 pixel, so we resize
    if ref_image.size != fig_image.size:
        fig_image = fig_image.resize(ref_image.size)
        fig_image.save(fig_name)
    assert compare_images(ref_name, fig_name, 0.01) is None
