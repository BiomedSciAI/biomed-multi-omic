import matplotlib
import matplotlib.pyplot as plt
import scanpy as sc

from bmfm_targets.datasets import quality_control
from bmfm_targets.tests import helpers


def test_qc_plots():
    source_h5ad_file_name = helpers.MockTestDataPaths.root / "h5ad" / "mock_test_data.h5ad"
    cell_type_label = "celltype"

    adata = sc.read_h5ad(source_h5ad_file_name)
    adata = quality_control.assign_mito_qc_metrics(adata)
    matplotlib.use("pdf")

    quality_control.show_qc_plots(adata, cell_type_label=cell_type_label)
    plt.close()
