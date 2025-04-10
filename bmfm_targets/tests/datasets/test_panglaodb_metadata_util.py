import os

from bmfm_targets.datasets.panglaodb import panglaodb_metadata_util
from bmfm_targets.tests.helpers import PanglaoPaths


def test_convert_panglao_metadata_to_csv():
    file_to_column_names = {
        "metadata_test_util.txt": [
            "SRA Accession",
            "SRS Accession",
            "Tissue Origin",
            "scRNA-seq Protocol",
            "Species",
            "Sequencing Instrument",
            "Number of Expressed Genes",
            "Median Expressed Genes per Cell",
            "Number of Cell Clusters",
            "Is Tumor Sample",
            "Is Primary Adult Tissue Sample",
            "Is Cell Line Sample",
        ]
    }
    input_dir = PanglaoPaths.all_metadata.parent

    panglaodb_metadata_util.convert_panglao_metadata_to_csv(
        file_to_column_names=file_to_column_names, input_dir=input_dir
    )
    assert os.path.exists(input_dir / "metadata_test_util.csv")
    os.remove(input_dir / "metadata_test_util.csv")
