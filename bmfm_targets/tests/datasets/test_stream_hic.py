import pytest

from bmfm_targets.datasets.SNPdb.streaming_snp_dataset import StreamingHiCDataModule


@pytest.mark.usefixtures("_convert_hic_raw_to_lit")
def test_init(streaming_hic_parameters):
    datamodule = StreamingHiCDataModule(**streaming_hic_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    assert datamodule is not None


# @pytest.mark.usefixtures("_convert_hic_raw_to_lit")
# def test_train_dataloader(streaming_hic_parameters):
#    datamodule = StreamingHiCDataModule(**streaming_hic_parameters)
#    datamodule.prepare_data()
#    datamodule.setup()
#    for batch in datamodule.train_dataloader():
#        assert tuple(batch["attention_mask"].squeeze().shape) == (10, 512)
#        assert tuple(batch["input_ids"].squeeze().shape) == (10, 512)
#        assert tuple(batch["labels"]["dna_chunks"].shape) == (10, 512)
#        assert tuple(batch["labels"]["hic_contact"].shape) == (10,)
#        break
#
#
# @pytest.mark.usefixtures("_convert_hic_raw_to_lit")
# def test_val_dataloader(streaming_hic_parameters):
#    datamodule = StreamingHiCDataModule(**streaming_hic_parameters)
#    datamodule.prepare_data()
#    datamodule.setup()
#    for batch in datamodule.val_dataloader():
#        assert tuple(batch["attention_mask"].squeeze().shape) == (10, 512)
#        assert tuple(batch["input_ids"].squeeze().shape) == (10, 512)
#        assert tuple(batch["labels"]["dna_chunks"].shape) == (10, 512)
#        assert tuple(batch["labels"]["hic_contact"].shape) == (10,)
#        break


@pytest.mark.usefixtures("_convert_hic_raw_to_lit")
def test_reading_content(streaming_hic_parameters):
    datamodule = StreamingHiCDataModule(**streaming_hic_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    # 70 = 100 samples * 0.7
    num_records = 70

    train_dataloader = datamodule.train_dataloader()
    n = 0
    for i in train_dataloader:
        n += i["input_ids"].shape[0]
    assert n == num_records


@pytest.mark.usefixtures("_convert_hic_raw_to_lit")
def test_restart(streaming_hic_parameters):
    datamodule = StreamingHiCDataModule(**streaming_hic_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()

    num_records = 7

    record = iter(train_dataloader)

    n = 0
    for i in range(num_records // 2):
        next(record)
        n += 1

    state = datamodule.get_train_dataloader_state()
    datamodule = StreamingHiCDataModule(**streaming_hic_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    train_dataloader.load_state_dict(state)

    for i in range(num_records // 2, num_records):
        next(record)
        n += 1

    assert n == num_records
