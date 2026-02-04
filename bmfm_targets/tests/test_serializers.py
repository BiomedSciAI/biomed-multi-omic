from bmfm_targets.datasets.data_conversion import (
    ListStrElementSerializer,
    get_user_serializer,
)


def test_list_str_serializer():
    serializer: ListStrElementSerializer = get_user_serializer(list[str])
    orig_data = ["aaaa", "fdfdfd", "cccc", "v"]
    encoded_data = serializer.serialize(orig_data)
    assert type(encoded_data) == bytes
    decoded_data = serializer.deserialize(encoded_data)
    assert orig_data == decoded_data
