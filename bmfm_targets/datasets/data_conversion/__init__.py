"""Serializers for Streaming."""

from .serializers import ListStrElementSerializer

_serializers = {list[str]: ListStrElementSerializer}


def get_user_serializer(arg_type, *args, **kwargs):
    try:
        return _serializers[arg_type](*args, **kwargs)
    except:
        raise ValueError(f"No user serializer for type {arg_type}.")
