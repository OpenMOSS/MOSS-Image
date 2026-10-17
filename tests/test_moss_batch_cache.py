"""CPU-only tests for portable prompt encoding records."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

from moss_batch import cache  # noqa: E402


def sample_tensors(dtype=torch.float32):
    return {
        "prompt_embeds": torch.arange(24, dtype=dtype).reshape(1, 3, 8),
        "prompt_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.int64),
    }


def sample_metadata():
    return {
        "model": {"id": "OpenMOSS-Team/MOSS-Image-Base", "revision": "pinned-commit"},
        "prompt": "猫咪 🐈\nKeep whitespace.  ",
        "negative_prompt": "",
        "encoding": {"max_sequence_length": 4096, "dtype": "bfloat16", "template": "{prompt}"},
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_round_trip_cpu_unicode_and_negative_prompt(tmp_path, dtype):
    tensors = sample_tensors(dtype)
    tensors["negative_prompt_embeds"] = tensors["prompt_embeds"]  # Shared storage must be supported.
    tensors["negative_prompt_attention_mask"] = tensors["prompt_attention_mask"]
    metadata = sample_metadata()
    directory = tmp_path / "嵌入"
    cache.save_encoding(directory, tensors, metadata)
    loaded, stored = cache.load_encoding(directory, {"model": metadata["model"]})
    assert stored == metadata
    assert "猫咪 🐈" in (directory / cache.METADATA_FILENAME).read_text(encoding="utf-8")
    for key, tensor in loaded.items():
        assert tensor.device.type == "cpu"
        assert tensor.dtype == (dtype if "embeds" in key else torch.int64)
        assert torch.equal(tensor, tensors[key])


@pytest.mark.parametrize("expected", [
    {"model": {"revision": "different"}},
    {"prompt": "猫咪 🐈\nKeep whitespace."},
    {"encoding": {"max_sequence_length": 4096.0}},
    {"tokenizer": "missing"},
])
def test_provenance_mismatches_are_actionable(tmp_path, expected):
    cache.save_encoding(tmp_path / "record", sample_tensors(), sample_metadata())
    with pytest.raises(cache.EncodingCacheError, match="metadata\\."):
        cache.load_encoding(tmp_path / "record", expected)


@pytest.mark.parametrize("expected", [[True], [1.0], [{"x": 1}]])
def test_provenance_list_values_match_exactly(tmp_path, expected):
    actual = [{"x": 1, "extra": 2}] if isinstance(expected[0], dict) else [1]
    cache.save_encoding(tmp_path / "record", sample_tensors(), {"tokens": actual})
    with pytest.raises(cache.EncodingCacheError, match="metadata.tokens"):
        cache.load_encoding(tmp_path / "record", {"tokens": expected})


@pytest.mark.parametrize(("key", "value", "message"), [
    ("prompt_embeds", torch.ones(2, 3, 8), "shape"),
    ("prompt_embeds", torch.ones(1, 0, 8), "shape"),
    ("prompt_embeds", torch.ones(1, 3, 8, dtype=torch.int64), "float16"),
    ("prompt_embeds", torch.full((1, 3, 8), float("nan")), "NaN"),
    ("prompt_embeds", torch.full((1, 3, 8), float("inf")), "NaN"),
    ("prompt_attention_mask", torch.ones(1, 2, dtype=torch.int64), "match"),
    ("prompt_attention_mask", torch.ones(1, 3), "boolean or integer"),
    ("prompt_attention_mask", torch.tensor([[1, 2, 0]]), "only 0/1"),
    ("prompt_attention_mask", torch.zeros(1, 3, dtype=torch.bool), "at least one"),
])
def test_invalid_tensors_do_not_publish(tmp_path, key, value, message):
    tensors = sample_tensors()
    tensors[key] = value
    with pytest.raises(cache.EncodingCacheError, match=message):
        cache.save_encoding(tmp_path / "record", tensors, {})
    assert not (tmp_path / "record").exists()


def test_negative_pair_and_width_are_checked(tmp_path):
    tensors = sample_tensors()
    tensors["negative_prompt_embeds"] = torch.ones(1, 3, 9)
    with pytest.raises(cache.EncodingCacheError, match="together"):
        cache.save_encoding(tmp_path / "record", tensors, {})
    tensors["negative_prompt_attention_mask"] = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(cache.EncodingCacheError, match="hidden size"):
        cache.save_encoding(tmp_path / "record", tensors, {})


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_tensor_names_are_validated(tmp_path, change):
    tensors = sample_tensors()
    if change == "missing":
        del tensors["prompt_attention_mask"]
    else:
        tensors["unexpected_weights"] = torch.ones(1)
    with pytest.raises(cache.EncodingCacheError, match="Encoding requires"):
        cache.save_encoding(tmp_path / "record", tensors, {})


def test_boolean_mask_without_negative_embedding_round_trips(tmp_path):
    tensors = sample_tensors()
    tensors["prompt_attention_mask"] = tensors["prompt_attention_mask"].bool()
    cache.save_encoding(tmp_path / "record", tensors, {})
    loaded, _ = cache.load_encoding(tmp_path / "record")
    assert set(loaded) == set(tensors)
    assert loaded["prompt_attention_mask"].dtype == torch.bool


@pytest.mark.parametrize("existing", ["empty", "corrupt", "valid"])
def test_existing_cache_is_never_replaced(tmp_path, existing):
    directory = tmp_path / "record"
    if existing == "valid":
        cache.save_encoding(directory, sample_tensors(), {"original": True})
    else:
        directory.mkdir()
        if existing == "corrupt":
            (directory / cache.METADATA_FILENAME).write_text("broken", encoding="utf-8")
    before = {entry.name: entry.read_bytes() for entry in directory.iterdir()}
    with pytest.raises(FileExistsError):
        cache.save_encoding(directory, sample_tensors(), {"replacement": True})
    assert {entry.name: entry.read_bytes() for entry in directory.iterdir()} == before


def test_serialization_failure_cleans_temporary_directory(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cache, "save_file", fail)
    with pytest.raises(OSError, match="disk full"):
        cache.save_encoding(tmp_path / "record", sample_tensors(), {})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("metadata", [{"bad": float("nan")}, {"bad": (1, 2)}, {1: "non-string key"}])
def test_metadata_must_round_trip_as_strict_json(tmp_path, metadata):
    with pytest.raises(cache.EncodingCacheError, match="JSON"):
        cache.save_encoding(tmp_path / "record", sample_tensors(), metadata)


@pytest.mark.parametrize("contents", [
    '{"schema_version":999,"metadata":{}}',
    '{"schema_version":true,"metadata":{}}',
    '{"schema_version":1,"metadata":{"x":1,"x":2}}',
    '{"schema_version":1,"metadata":{"x":NaN}}',
    '{"schema_version":1,"metadata":[]}',
    'broken JSON',
])
def test_invalid_metadata_is_rejected(tmp_path, contents):
    directory = tmp_path / "record"
    directory.mkdir()
    (directory / cache.METADATA_FILENAME).write_text(contents, encoding="utf-8")
    with pytest.raises(cache.EncodingCacheError):
        cache.load_encoding(directory)


def test_loading_validates_tensor_values_not_just_format(tmp_path):
    directory = tmp_path / "record"
    cache.save_encoding(directory, sample_tensors(), {})
    tensors = sample_tensors()
    tensors["prompt_embeds"][0, 0, 0] = float("nan")
    save_file(tensors, str(directory / cache.TENSOR_FILENAME))
    with pytest.raises(cache.EncodingCacheError, match="NaN"):
        cache.load_encoding(directory)


def test_missing_or_corrupt_files_report_record_path(tmp_path):
    directory = tmp_path / "record"
    with pytest.raises(cache.EncodingCacheError, match="record"):
        cache.load_encoding(directory)
    directory.mkdir()
    (directory / cache.METADATA_FILENAME).write_text(
        json.dumps({"schema_version": 1, "metadata": {}}), encoding="utf-8"
    )
    with pytest.raises(cache.EncodingCacheError, match="safetensors"):
        cache.load_encoding(directory)
