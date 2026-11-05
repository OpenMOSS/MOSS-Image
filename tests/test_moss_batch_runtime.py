"""Exercise stage orchestration with real cache and output persistence on CPU."""

import json
from fnmatch import fnmatchcase
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from PIL import Image  # noqa: E402

from moss_batch import runtime  # noqa: E402
from moss_batch.cache import EncodingCacheError, load_encoding, save_encoding  # noqa: E402
from moss_batch.jobs import Job  # noqa: E402

IDENTITY = {"kind": "hub", "repo_id": "org/model", "revision": "immutable-commit"}


def make_job(identifier="image", prompt="A cat 猫 🐈", seed=42, **parameters):
    return Job(
        id=identifier, prompt=prompt, negative_prompt="blurred", seed=seed,
        parameters={"height": 32, "width": 16, "max_sequence_length": 8, **parameters},
    )


def persist_encoding(cache_dir, job, negative=True):
    tensors = {
        "prompt_embeds": torch.ones(1, 3, 4, dtype=torch.bfloat16),
        "prompt_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.int64),
    }
    if negative:
        tensors.update({
            "negative_prompt_embeds": torch.zeros(1, 3, 4, dtype=torch.bfloat16),
            "negative_prompt_attention_mask": torch.tensor([[1, 0, 0]], dtype=torch.int64),
        })
    save_encoding(cache_dir / job.id, tensors, runtime.encoding_request(job, IDENTITY))


class FakeScheduler:
    def __init__(self, shift=5.0):
        self.config = {"shift": shift}

    @classmethod
    def from_config(cls, config, shift):
        return cls(shift)


class FakeStage:
    def __init__(self, failure=None):
        self.text_encoder = SimpleNamespace()
        self.tokenizer = SimpleNamespace(chat_template="[USER]{prompt}[/USER]")
        self.scheduler = FakeScheduler()
        self.transformer = SimpleNamespace(config=SimpleNamespace(num_layers=32))
        self.encode_calls = []
        self.generation_calls = []
        self.freed = 0
        self.released = 0
        self.failure = failure

    def encode_prompt(self, prompt, max_sequence_length):
        self.encode_calls.append((prompt, max_sequence_length))
        return torch.ones(1, 3, 4, dtype=torch.bfloat16), torch.ones(1, 3, dtype=torch.int64)

    def maybe_free_model_hooks(self):
        self.freed += 1

    def remove_all_hooks(self):
        self.released += 1

    def __call__(self, **kwargs):
        self.generation_calls.append(kwargs)
        if self.failure:
            raise self.failure
        return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "red")])


def install_stage(monkeypatch, stage):
    loads = []

    def load(source, phase, offload, device):
        loads.append((source, phase, offload, device))
        return stage

    monkeypatch.setattr(runtime, "load_stage", load)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda package: "test-version")
    return loads


def forbid_stage(*args, **kwargs):
    pytest.fail("A completed cached operation must not load model components")


def run_encode(jobs, cache_dir):
    runtime.encode_jobs(jobs, "source", IDENTITY, cache_dir, offload="none", device="cpu", report=lambda _: None)


def run_generate(jobs, cache_dir, output_dir):
    runtime.generate_jobs(
        jobs, "source", IDENTITY, cache_dir, output_dir,
        offload="none", device="cpu", report=lambda _: None,
    )


def test_encode_reuses_nonadjacent_prompt_pair_across_seeds(tmp_path, monkeypatch):
    stage = FakeStage()
    loads = install_stage(monkeypatch, stage)
    jobs = [make_job("first"), make_job("middle", prompt="A dog"), make_job("last", seed=99, guidance_scale=1)]
    run_encode(jobs, tmp_path)
    assert len(loads) == 1
    assert stage.encode_calls == [(jobs[0].prompt, 8), ("blurred", 8), ("A dog", 8), ("blurred", 8)]
    assert stage.released == 1
    for job in jobs:
        tensors, metadata = load_encoding(tmp_path / job.id, runtime.encoding_request(job, IDENTITY))
        assert "negative_prompt_embeds" in tensors
        assert metadata["prompt"] == job.prompt


def test_existing_encoding_skips_loading_any_stage(tmp_path, monkeypatch):
    job = make_job()
    persist_encoding(tmp_path, job)
    monkeypatch.setattr(runtime, "load_stage", forbid_stage)
    run_encode([job], tmp_path)


@pytest.mark.parametrize("operation", ["encode", "generate"])
def test_cache_mismatch_fails_before_loading_model(tmp_path, monkeypatch, operation):
    job = make_job()
    cache_dir = tmp_path / "cache"
    persist_encoding(cache_dir, job)
    changed_job = make_job(prompt="A different prompt")
    monkeypatch.setattr(runtime, "load_stage", forbid_stage)
    with pytest.raises(EncodingCacheError, match="metadata.prompt"):
        if operation == "encode":
            run_encode([changed_job], cache_dir)
        else:
            run_generate([changed_job], cache_dir, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_generation_persists_image_parameters_and_skips_completed_model_load(tmp_path, monkeypatch):
    job = make_job(shift=3.5)
    cache_dir, output_dir = tmp_path / "cache", tmp_path / "output"
    persist_encoding(cache_dir, job)
    stage = FakeStage()
    loads = install_stage(monkeypatch, stage)
    run_generate([job], cache_dir, output_dir)
    assert len(loads) == 1
    assert stage.released == 1
    call = stage.generation_calls[0]
    assert call["generator"].initial_seed() == job.seed
    assert call["num_images_per_prompt"] == 1
    assert "shift" not in call
    assert stage.scheduler.config["shift"] == 3.5
    record_path = output_dir / job.id / "parameters.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["actual_seed"] == job.seed
    assert record["status"] == "complete"
    assert record["job"]["prompt"] == job.prompt
    with Image.open(output_dir / job.id / "01.png") as image:
        assert image.size == (16, 32)
    before = record_path.read_bytes()
    monkeypatch.setattr(runtime, "load_stage", forbid_stage)
    run_generate([job], cache_dir, output_dir)
    assert record_path.read_bytes() == before


def test_failed_generation_reuses_persisted_random_seed(tmp_path, monkeypatch):
    job = make_job(seed=-1)
    cache_dir, output_dir = tmp_path / "cache", tmp_path / "output"
    persist_encoding(cache_dir, job)
    monkeypatch.setattr("moss_batch.jobs.secrets.randbelow", lambda maximum: 123456)
    failing = FakeStage(failure=RuntimeError("simulated interrupted inference"))
    install_stage(monkeypatch, failing)
    with pytest.raises(RuntimeError, match="interrupted inference"):
        run_generate([job], cache_dir, output_dir)
    assert failing.released == 1
    record_path = output_dir / job.id / "parameters.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "pending"
    assert record["actual_seed"] == 123456
    assert not (output_dir / job.id / "01.png").exists()

    def forbid_random(maximum):
        pytest.fail("Resume must reuse the persisted random seed")

    monkeypatch.setattr("moss_batch.jobs.secrets.randbelow", forbid_random)
    succeeding = FakeStage()
    install_stage(monkeypatch, succeeding)
    run_generate([job], cache_dir, output_dir)
    assert failing.generation_calls[0]["generator"].initial_seed() == 123456
    assert succeeding.generation_calls[0]["generator"].initial_seed() == 123456
    assert json.loads(record_path.read_text(encoding="utf-8"))["status"] == "complete"


def test_cfg_zero_requires_negative_branch(tmp_path, monkeypatch):
    job = make_job(guidance_scale=0)
    cache_dir = tmp_path / "cache"
    persist_encoding(cache_dir, job, negative=False)
    stage = FakeStage()
    install_stage(monkeypatch, stage)
    with pytest.raises(ValueError, match="CFG requires negative"):
        run_generate([job], cache_dir, tmp_path / "output")
    assert not stage.generation_calls


def test_cfg_one_can_generate_without_negative_branch(tmp_path, monkeypatch):
    job = make_job(guidance_scale=1)
    cache_dir = tmp_path / "cache"
    persist_encoding(cache_dir, job, negative=False)
    stage = FakeStage()
    install_stage(monkeypatch, stage)
    run_generate([job], cache_dir, tmp_path / "output")
    assert len(stage.generation_calls) == 1
    assert "negative_prompt_embeds" not in stage.generation_calls[0]


@pytest.mark.parametrize("operation", ["encode", "generate"])
def test_mislabeled_cached_dtype_is_rejected_before_model_load(tmp_path, monkeypatch, operation):
    job = make_job()
    cache_dir = tmp_path / "cache"
    tensors = {
        "prompt_embeds": torch.ones(1, 3, 4, dtype=torch.float32),
        "prompt_attention_mask": torch.ones(1, 3, dtype=torch.int64),
        "negative_prompt_embeds": torch.zeros(1, 3, 4, dtype=torch.float32),
        "negative_prompt_attention_mask": torch.ones(1, 3, dtype=torch.int64),
    }
    # The generic format supports float32, but this runtime promises BF16 records.
    metadata = runtime.encoding_request(job, IDENTITY)
    assert metadata["dtype"] == "bfloat16"
    save_encoding(cache_dir / job.id, tensors, metadata)
    monkeypatch.setattr(runtime, "load_stage", forbid_stage)
    with pytest.raises(ValueError, match="recorded bfloat16"):
        if operation == "encode":
            run_encode([job], cache_dir)
        else:
            run_generate([job], cache_dir, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_fresh_float32_encoder_output_is_not_published_as_bfloat16(tmp_path, monkeypatch):
    job = make_job()
    stage = FakeStage()

    def encode_float32(prompt, max_sequence_length):
        return torch.ones(1, 3, 4, dtype=torch.float32), torch.ones(1, 3, dtype=torch.int64)

    stage.encode_prompt = encode_float32
    install_stage(monkeypatch, stage)
    with pytest.raises(ValueError, match="recorded bfloat16"):
        run_encode([job], tmp_path / "cache")
    assert not (tmp_path / "cache" / job.id).exists()
    assert stage.released == 1


@pytest.mark.parametrize("phase", ["encode", "generate"])
def test_hub_resolution_downloads_only_needed_stage_and_pins_resolved_revision(tmp_path, monkeypatch, phase):
    snapshot = tmp_path / "snapshots" / "resolved-immutable-commit"
    snapshot.mkdir(parents=True)
    (snapshot / "pipeline_moss_image.py").write_text("# fixture\n", encoding="utf-8")
    downloads = []

    def snapshot_download(repo_id, **kwargs):
        downloads.append((repo_id, kwargs))
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    root, identity = runtime.resolve_model(
        "org/model", phase, revision="release-tag", local_files_only=True,
    )
    assert root == snapshot.resolve()
    assert identity == {"kind": "hub", "repo_id": "org/model", "revision": snapshot.name}
    assert downloads[0][0] == "org/model"
    options = downloads[0][1]
    assert options["revision"] == "release-tag"
    assert options["local_files_only"] is True
    patterns = options["allow_patterns"]

    def downloaded(filename):
        return any(fnmatchcase(filename, pattern) for pattern in patterns)

    assert downloaded("pipeline_moss_image.py")
    assert downloaded("model_index.json")
    encoding_files = ["text_encoder/model-00001-of-00003.safetensors", "tokenizer/tokenizer.json"]
    generation_files = [
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "vae/diffusion_pytorch_model.safetensors", "scheduler/scheduler_config.json",
    ]
    needed, omitted = (encoding_files, generation_files) if phase == "encode" else (
        generation_files, encoding_files,
    )
    assert all(downloaded(filename) for filename in needed)
    assert not any(downloaded(filename) for filename in omitted)


def test_local_model_requires_explicit_portable_version(tmp_path, monkeypatch):
    (tmp_path / "pipeline_moss_image.py").write_text("# fixture\n", encoding="utf-8")

    def forbid_download(*args, **kwargs):
        pytest.fail("Resolving a local bundle must not contact the Hub")

    monkeypatch.setattr("huggingface_hub.snapshot_download", forbid_download)
    with pytest.raises(ValueError, match="local-model-version"):
        runtime.resolve_model(str(tmp_path), "encode")


def test_local_identity_survives_copying_bundle_to_another_path(tmp_path, monkeypatch):
    def forbid_download(*args, **kwargs):
        pytest.fail("Resolving a local bundle must not contact the Hub")

    monkeypatch.setattr("huggingface_hub.snapshot_download", forbid_download)
    identities = []
    for machine, phase in (("encoding-machine", "encode"), ("generation-machine", "generate")):
        bundle = tmp_path / machine
        bundle.mkdir()
        (bundle / "pipeline_moss_image.py").write_text("# fixture\n", encoding="utf-8")
        root, identity = runtime.resolve_model(str(bundle), phase, local_model_version="bundle-v1")
        assert root == bundle.resolve()
        identities.append(identity)
    assert identities[0] == identities[1] == {"kind": "local", "version": "bundle-v1"}


@pytest.mark.parametrize("kind", ["local", "hub"])
def test_missing_pipeline_source_is_rejected(tmp_path, monkeypatch, kind):
    snapshot = tmp_path / "incomplete-bundle"
    snapshot.mkdir()
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: str(snapshot))
    model = str(snapshot) if kind == "local" else "org/model"
    version = "bundle-v1" if kind == "local" else None
    with pytest.raises(ValueError, match="pipeline_moss_image.py"):
        runtime.resolve_model(model, "generate", local_model_version=version)
