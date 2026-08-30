# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("nemo_automodel")

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine.automodel import transformer_impl
from verl.workers.engine.automodel.transformer_impl import AutomodelEngine, AutomodelEngineWithLMHead


class _RecordingTrainingEngine:
    """Stand-in for nemo_automodel.engine.Engine with the DeepSpeed-style surface."""

    def __init__(self):
        self.gas_calls = []
        self.backward_calls = []
        self.step_calls = 0

    def __call__(self, **model_inputs):
        return model_inputs["input_ids"].sum()

    def set_gradient_accumulation_steps(self, steps):
        self.gas_calls.append(steps)

    def backward(self, loss, retain_graph=False, scale_wrt_gas=True):
        self.backward_calls.append((loss.detach(), scale_wrt_gas))

    def step(self):
        self.step_calls += 1


def _microbatch(token: int) -> TensorDict:
    return TensorDict({"token": torch.tensor([token])}, batch_size=[1])


def _packed_microbatch() -> TensorDict:
    input_ids = torch.nested.as_nested_tensor(
        [torch.tensor([10, 11, 12]), torch.tensor([20, 21])],
        layout=torch.jagged,
    )
    position_ids = torch.nested.as_nested_tensor(
        [torch.arange(3), torch.arange(2)],
        layout=torch.jagged,
    )
    loss_mask = torch.nested.as_nested_tensor(
        [torch.ones(3), torch.ones(2)],
        layout=torch.jagged,
    )
    data = TensorDict(
        {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "temperature": torch.ones(2),
        },
        batch_size=[2],
    )
    return tu.assign_non_tensor(
        data,
        use_remove_padding=True,
        use_fused_kernels=False,
        pad_mode=DatasetPadMode.NO_PADDING,
    )


@pytest.mark.parametrize(
    ("forward_only", "return_model_output", "keeps_model_output"),
    [
        (True, False, True),
        (False, False, False),
        (False, True, True),
    ],
)
def test_forward_backward_drives_one_engine_microstep_per_microbatch(
    monkeypatch,
    forward_only,
    return_model_output,
    keeps_model_output,
):
    microbatches = [_microbatch(2), _microbatch(5)]
    for microbatch in microbatches:
        tu.assign_non_tensor(microbatch, return_model_output=return_model_output)
    data = TensorDict({"loss_mask": torch.ones(2)}, batch_size=[2])
    training_engine = _RecordingTrainingEngine()

    engine = object.__new__(AutomodelEngineWithLMHead)
    engine.training_engine = training_engine
    engine.prepare_model_inputs = lambda micro_batch: (
        {"input_ids": micro_batch["token"].reshape(1, 1)},
        {},
    )
    engine.prepare_model_outputs = lambda raw_output, _args, _micro_batch: {"value": raw_output}
    engine.get_data_parallel_group = lambda: None
    engine.get_data_parallel_size = lambda: 4

    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(transformer_impl, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        transformer_impl,
        "prepare_micro_batches",
        lambda **kwargs: (microbatches, torch.tensor([1, 0])),
    )
    monkeypatch.setattr(
        transformer_impl,
        "postprocess_batch_func",
        lambda output_lst, indices, data: (output_lst, indices),
    )

    seen_tokens = []

    def loss_fn(model_output, data, dp_group):
        seen_tokens.append(int(data["token"].item()))
        return model_output["value"].float() + 8, {"token": int(data["token"].item())}

    outputs, indices = AutomodelEngine.forward_backward_batch(
        engine,
        data,
        loss_fn,
        forward_only=forward_only,
    )

    assert seen_tokens == [2, 5]
    if forward_only:
        # Scoring never touches the training state machine.
        assert training_engine.gas_calls == []
        assert training_engine.backward_calls == []
        assert training_engine.step_calls == 0
    else:
        # One accumulation window per veRL mini-batch; every microbatch is one
        # engine microstep with veRL's own normalization passed through raw.
        assert training_engine.gas_calls == [2]
        assert [loss.item() for loss, _ in training_engine.backward_calls] == pytest.approx([10.0, 13.0])
        assert all(scale_wrt_gas is False for _, scale_wrt_gas in training_engine.backward_calls)
        assert training_engine.step_calls == 2
    assert [record["loss"] for record in outputs] == pytest.approx([10.0, 13.0])
    assert [("model_output" in record) for record in outputs] == [keeps_model_output, keeps_model_output]
    assert torch.equal(indices, torch.tensor([1, 0]))


def test_optimizer_and_scheduler_stay_on_their_owners():
    class Optimizer:
        param_groups = [{"lr": 0.25}]

        def __init__(self):
            self.zero_grad_calls = 0

        def zero_grad(self):
            self.zero_grad_calls += 1

    class Scheduler:
        def __init__(self):
            self.increments = []

        def step(self, increment):
            self.increments.append(increment)

    class TrainingEngine:
        def __init__(self):
            self.norm_reads = 0

        def get_global_grad_norm(self):
            self.norm_reads += 1
            return torch.tensor(3.5)

    engine = object.__new__(AutomodelEngine)
    engine.optimizer = Optimizer()
    engine.lr_scheduler = Scheduler()
    engine.training_engine = TrainingEngine()

    AutomodelEngine.optimizer_zero_grad(engine)
    assert AutomodelEngine.optimizer_step(engine) == 3.5
    assert AutomodelEngine.lr_scheduler_step(engine) == 0.25
    assert engine.optimizer.zero_grad_calls == 1
    assert engine.training_engine.norm_reads == 1
    assert engine.lr_scheduler.increments == [1]


def test_initialize_keeps_scheduler_out_of_training_engine(monkeypatch):
    module = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    scheduler = object()
    captured = {}

    engine = object.__new__(AutomodelEngine)
    engine.engine_config = SimpleNamespace(forward_only=False, defer_fsdp_grad_sync=True)
    engine.model_config = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=7))
    engine.optimizer_config = SimpleNamespace(clip_grad=1.0)
    engine.distributed_config = object()
    engine.distributed_setup = SimpleNamespace(mesh_context=object())
    engine._is_offload_param = False
    engine._is_offload_optimizer = False
    engine._build_model = lambda: module
    engine._build_optimizer = lambda _module: optimizer
    engine._build_lr_scheduler = lambda _optimizer: scheduler
    engine._build_checkpointer = lambda: None
    engine.to = lambda **kwargs: None

    def build_training_engine(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(transformer_impl, "Engine", build_training_engine)
    monkeypatch.setattr(transformer_impl, "maybe_shard_optimizer", lambda _model, optim, _config: optim)
    monkeypatch.setattr(transformer_impl, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: 0)
    monkeypatch.setattr(transformer_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    AutomodelEngine.initialize(engine)

    assert captured["args"] == (module,)
    assert captured["kwargs"]["optimizer"] is optimizer
    assert captured["kwargs"]["lr_scheduler"] is None
    assert captured["kwargs"]["max_grad_norm"] == 1.0
    assert captured["kwargs"]["mesh_context"] is engine.distributed_setup.mesh_context
    assert engine.lr_scheduler is scheduler


@pytest.mark.parametrize("attention", ["te", "flash_attention_2"])
def test_remove_padding_uses_raw_thd_or_indexed_mask(attention):
    engine = object.__new__(AutomodelEngineWithLMHead)
    engine.engine_config = SimpleNamespace(attn_implementation=attention)
    engine._packing_layout = "thd" if attention == "te" else "indexed_mask"

    model_inputs, output_args = AutomodelEngineWithLMHead.prepare_model_inputs(engine, _packed_microbatch())

    torch.testing.assert_close(model_inputs["input_ids"], torch.tensor([[10, 11, 12, 20, 21]]))
    torch.testing.assert_close(model_inputs["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
    assert output_args["original_token_count"] == 5
    assert "cu_seqlens" not in model_inputs
    assert "max_seqlen" not in model_inputs
    if attention == "te":
        assert model_inputs["qkv_format"] == "thd"
        torch.testing.assert_close(model_inputs["seq_lens"], torch.tensor([[3, 2]], dtype=torch.int32))
        torch.testing.assert_close(model_inputs["seq_lens_padded"], model_inputs["seq_lens"])
        assert model_inputs["attention_mask"] is None
    else:
        packed_seq_ids = torch.tensor([[1, 1, 1, 2, 2]])
        torch.testing.assert_close(model_inputs["attention_mask"], packed_seq_ids)
        assert "_packed_seq_ids" not in model_inputs
        assert "qkv_format" not in model_inputs
        assert "seq_lens" not in model_inputs


def test_fused_log_probs_fail_closed():
    engine = object.__new__(AutomodelEngineWithLMHead)
    engine.engine_config = SimpleNamespace(attn_implementation="te", use_remove_padding=True)
    batch = _packed_microbatch()
    tu.assign_non_tensor(batch, use_fused_kernels=True)

    with pytest.raises(NotImplementedError, match="has not integrated VERL fused log-prob kernels"):
        AutomodelEngineWithLMHead.prepare_model_inputs(engine, batch)


@pytest.mark.parametrize("flag", ["calculate_sum_pi_squared", "distillation_use_topk", "distillation_only"])
def test_unintegrated_output_processing_fails_closed(flag):
    engine = object.__new__(AutomodelEngineWithLMHead)
    engine.engine_config = SimpleNamespace(attn_implementation="te", use_remove_padding=True)
    batch = _packed_microbatch()
    tu.assign_non_tensor(batch, **{flag: True})

    with pytest.raises(NotImplementedError, match=flag):
        AutomodelEngineWithLMHead.prepare_model_inputs(engine, batch)


def test_unpacked_outputs_use_flat_jagged_logits_and_one_global_target_roll():
    engine = object.__new__(AutomodelEngineWithLMHead)
    engine.engine_config = SimpleNamespace(attn_implementation="sdpa", use_remove_padding=False)
    engine._packing_layout = None
    batch = _packed_microbatch()
    tu.assign_non_tensor(batch, use_remove_padding=False)
    model_inputs, output_args = AutomodelEngineWithLMHead.prepare_model_inputs(engine, batch)

    seen = {}

    def token_statistics(logits, targets, temperature, calculate_entropy):
        seen.update(logits=logits, targets=targets, temperature=temperature)
        return torch.arange(5, dtype=torch.float32), None

    engine._compute_token_statistics = token_statistics
    logits = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    output = AutomodelEngineWithLMHead.prepare_model_outputs(engine, SimpleNamespace(logits=logits), output_args, batch)

    torch.testing.assert_close(model_inputs["input_ids"], torch.tensor([[10, 11, 12], [20, 21, 0]]))
    torch.testing.assert_close(seen["logits"], torch.cat((logits[0, :3], logits[1, :2])))
    torch.testing.assert_close(seen["targets"], torch.tensor([11, 12, 20, 21, 10]))
    torch.testing.assert_close(seen["temperature"], torch.ones(5))
    torch.testing.assert_close(output["log_probs"].values(), torch.arange(5, dtype=torch.float32))


def test_unpacked_attention_mask_uses_full_input_lengths_not_response_loss_mask():
    engine = object.__new__(AutomodelEngineWithLMHead)
    engine.engine_config = SimpleNamespace(attn_implementation="sdpa", use_remove_padding=False)
    engine._packing_layout = None
    batch = _packed_microbatch()
    batch["loss_mask"] = torch.nested.as_nested_tensor(
        [torch.ones(1), torch.ones(1)],
        layout=torch.jagged,
    )
    tu.assign_non_tensor(batch, use_remove_padding=False)

    model_inputs, _ = AutomodelEngineWithLMHead.prepare_model_inputs(engine, batch)

    torch.testing.assert_close(
        model_inputs["attention_mask"],
        torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.int32),
    )


@pytest.mark.parametrize("vocab_shard_dim", [2, -1])
def test_unpacked_tp_logits_remap_vocab_shard_after_flatten(monkeypatch, vocab_shard_dim):
    class FakeReplicate:
        pass

    class FakeShard:
        def __init__(self, dim):
            self.dim = dim

    class FakeDTensor:
        reconstructed = None

        def __init__(self, local, placements, *, shape):
            self.local = local
            self.placements = placements
            self.shape = torch.Size(shape)
            self.device_mesh = object()

        @property
        def device(self):
            return self.local.device

        @property
        def ndim(self):
            return len(self.shape)

        def __getitem__(self, key):
            return type(self)(self.local[key], self.placements, shape=self.shape)

        def to_local(self):
            return self.local

        @classmethod
        def from_local(cls, local, device_mesh, placements, **kwargs):
            cls.reconstructed = cls(local, placements, shape=kwargs["shape"])
            return cls.reconstructed

    monkeypatch.setattr(transformer_impl, "DTensor", FakeDTensor)
    monkeypatch.setattr(transformer_impl, "Replicate", FakeReplicate)
    monkeypatch.setattr(transformer_impl, "Shard", FakeShard)

    engine = object.__new__(AutomodelEngineWithLMHead)
    batch = _packed_microbatch()
    tu.assign_non_tensor(batch, use_remove_padding=False)
    output_args = {
        "padded_sequence_length": 3,
        "target_tokens": torch.tensor([11, 12, 20, 21, 10]),
        "token_temperatures": torch.ones(5),
    }
    seen = {}

    def token_statistics(logits, *_args):
        seen["logits"] = logits
        return torch.arange(5, dtype=torch.float32), None

    engine._compute_token_statistics = token_statistics
    local_logits = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    logits = FakeDTensor(local_logits, (FakeReplicate(), FakeShard(vocab_shard_dim)), shape=(2, 3, 4))

    AutomodelEngineWithLMHead.prepare_model_outputs(
        engine,
        SimpleNamespace(logits=logits),
        output_args,
        batch,
    )

    flat_logits = seen["logits"]
    assert flat_logits is FakeDTensor.reconstructed
    assert flat_logits.shape == torch.Size((5, 4))
    assert isinstance(flat_logits.placements[0], FakeReplicate)
    assert isinstance(flat_logits.placements[1], FakeShard)
    assert flat_logits.placements[1].dim == 1
    torch.testing.assert_close(flat_logits.local, torch.cat((local_logits[0, :3], local_logits[1, :2])))


def test_flash_attention_model_enables_indexed_packing(monkeypatch):
    from nemo_automodel.components.models.common import packing

    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="flash_attention_2"))
    loaded = {}
    configured = []

    def from_pretrained(path, **kwargs):
        loaded.update(path=path, **kwargs)
        return model

    monkeypatch.setattr(transformer_impl.NeMoAutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(packing, "configure_packing", configured.append)

    engine = object.__new__(AutomodelEngine)
    engine.distributed_setup = object()
    engine.model_config = SimpleNamespace(
        local_path="local-model",
        path="remote-model",
        trust_remote_code=False,
        use_liger=False,
        architectures=["QwenForCausalLM"],
        hf_config=SimpleNamespace(architectures=["QwenForCausalLM"]),
    )
    engine.engine_config = SimpleNamespace(
        attn_implementation="flash_attention_2",
        use_remove_padding=True,
        model_dtype="fp32",
        ep_size=1,
        backend_config={},
        enable_fp8=False,
        enable_compile=False,
    )

    assert AutomodelEngine._build_model(engine) is model
    assert loaded["path"] == "local-model"
    assert loaded["force_hf"] is True
    assert loaded["has_packed_sequence"] is True
    assert loaded["config"] is engine.model_config.hf_config
    assert loaded["use_liger_kernel"] is False
    assert configured == ["flash_attention_2"]
    assert engine._packing_layout == "indexed_mask"


def test_dtensor_token_statistics_delegate_to_vocab_parallel_primitives(monkeypatch):
    class FakeDTensor:
        def __init__(self, local, *, shape=None):
            self.local = local
            self.shape = torch.Size(shape or local.shape)
            self.device_mesh = object()
            self.placements = (object(),)

        def to_local(self):
            return self.local

        def stride(self):
            return (3, 1)

        def full_tensor(self):  # pragma: no cover - the test fails if this path is used
            raise AssertionError("the TP vocabulary must not be all-gathered")

        @classmethod
        def from_local(cls, local, device_mesh, placements, **kwargs):
            return cls(local, shape=kwargs["shape"])

    seen = {}

    def selected(logits, targets):
        seen["selected"] = (logits, targets)
        return torch.tensor([-0.5, -1.5])

    def entropy(logits):
        seen["entropy"] = logits
        return torch.tensor([0.25, 0.75])

    monkeypatch.setattr(transformer_impl, "DTensor", FakeDTensor)
    monkeypatch.setattr(transformer_impl, "token_log_probs", selected)
    monkeypatch.setattr(transformer_impl, "token_entropy", entropy)

    engine = object.__new__(AutomodelEngineWithLMHead)
    logits = FakeDTensor(torch.tensor([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]]))
    targets = torch.tensor([1, 2])
    log_probs, token_entropy = AutomodelEngineWithLMHead._compute_token_statistics(
        engine,
        logits,
        targets,
        torch.tensor([2.0, 4.0]),
        calculate_entropy=True,
    )

    torch.testing.assert_close(seen["selected"][0].local, torch.tensor([[1.0, 2.0, 3.0]]).expand(2, -1))
    assert seen["selected"][0] is seen["entropy"]
    torch.testing.assert_close(seen["selected"][1], targets)
    torch.testing.assert_close(log_probs, torch.tensor([-0.5, -1.5]))
    torch.testing.assert_close(token_entropy, torch.tensor([0.25, 0.75]))


@pytest.mark.parametrize(("tp_rank", "expected"), [(0, True), (1, False)])
def test_only_tp_source_rank_returns_outputs(tp_rank, expected):
    class Axis:
        def size(self):
            return 2

    class Mesh:
        mesh_dim_names = ("dp", "tp")

        def __getitem__(self, name):
            assert name == "tp"
            return Axis()

        def get_local_rank(self, name):
            assert name == "tp"
            return tp_rank

    engine = object.__new__(AutomodelEngine)
    engine.device_mesh = Mesh()

    assert AutomodelEngine.is_mp_src_rank_with_outputs(engine) is expected


@pytest.mark.parametrize(
    ("override", "error", "message"),
    [
        ({"dtype": "float16"}, ValueError, "only bfloat16"),
        ({"mp_param_dtype": "fp16"}, ValueError, "mp_param_dtype"),
        ({"mp_output_dtype": "fp16"}, ValueError, "mp_output_dtype"),
        ({"mp_reduce_dtype": "fp16"}, ValueError, "mp_reduce_dtype"),
        ({"model_dtype": "fp16"}, ValueError, "model_dtype"),
        ({"backend_config": {"gate_precision": "fp16"}}, ValueError, "gate_precision"),
        ({"moe_config": {"lm_head_precision": "float16"}}, ValueError, "lm_head_precision"),
        ({"pp_size": 2}, NotImplementedError, "pipeline parallelism"),
        ({"cp_size": 2}, NotImplementedError, "context parallelism"),
        ({"grad_offload": True}, NotImplementedError, "gradient offload"),
        ({"router_replay": SimpleNamespace(mode="R2")}, NotImplementedError, "router replay"),
        (
            {"param_offload": True, "distributed_strategy": "ddp"},
            NotImplementedError,
            "parameter offload requires.*FSDP2",
        ),
    ],
)
def test_unsupported_precision_and_parallelism_fail_closed(override, error, message):
    config = {
        "dtype": "bfloat16",
        "mp_param_dtype": "bf16",
        "mp_output_dtype": "bf16",
        "mp_reduce_dtype": "fp32",
        "model_dtype": "fp32",
        "pp_size": 1,
        "cp_size": 1,
        "ep_size": 1,
        "grad_offload": False,
        "router_replay": SimpleNamespace(mode="disabled"),
        "param_offload": False,
        "distributed_strategy": "fsdp2",
        "use_remove_padding": True,
        "attn_implementation": "te",
    }
    config.update(override)
    engine = object.__new__(AutomodelEngine)
    engine.engine_config = SimpleNamespace(**config)
    engine.optimizer_config = SimpleNamespace(
        exp_avg_dtype=None,
        exp_avg_sq_dtype=None,
        master_weight_dtype=None,
    )

    with pytest.raises(error, match=message):
        AutomodelEngine._validate_precision_and_parallelism(engine)


def test_activation_offload_fails_closed():
    engine = object.__new__(AutomodelEngine)
    engine.model_config = SimpleNamespace(enable_activation_offload=True)
    engine.engine_config = SimpleNamespace(
        dtype="bfloat16",
        mp_param_dtype="bf16",
        mp_output_dtype="bf16",
        mp_reduce_dtype="fp32",
        model_dtype="fp32",
        pp_size=1,
        cp_size=1,
        grad_offload=False,
        router_replay=SimpleNamespace(mode="disabled"),
        param_offload=False,
    )
    engine.optimizer_config = SimpleNamespace()

    with pytest.raises(NotImplementedError, match="activation offload"):
        AutomodelEngine._validate_precision_and_parallelism(engine)


@pytest.mark.parametrize("name", ["exp_avg_dtype", "exp_avg_sq_dtype", "master_weight_dtype"])
def test_fp16_optimizer_state_is_rejected(name):
    engine = object.__new__(AutomodelEngine)
    engine.engine_config = SimpleNamespace(
        dtype="bfloat16",
        mp_param_dtype="bf16",
        mp_output_dtype="bf16",
        mp_reduce_dtype="fp32",
        model_dtype="fp32",
        pp_size=1,
        cp_size=1,
        ep_size=1,
        use_remove_padding=True,
        attn_implementation="te",
    )
    engine.optimizer_config = SimpleNamespace(**{name: "fp16"})

    with pytest.raises(ValueError, match=rf"{name}.*float16"):
        AutomodelEngine._validate_precision_and_parallelism(engine)


@pytest.mark.parametrize("name", ["exp_avg_dtype", "exp_avg_sq_dtype", "master_weight_dtype"])
def test_fp16_optimizer_override_is_rejected(name):
    engine = object.__new__(AutomodelEngine)
    engine.device_mesh = None
    engine.optimizer_config = SimpleNamespace(
        lr=1e-4,
        weight_decay=0.0,
        eps=1e-8,
        betas=(0.9, 0.999),
        master_weights=False,
        store_param_remainders=False,
        exp_avg_dtype=None,
        exp_avg_sq_dtype=None,
        master_weight_dtype=None,
        override_optimizer_config={name: "fp16"},
        optimizer_impl="torch.optim",
        optimizer="AdamW",
    )

    with pytest.raises(ValueError, match=rf"override {name}.*float16"):
        AutomodelEngine._build_optimizer(engine, torch.nn.Linear(1, 1))


def test_packed_attention_layout_is_validated_after_model_load():
    base = {
        "dtype": "bfloat16",
        "mp_param_dtype": "bf16",
        "mp_output_dtype": "bf16",
        "mp_reduce_dtype": "fp32",
        "model_dtype": "fp32",
        "pp_size": 1,
        "cp_size": 1,
        "ep_size": 2,
        "use_remove_padding": True,
    }
    engine = object.__new__(AutomodelEngine)
    engine.optimizer_config = SimpleNamespace()
    for attention in ("te", "flash_attention_2", "sdpa"):
        engine.engine_config = SimpleNamespace(**base, attn_implementation=attention)
        AutomodelEngine._validate_precision_and_parallelism(engine)


@pytest.mark.parametrize(("ep_size", "expected_moe_config"), [(1, None), (2, {"dispatcher": "hybridep"})])
def test_moe_config_is_only_passed_when_expert_parallelism_is_enabled(
    monkeypatch,
    ep_size,
    expected_moe_config,
):
    captured = {}

    def build(**kwargs):
        captured.update(kwargs)
        return "setup"

    monkeypatch.setattr(transformer_impl.DistributedSetup, "build", staticmethod(build))
    engine = object.__new__(AutomodelEngine)
    engine.engine_config = SimpleNamespace(
        distributed_strategy="ddp",
        dp_replicate_size=1,
        tp_size=1,
        pp_size=1,
        cp_size=1,
        ep_size=ep_size,
        moe_config={"dispatcher": "hybridep"},
        activation_checkpointing=False,
    )

    assert AutomodelEngine._build_distributed_setup(engine, world_size=max(ep_size, 1)) == "setup"
    assert captured["moe_parallel_config"] == expected_moe_config


def test_weight_export_streams_normalized_adapter_weights_on_every_rank(monkeypatch):
    class FakeDTensor(torch.Tensor):
        full_tensor_calls = 0

        @staticmethod
        def __new__(cls, value):
            return torch.Tensor._make_subclass(cls, value, require_grad=False)

        def full_tensor(self):
            type(self).full_tensor_calls += 1
            return self.as_subclass(torch.Tensor) + 1

    class Adapter:
        def __init__(self):
            self.calls = []

        def convert_single_tensor_to_hf(self, name, tensor, **kwargs):
            self.calls.append((name, tensor, kwargs))
            return [(f"hf.{name}", tensor * 2)]

    module = torch.nn.Linear(1, 1, bias=False)
    module.state_dict_adapter = Adapter()
    engine = object.__new__(AutomodelEngine)
    engine.module = module
    engine._is_offload_param = True
    engine.is_mp_src_rank_with_outputs = lambda: False

    staged = []
    monkeypatch.setattr(transformer_impl, "DTensor", FakeDTensor)
    monkeypatch.setattr(
        transformer_impl,
        "get_model_state_dict",
        lambda _module: {"weight": FakeDTensor(torch.tensor([3.0])), "layer._extra_state": torch.tensor(0)},
    )
    monkeypatch.setattr(transformer_impl, "load_automodel_model_to_gpu", lambda _module: staged.append("load"))
    monkeypatch.setattr(transformer_impl, "offload_automodel_model_to_cpu", lambda _module: staged.append("offload"))

    params, peft_config = AutomodelEngine.get_per_tensor_param(engine)
    assert staged == []
    exported = list(params)

    assert peft_config is None
    assert staged == ["load", "offload"]
    assert FakeDTensor.full_tensor_calls == 1
    assert [name for name, _ in exported] == ["hf.weight"]
    torch.testing.assert_close(exported[0][1], torch.tensor([8.0]))
    assert module.state_dict_adapter.calls[0][0] == "weight"


def test_ep_weight_export_rejects_plain_local_experts(monkeypatch):
    module = torch.nn.Linear(1, 1, bias=False)
    engine = object.__new__(AutomodelEngine)
    engine.module = module
    engine.engine_config = SimpleNamespace(ep_size=2)
    engine._is_offload_param = False

    monkeypatch.setattr(
        transformer_impl,
        "get_model_state_dict",
        lambda _module: {"model.layers.0.experts.weight": torch.tensor([1.0])},
    )
    monkeypatch.setattr(transformer_impl, "convert_weight_keys", lambda params, _model: params)
    monkeypatch.setattr(transformer_impl, "load_automodel_model_to_gpu", lambda _module: None)

    params, _ = AutomodelEngine.get_per_tensor_param(engine)
    with pytest.raises(RuntimeError, match="not a DTensor"):
        list(params)


def test_lora_configuration_fails_before_initialization():
    model_config = SimpleNamespace(lora_rank=8, lora_adapter_path=None, lora={})
    with pytest.raises(NotImplementedError, match="LoRA/PEFT"):
        AutomodelEngine(model_config, SimpleNamespace(), SimpleNamespace(), None)


def test_checkpoint_policy_selects_optimizer_and_roundtrips_extra(monkeypatch, tmp_path):
    from verl.trainer.config import CheckpointConfig
    from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

    class Scheduler:
        def __init__(self):
            self.step_count = 7

        def state_dict(self):
            return {"step_count": self.step_count}

        def load_state_dict(self, state):
            self.step_count = state["step_count"]

    class Checkpointer:
        def __init__(self):
            self.calls = []

        def save_model(self, *args, **kwargs):
            self.calls.append("save_model")

        def save_optimizer(self, optimizer, model, path, scheduler):
            self.calls.append(("save_optimizer", path, scheduler))

        def load_model(self, *args, **kwargs):
            self.calls.append("load_model")

        def load_optimizer(self, optimizer, model, path, scheduler):
            self.calls.append(("load_optimizer", path, scheduler))

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(transformer_impl, "load_automodel_model_to_gpu", lambda _module: None)

    module = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    scheduler = Scheduler()
    config = CheckpointConfig(
        save_contents=["optimizer", "extra"],
        load_contents=["optimizer", "extra"],
    )
    policy = BaseCheckpointManager(module, optimizer, scheduler, checkpoint_config=config)
    policy.get_rng_state = lambda: {"test": 11}
    loaded_rng = []
    policy.load_rng_state = loaded_rng.append
    retention = []
    policy.ensure_checkpoint_capacity = lambda count: retention.append(("ensure", count))
    policy.register_checkpoint = lambda path, count: retention.append(("register", path, count))

    engine = object.__new__(AutomodelEngine)
    engine.module = module
    engine.optimizer = optimizer
    engine.lr_scheduler = scheduler
    engine.checkpoint_manager = policy
    engine.checkpointer = Checkpointer()
    engine._is_offload_param = False
    engine._is_offload_optimizer = False

    AutomodelEngine.save_checkpoint(engine, str(tmp_path), global_step=19, max_ckpt_to_keep=2)
    assert [call[0] for call in engine.checkpointer.calls] == ["save_optimizer"]
    assert engine.checkpointer.calls[0][2] is None
    assert retention == [("ensure", 2), ("register", str(tmp_path), 2)]

    scheduler.step_count = -1
    AutomodelEngine.load_checkpoint(engine, str(tmp_path))
    assert [call[0] for call in engine.checkpointer.calls] == ["save_optimizer", "load_optimizer"]
    assert engine.checkpointer.calls[1][2] is None
    assert scheduler.step_count == 7
    assert loaded_rng == [{"test": 11}]
    assert policy.previous_global_step == 19


@pytest.mark.parametrize("method", ["save_checkpoint", "load_checkpoint"])
def test_checkpoint_hdfs_fails_closed(method):
    engine = object.__new__(AutomodelEngine)
    with pytest.raises(NotImplementedError, match="HDFS"):
        getattr(AutomodelEngine, method)(engine, "/tmp/local", hdfs_path="hdfs://checkpoint")
