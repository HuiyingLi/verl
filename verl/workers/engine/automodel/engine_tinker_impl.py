# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""verl automodel engine — redone against the current nemo_automodel + Engine.

Standalone ``BaseEngine`` implementation (does not depend on the legacy
automodel engine). It builds the model with the current Automodel API
(``NeMoAutoModelForCausalLM.from_pretrained`` +
``DistributedSetup.build``) and delegates the training step to
``nemo_automodel.components.training.engine.Engine`` via its ``PackedBatch``
pass-through door: the Engine owns the microbatch lifecycle, forward, per-datum
logprob extraction, gradient clipping and the optimizer step. verl keeps its
data layout (THD remove-padding) and its loss functions; a thin bridge maps the
Engine's per-datum ``ModelOutput`` back to verl's flat ``model_output["log_probs"]``.
"""

import torch
from nemo_automodel.components.datasets.datum import PackedBatch
from nemo_automodel.components.training.engine import Engine

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_device_id

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import postprocess_batch_func, prepare_micro_batches


@EngineRegistry.register(model_type="language_model", backend=["automodel"], device=["cuda"])
class AutomodelEngine(BaseEngine):
    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.mode = None
        self._engine = None
        # Build the distributed setup eagerly: the worker queries dp rank/size in
        # its __init__ (before init_model), so the mesh must exist now.
        self._build_distributed()

    def _build_distributed(self):
        from nemo_automodel.components.distributed.config import (
            DDPConfig,
            DistributedSetup,
            FSDP2Config,
            MegatronFSDPConfig,
            MoEParallelizerConfig,
        )
        from nemo_automodel.components.distributed.mesh import ParallelismSizes
        from torch.distributed.fsdp import MixedPrecisionPolicy

        from verl.utils.torch_dtypes import PrecisionType

        ec = self.engine_config
        parallelism_sizes = ParallelismSizes(
            tp_size=ec.tp_size,
            pp_size=ec.pp_size,
            cp_size=ec.cp_size,
            ep_size=ec.ep_size,
            dp_replicate_size=ec.dp_replicate_size,
        )

        if ec.distributed_strategy == "fsdp2":
            mp_policy = MixedPrecisionPolicy(
                param_dtype=PrecisionType.to_dtype(ec.mp_param_dtype),
                reduce_dtype=PrecisionType.to_dtype(ec.mp_reduce_dtype),
                output_dtype=PrecisionType.to_dtype(ec.mp_output_dtype),
                cast_forward_inputs=True,
            )
            strategy = FSDP2Config(
                sequence_parallel=ec.sequence_parallel,
                mp_policy=mp_policy,
                activation_checkpointing=ec.activation_checkpointing,
                defer_fsdp_grad_sync=ec.defer_fsdp_grad_sync,
            )
        elif ec.distributed_strategy == "ddp":
            strategy = DDPConfig(activation_checkpointing=ec.activation_checkpointing)
        elif ec.distributed_strategy == "megatron_fsdp":
            strategy = MegatronFSDPConfig(activation_checkpointing=ec.activation_checkpointing)
        else:
            raise ValueError(f"Unsupported distributed_strategy: {ec.distributed_strategy}")

        moe_parallel_config = None
        if ec.ep_size > 1:
            moe_parallel_config = MoEParallelizerConfig(
                **({"mp_policy": strategy.mp_policy} if hasattr(strategy, "mp_policy") else {}),
                **(ec.moe_config or {}),
            )

        self._dist_setup = DistributedSetup.build(
            strategy=strategy,
            parallelism_sizes=parallelism_sizes,
            moe_parallel_config=moe_parallel_config,
            activation_checkpointing=ec.activation_checkpointing,
            world_size=torch.distributed.get_world_size(),
        )
        self.device_mesh = self._dist_setup.mesh_context.device_mesh
        self.moe_mesh = self._dist_setup.mesh_context.moe_mesh

    @property
    def is_param_offload_enabled(self) -> bool:
        return self.engine_config.param_offload

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self.engine_config.optimizer_offload

    # ── construction: current Automodel build + tinker Engine ────────────────

    def _build_module(self):
        from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM
        from nemo_automodel.components.training.rng import ScopedRNG

        from verl.utils.torch_dtypes import PrecisionType

        ec = self.engine_config
        backend = dict(ec.backend_config or {})
        # Disable fused TE RoPE: the fused kernel indexes rotary angles by
        # physical sequence position and assumes contiguous [0, seq_len)
        # positions, so it does NOT honor the per-sequence position_id resets
        # of a THD-packed batch. With packed GRPO micro-batches that silently
        # corrupts RoPE for every non-first sequence in a pack (logprobs drift
        # vs the vLLM rollout, breaking the importance ratio). The non-fused
        # path gathers cos/sin by position_id value and is packing-correct.
        backend["rope_fusion"] = False

        with ScopedRNG(seed=ec.seed, ranked=True):
            return NeMoAutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_config.path,
                trust_remote_code=getattr(self.model_config, "trust_remote_code", False),
                attn_implementation=ec.attn_implementation,
                torch_dtype=PrecisionType.to_dtype(ec.model_dtype),
                distributed_setup=self._dist_setup,
                backend=backend,
            )

    def _build_optimizers(self):
        from nemo_automodel.components.optim import build_optimizer_config

        cfg = self.optimizer_config
        overrides = dict(cfg.override_optimizer_config or {})
        target = cfg.optimizer
        optimizer_impl = getattr(cfg, "optimizer_impl", None)
        if optimizer_impl and "." not in target and target != target.lower():
            target = f"{optimizer_impl}.{target}"
        opt_cfg = build_optimizer_config(target, overrides)

        optimizer_kwargs = {
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "eps": cfg.eps,
            "betas": tuple(cfg.betas),
        }
        for key in (
            "master_weights",
            "store_param_remainders",
            "exp_avg_dtype",
            "exp_avg_sq_dtype",
            "master_weight_dtype",
        ):
            value = getattr(cfg, key, None)
            if value:
                optimizer_kwargs[key] = value

        for key, value in optimizer_kwargs.items():
            if key in overrides:
                continue
            if hasattr(opt_cfg, key):
                setattr(opt_cfg, key, value)
            elif hasattr(opt_cfg, "kwargs"):
                opt_cfg.kwargs.setdefault(key, value)
        return opt_cfg.build(self.module, device_mesh=self.device_mesh, is_peft=False)

    def _build_lr_schedulers(self, optimizers):
        from nemo_automodel.components.optim import OptimizerParamScheduler

        cfg = self.optimizer_config
        total_steps = cfg.total_training_steps
        if total_steps <= 0:
            raise ValueError("optim.total_training_steps must be set before building the Automodel LR scheduler.")

        num_warmup_steps = cfg.lr_warmup_steps
        if num_warmup_steps is None or num_warmup_steps <= 0:
            num_warmup_steps = int(cfg.lr_warmup_steps_ratio * total_steps)

        init_lr_ratio = cfg.init_lr_ratio if cfg.init_lr_ratio is not None else 0.1
        min_lr_ratio = cfg.min_lr_ratio if cfg.min_lr_ratio is not None else 0.01
        scheduler_kwargs = dict(
            lr_warmup_steps=num_warmup_steps,
            lr_decay_steps=total_steps,
            lr_decay_style=cfg.lr_scheduler_type,
            init_lr=cfg.lr * init_lr_ratio,
            max_lr=cfg.lr,
            min_lr=cfg.lr * min_lr_ratio,
            start_wd=cfg.weight_decay,
            end_wd=cfg.weight_decay,
            wd_incr_steps=total_steps,
            wd_incr_style=getattr(cfg, "wd_incr_style", "constant"),
        )
        return [OptimizerParamScheduler(optimizer=optimizer, **scheduler_kwargs) for optimizer in optimizers]

    def initialize(self):
        ec = self.engine_config
        self.module = self._build_module()

        optimizers = [] if ec.forward_only else self._build_optimizers()
        lr_schedulers = [] if ec.forward_only else self._build_lr_schedulers(optimizers)

        engine_cfg = Engine.Config(max_grad_norm=self.optimizer_config.clip_grad)
        self._engine = Engine(
            config=engine_cfg,
            model_parts=[self.module],
            optimizers=optimizers,
            lr_schedulers=lr_schedulers,
            distributed_setup=self._dist_setup,
        )

    # ── data: verl THD micro-batch -> Engine PackedBatch ─────────────────────

    def _to_packed(self, micro_batch) -> PackedBatch:
        pad_mode = tu.get_non_tensor_data(micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]
        input_ids_rmpad = input_ids.values().unsqueeze(0)  # [1, total]
        if position_ids.dim() == 3:
            pos_rmpad = position_ids.values().unsqueeze(1)
        else:
            pos_rmpad = position_ids.values().unsqueeze(0)
        targets = torch.roll(input_ids_rmpad, shifts=-1, dims=1).squeeze(0)  # [total]
        seq_lens = input_ids.offsets().diff().tolist()
        return PackedBatch(
            model_inputs={"input_ids": input_ids_rmpad, "position_ids": pos_rmpad},
            seq_lens=seq_lens,
            targets=targets,
        )

    def _verl_loss_closure(self, micro_batch, loss_function, sink: list):
        dp_group = self.get_data_parallel_group()

        def closure(model_output):
            # Bridge: Engine per-datum outputs -> verl jagged model_output fields.
            # verl's loss (no_padding_2_padding) and postprocess_batch_func (nt.unbind)
            # both expect jagged nested tensors; as_nested_tensor preserves autograd.
            verl_mo = {"log_probs": torch.nested.as_nested_tensor(list(model_output.logprobs), layout=torch.jagged)}
            if model_output.entropy is not None:
                # compute_old_log_prob also consumes entropy.
                verl_mo["entropy"] = torch.nested.as_nested_tensor(list(model_output.entropy), layout=torch.jagged)
            if loss_function is None:
                # Inference (compute_log_prob): capture outputs, no loss/backward.
                sink.append({"model_output": verl_mo})
                return None
            loss, metrics = loss_function(model_output=verl_mo, data=micro_batch, dp_group=dp_group)
            # verl metrics (Metric objects) go to the sink for postprocess_batch_func;
            # the Engine only needs the scalar loss to back-prop.
            sink.append({"model_output": verl_mo, "loss": loss.detach().item(), "metrics": metrics})
            return loss, {}

        return closure

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )
        packs, closures, outputs = [], [], []
        for mb in micro_batches:
            mb = mb.to(get_device_id())
            packs.append(self._to_packed(mb))
            closures.append(self._verl_loss_closure(mb, loss_function, outputs))

        self._engine.forward_backward(packs, loss_fn=closures, forward_only=forward_only)
        return postprocess_batch_func(output_lst=outputs, indices=indices, data=data)

    # ── optimizer / lr / dp / mode / device — delegate to the Engine ─────────

    def optimizer_zero_grad(self):
        self._engine.zero_grad()

    def optimizer_step(self):
        _, grad_norm = self._engine.optimizer_step()
        return grad_norm

    def lr_scheduler_step(self):
        return self._engine.lr_scheduler_step()

    def _dp_mesh(self):
        if not self.device_mesh:
            return None
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

        name = "dp_cp" if self.device_mesh["cp"].size() > 1 else "dp"
        return get_flat_mesh(self.device_mesh, name)

    def get_data_parallel_rank(self):
        mesh = self._dp_mesh()
        return mesh.get_local_rank() if mesh is not None else torch.distributed.get_rank()

    def get_data_parallel_size(self):
        group = self.get_data_parallel_group()
        return group.size() if group is not None else torch.distributed.get_world_size()

    def get_data_parallel_group(self):
        mesh = self._dp_mesh()
        return mesh.get_group() if mesh is not None else None

    def is_mp_src_rank_with_outputs(self):
        return True  # non-PP: every rank carries outputs

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        self._engine.to(device, model=model, optimizer=optimizer)

    # ── rollout weight sync ──────────────────────────────────────────────────

    def get_per_tensor_param(self, **kwargs):
        """Yield HF-named (name, full_tensor) for rollout (vllm) weight refit.

        Mirrors the reference engine: verl's convert_weight_keys maps Automodel
        state-dict keys to HF names; DTensors are materialized to full tensors.
        """
        from torch.distributed.tensor import DTensor

        from verl.utils.model import convert_weight_keys

        params = self.module.state_dict()
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        def param_generator():
            for name, param in params.items():
                yield name, (param.full_tensor() if isinstance(param, DTensor) else param)

        return param_generator(), None

    # ── checkpoint (delegate to Engine.save_state/load_state) ────────────────

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        self._engine.save_state(local_path)

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True, **kwargs):
        self._engine.load_state(local_path)

    def disable_adapter(self):
        return self._engine.disable_adapter()

    def train_mode(self, **kwargs):
        return EngineModeCtx(self, training=True, **kwargs)

    def eval_mode(self, **kwargs):
        return EngineModeCtx(self, training=False, **kwargs)


class EngineModeCtx(BaseEngineCtx):
    def __init__(self, engine: AutomodelEngine, training: bool, **kwargs):
        self.training = training
        super().__init__(engine=engine, mode="train" if training else "eval", **kwargs)

    def __enter__(self):
        super().__enter__()
        for part in self.engine._engine.model_parts:
            part.train(self.training)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.training:
            self.engine.optimizer_zero_grad()
        return super().__exit__(exc_type, exc_value, traceback)
