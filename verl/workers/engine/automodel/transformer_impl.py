# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""NeMo AutoModel training-engine adapter."""

import gc
import logging
import os
from typing import Any, Callable, Optional

import torch
import torch.distributed
from huggingface_hub.constants import HF_HUB_CACHE
from nemo_automodel import (
    Datum,
    LossInputLayout,
    NeMoAutoModelForCausalLM,
    NeMoAutoModelForImageTextToText,
)
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.distributed import (
    DDPConfig,
    DistributedSetup,
    FSDP2Config,
    MegatronFSDPConfig,
    ParallelismSizes,
)
from nemo_automodel.components.distributed.megatron_fsdp import maybe_shard_optimizer
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.components.loss import vocab_parallel_entropy, vocab_parallel_log_probs
from nemo_automodel.components.optim import OptimizerParamScheduler, build_optimizer
from nemo_automodel.engine import Engine, collate_prebatched
from tensordict import TensorDict
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.tensor import DTensor, Replicate, Shard

import verl.utils.torch_functional as verl_F
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.model import convert_weight_keys, extract_multi_modal_inputs, get_hf_auto_model_class
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.config import AutomodelEngineConfig, AutomodelOptimizerConfig, HFModelConfig
from verl.workers.utils.padding import build_attention_mask_from_nested

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches
from .utils import (
    get_pp_rank,
    get_tp_rank,
    load_automodel_model_to_gpu,
    load_automodel_optimizer,
    offload_automodel_model_to_cpu,
    offload_automodel_optimizer,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AutomodelEngine(BaseEngine):
    """Engine implementation using Automodel for distributed training."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: AutomodelEngineConfig,
        optimizer_config: AutomodelOptimizerConfig,
        checkpoint_config: CheckpointConfig,
        **kwargs,
    ):
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        lora_config = getattr(self.model_config, "lora", None) or {}
        if (
            getattr(self.model_config, "lora_rank", 0) > 0
            or getattr(self.model_config, "lora_adapter_path", None)
            or lora_config.get("rank", 0) > 0
            or lora_config.get("merge", False)
        ):
            raise NotImplementedError("AutoModel veRL has not integrated LoRA/PEFT training or export")

        self.mode = None
        self.rank = torch.distributed.get_rank()

        # Apply compatibility patches early in the process
        from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
        from nemo_automodel.shared.te_patches import apply_te_patches

        apply_cache_compatibility_patches()
        apply_te_patches()

        self._validate_precision_and_parallelism()
        world_size = torch.distributed.get_world_size()
        self.distributed_setup = self._build_distributed_setup(world_size)
        self.distributed_config = self.distributed_setup.strategy_config
        self.device_mesh = self.distributed_setup.mesh_context.device_mesh
        self.moe_mesh = self.distributed_setup.mesh_context.moe_mesh
        self.data_parallel_mesh = get_flat_mesh(self.device_mesh, "dp") if self.device_mesh is not None else None

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile
            else entropy_from_logits
        )

    def _validate_precision_and_parallelism(self) -> None:
        from verl.utils.torch_dtypes import PrecisionType

        if not PrecisionType.is_bf16(self.engine_config.dtype):
            raise ValueError(f"AutoModel training supports only bfloat16 compute, got {self.engine_config.dtype!r}")
        if self.engine_config.pp_size != 1:
            raise NotImplementedError("AutoModel pipeline parallelism is not integrated with veRL")
        if self.engine_config.cp_size != 1:
            raise NotImplementedError(
                "AutoModel context parallelism requires a CP-local additive loss; "
                "veRL loss callbacks currently consume complete sequences"
            )
        if getattr(self.engine_config, "grad_offload", False):
            raise NotImplementedError("AutoModel veRL has not integrated gradient offload")
        if getattr(getattr(self, "model_config", None), "enable_activation_offload", False):
            raise NotImplementedError("AutoModel veRL has not integrated activation offload")
        router_replay = getattr(self.engine_config, "router_replay", None)
        router_replay_mode = (
            router_replay.get("mode", "disabled")
            if isinstance(router_replay, dict)
            else getattr(router_replay, "mode", "disabled")
        )
        if router_replay_mode != "disabled":
            raise NotImplementedError("AutoModel veRL has not integrated router replay")
        if getattr(self.engine_config, "param_offload", False) and self.engine_config.distributed_strategy != "fsdp2":
            raise NotImplementedError("AutoModel parameter offload requires the FSDP2 distributed strategy")
        for name in ("mp_param_dtype", "mp_output_dtype"):
            value = getattr(self.engine_config, name)
            if not PrecisionType.is_bf16(value):
                raise ValueError(f"AutoModel {name} must be bfloat16, got {value!r}")
        if not (
            PrecisionType.is_fp32(self.engine_config.mp_reduce_dtype)
            or PrecisionType.is_bf16(self.engine_config.mp_reduce_dtype)
        ):
            raise ValueError(
                f"AutoModel mp_reduce_dtype must be fp32 or bf16, got {self.engine_config.mp_reduce_dtype!r}"
            )
        if not (
            PrecisionType.is_fp32(self.engine_config.model_dtype)
            or PrecisionType.is_bf16(self.engine_config.model_dtype)
        ):
            raise ValueError(f"AutoModel model_dtype must be fp32 or bf16, got {self.engine_config.model_dtype!r}")
        for name in ("exp_avg_dtype", "exp_avg_sq_dtype", "master_weight_dtype"):
            value = getattr(self.optimizer_config, name, None)
            if PrecisionType.is_fp16(value):
                raise ValueError(f"AutoModel {name} cannot use float16")
        backend_config = getattr(self.engine_config, "backend_config", None) or {}
        gate_precision = (
            backend_config.get("gate_precision")
            if isinstance(backend_config, dict)
            else getattr(backend_config, "gate_precision", None)
        )
        if PrecisionType.is_fp16(gate_precision):
            raise ValueError("AutoModel backend_config.gate_precision cannot use float16")
        moe_config = getattr(self.engine_config, "moe_config", None) or {}
        lm_head_precision = (
            moe_config.get("lm_head_precision")
            if isinstance(moe_config, dict)
            else getattr(moe_config, "lm_head_precision", None)
        )
        if PrecisionType.is_fp16(lm_head_precision):
            raise ValueError("AutoModel moe_config.lm_head_precision cannot use float16")

    def _build_distributed_setup(self, world_size: int) -> DistributedSetup:
        from torch.distributed.fsdp import MixedPrecisionPolicy

        from verl.utils.torch_dtypes import PrecisionType

        if self.engine_config.distributed_strategy == "fsdp2":
            strategy = FSDP2Config(
                sequence_parallel=self.engine_config.sequence_parallel,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=PrecisionType.to_dtype(self.engine_config.mp_param_dtype),
                    reduce_dtype=PrecisionType.to_dtype(self.engine_config.mp_reduce_dtype),
                    output_dtype=PrecisionType.to_dtype(self.engine_config.mp_output_dtype),
                    cast_forward_inputs=True,
                ),
                defer_fsdp_grad_sync=self.engine_config.defer_fsdp_grad_sync,
            )
        elif self.engine_config.distributed_strategy == "megatron_fsdp":
            strategy = MegatronFSDPConfig()
        else:
            strategy = DDPConfig()

        return DistributedSetup.build(
            strategy=strategy,
            parallelism_sizes=ParallelismSizes(
                dp_replicate_size=self.engine_config.dp_replicate_size,
                tp_size=self.engine_config.tp_size,
                pp_size=self.engine_config.pp_size,
                cp_size=self.engine_config.cp_size,
                ep_size=self.engine_config.ep_size,
            ),
            moe_parallel_config=self.engine_config.moe_config if self.engine_config.ep_size > 1 else None,
            activation_checkpointing=self.engine_config.activation_checkpointing,
            world_size=world_size,
        )

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def initialize(self):
        """Build the model and veRL-owned optimizer, scheduler, and checkpointer."""
        self.module = self._build_model()
        log_gpu_memory_usage("After Automodel model build", logger=logger)

        if not self.engine_config.forward_only:
            self.optimizer = self._build_optimizer(self.module)
            self.optimizer = maybe_shard_optimizer(self.module, self.optimizer, self.distributed_config)
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer)
        else:
            self.optimizer = None
            self.lr_scheduler = None

        pad_token_id = getattr(self.model_config.tokenizer, "pad_token_id", 0) or 0
        self.execution_engine = Engine(
            self.module,
            device=torch.device(get_device_name(), get_device_id()),
            mesh_context=self.distributed_setup.mesh_context,
            microbatch_size=1,
            collate_fn=collate_prebatched,
            padding_token_id=pad_token_id,
            context_fn=lambda _inputs: torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16),
            defer_fsdp_grad_sync=self.engine_config.defer_fsdp_grad_sync,
            optimizers=self.optimizer,
            # veRL advances and checkpoints its scheduler separately.
            lr_schedulers=None,
            max_grad_norm=self.optimizer_config.clip_grad,
        )
        self._build_checkpointer()

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)
        torch.cuda.empty_cache()

    def _build_model(self):
        from nemo_automodel.components.models.common import BackendConfig

        kwargs = {
            "attn_implementation": self.engine_config.attn_implementation,
            "config": self.model_config.hf_config,
            "distributed_setup": self.distributed_setup,
            "has_packed_sequence": self.engine_config.use_remove_padding,
            "trust_remote_code": self.model_config.trust_remote_code,
            "use_liger_kernel": self.model_config.use_liger,
        }

        from verl.utils.torch_dtypes import PrecisionType

        kwargs["torch_dtype"] = PrecisionType.to_dtype(self.engine_config.model_dtype)
        kwargs["force_hf"] = (
            self.engine_config.ep_size == 1
            and self.engine_config.use_remove_padding
            and self.engine_config.attn_implementation == "flash_attention_2"
        )
        if self.engine_config.backend_config and not kwargs["force_hf"]:
            kwargs["backend"] = BackendConfig(**self.engine_config.backend_config)
        if self.engine_config.enable_fp8:
            from nemo_automodel.components.quantization.fp8 import FP8Config

            kwargs["fp8_config"] = FP8Config()
        if self.engine_config.enable_compile:
            from nemo_automodel.components.utils.compile_utils import CompileConfig

            kwargs["compile_config"] = CompileConfig()

        from transformers import AutoModelForCausalLM

        from verl.utils.transformers_compat import get_auto_model_for_vision2seq

        hf_model_class = get_hf_auto_model_class(self.model_config.hf_config)
        if hf_model_class is AutoModelForCausalLM:
            model_class = NeMoAutoModelForCausalLM
        elif hf_model_class is get_auto_model_for_vision2seq():
            model_class = NeMoAutoModelForImageTextToText
        else:
            raise NotImplementedError(
                f"AutoModel veRL supports causal LM and image-text model configs, got {hf_model_class.__name__}"
            )
        model = model_class.from_pretrained(
            self.model_config.local_path or self.model_config.path,
            **kwargs,
        )
        self._packing_layout = None
        if self.engine_config.use_remove_padding:
            from nemo_automodel.components.models.common.packing import configure_packing, get_attn_implementation

            live_model = getattr(model, "module", model)
            custom_attention = getattr(getattr(live_model, "backend", None), "attn", None)
            loaded_attention = get_attn_implementation(None, model=model)
            if custom_attention == "te":
                self._packing_layout = "thd"
            elif custom_attention is None and loaded_attention == "flash_attention_2":
                if self.engine_config.ep_size > 1:
                    raise NotImplementedError(
                        "FlashAttention indexed packing does not support expert parallelism; use a TE THD model"
                    )
                configure_packing("flash_attention_2")
                self._packing_layout = "indexed_mask"
            else:
                raise NotImplementedError(
                    "packed AutoModel batches require a live custom TE backend or resolved HF FlashAttention 2; "
                    f"got custom={custom_attention!r}, resolved={loaded_attention!r}"
                )
        self._use_indexed_packing = self._packing_layout == "indexed_mask"
        return model

    def _build_optimizer(self, module):
        """Build the configured optimizer through AutoModel's public optimizer API."""
        from verl.utils.torch_dtypes import PrecisionType

        config = self.optimizer_config
        opt_dict = {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "eps": config.eps,
            "betas": list(config.betas),
        }

        if config.master_weights:
            opt_dict["master_weights"] = config.master_weights
        if config.store_param_remainders:
            opt_dict["store_param_remainders"] = config.store_param_remainders

        _short_to_torch = {"bf16": torch.bfloat16, "fp32": torch.float32}
        for attr in ("exp_avg_dtype", "exp_avg_sq_dtype", "master_weight_dtype"):
            val = getattr(config, attr, None)
            if val is not None:
                opt_dict[attr] = _short_to_torch.get(val, val)

        if config.override_optimizer_config:
            opt_dict.update(config.override_optimizer_config)
        for name in ("exp_avg_dtype", "exp_avg_sq_dtype", "master_weight_dtype"):
            if PrecisionType.is_fp16(opt_dict.get(name)):
                raise ValueError(f"AutoModel optimizer override {name} cannot use float16")

        optimizers = build_optimizer(
            module,
            (f"{config.optimizer_impl}.{config.optimizer}", opt_dict),
            device_mesh=self.device_mesh,
        )
        assert len(optimizers) == 1, f"Expected 1 optimizer, got {len(optimizers)}"
        return optimizers[0]

    def _build_lr_scheduler(self, optimizer):
        cfg = self.optimizer_config
        total_steps = cfg.total_training_steps
        num_warmup_steps = cfg.lr_warmup_steps

        if num_warmup_steps <= 0:
            num_warmup_steps = int(cfg.lr_warmup_steps_ratio * total_steps)

        base_lr = cfg.lr
        init_lr_ratio = cfg.init_lr_ratio if cfg.init_lr_ratio is not None else 0.1
        min_lr_ratio = cfg.min_lr_ratio if cfg.min_lr_ratio is not None else 0.01

        if self.rank == 0:
            print(
                f"Automodel LR Scheduler: total_steps={total_steps}, warmup={num_warmup_steps}, "
                f"decay_style={cfg.lr_scheduler_type}, init_lr={base_lr * init_lr_ratio:.2e}, "
                f"max_lr={base_lr:.2e}, min_lr={base_lr * min_lr_ratio:.2e}"
            )

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=base_lr * init_lr_ratio,
            max_lr=base_lr,
            min_lr=base_lr * min_lr_ratio,
            lr_warmup_steps=num_warmup_steps,
            lr_decay_steps=total_steps,
            lr_decay_style=cfg.lr_scheduler_type,
            start_wd=cfg.weight_decay,
            end_wd=cfg.weight_decay,
            wd_incr_steps=total_steps,
            wd_incr_style=getattr(cfg, "wd_incr_style", "constant"),
        )
        return scheduler

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        dp_size = self.get_data_parallel_size()
        num_micro_batches = len(micro_batches)
        datums = []
        prepared_batches = []
        output_args_list = []
        for index, micro_batch in enumerate(micro_batches):
            micro_batch = micro_batch.to(get_device_id())
            model_inputs, output_args = self.prepare_model_inputs(micro_batch)
            token_template = model_inputs.get("input_ids", model_inputs.get("inputs_embeds"))
            if not isinstance(token_template, torch.Tensor) or token_template.ndim < 2:
                raise ValueError("AutoModel inputs must contain a batched token tensor")
            token_shape = token_template.shape[:2]
            token_count = token_template.shape[0] * token_template.shape[1]
            if token_count == 0:
                raise ValueError("AutoModel cannot execute an empty microbatch")

            # The sum is 1 / dp_size over the complete window. Engine therefore
            # normalizes the caller's already-scaled veRL scalar without changing
            # its gradient under averaged or summed data-parallel reductions.
            weights = torch.full(
                token_shape,
                1.0 / (dp_size * num_micro_batches * token_count),
                dtype=torch.float32,
                device=token_template.device,
            )
            datums.append(
                Datum(
                    model_inputs=model_inputs,
                    loss_fn_inputs={
                        "weights": weights,
                        "_verl_microbatch_index": torch.tensor(index, device=token_template.device),
                    },
                    loss_fn_input_layouts={
                        "weights": LossInputLayout.PER_TOKEN,
                        "_verl_microbatch_index": LossInputLayout.REPLICATED,
                    },
                )
            )
            prepared_batches.append(micro_batch)
            output_args_list.append(output_args)

        def engine_loss_fn(raw_output, loss_inputs):
            index = int(loss_inputs["_verl_microbatch_index"].item())
            micro_batch = prepared_batches[index]
            model_output = self.prepare_model_outputs(raw_output, output_args_list[index], micro_batch)
            if loss_function is None:
                if not forward_only:
                    raise ValueError("training requires a loss function")
                loss = torch.ones((), device=loss_inputs["weights"].device)
                metrics = {}
            else:
                loss, metrics = loss_function(
                    model_output=model_output,
                    data=micro_batch,
                    dp_group=self.get_data_parallel_group(),
                )
            record = {"loss": loss.detach().item(), "metrics": metrics}
            if forward_only or tu.get_non_tensor_data(data=micro_batch, key="return_model_output", default=False):
                record["model_output"] = model_output
            return loss / dp_size, [record]

        if forward_only:
            result = self.execution_engine.forward(datums, engine_loss_fn)
        else:
            result = self.execution_engine.forward_backward(
                datums,
                engine_loss_fn,
                accumulate_gradients=True,
            )
        return postprocess_batch_func(output_lst=result.loss_fn_outputs, indices=indices, data=data)

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        grad_norm = self.execution_engine.optim_step().grad_norm
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

    def lr_scheduler_step(self):
        """Step Automodel's OptimizerParamScheduler and return current LR."""
        self.lr_scheduler.step(increment=1)
        lr = self.optimizer.param_groups[0]["lr"]
        return lr

    def get_data_parallel_rank(self):
        if self.data_parallel_mesh is not None:
            return self.data_parallel_mesh.get_local_rank()
        return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        if self.data_parallel_mesh is not None:
            return self.data_parallel_mesh.size()
        return torch.distributed.get_world_size()

    def get_data_parallel_group(self):
        if self.data_parallel_mesh is not None:
            return self.data_parallel_mesh.get_group()
        return torch.distributed.group.WORLD

    def is_mp_src_rank_with_outputs(self):
        if self.device_mesh is not None and "tp" in self.device_mesh.mesh_dim_names:
            if self.device_mesh["tp"].size() > 1:
                return self.device_mesh.get_local_rank("tp") == 0
        return True

    def train_mode(self, **kwargs):
        return AutomodelTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        return AutomodelEvalModeCtx(self, **kwargs)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()
        assert device in (device_name, "cpu")

        if device == device_name:
            if model:
                load_automodel_model_to_gpu(self.module)
            if not self.engine_config.forward_only and optimizer and self.optimizer is not None:
                load_automodel_optimizer(self.optimizer, get_device_id())
            gc.collect()
        elif device == "cpu":
            if model:
                offload_automodel_model_to_cpu(self.module)
            if not self.engine_config.forward_only and optimizer and self.optimizer is not None:
                offload_automodel_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def _build_checkpointer(self):
        if self.checkpoint_config and self.checkpoint_config.get("async_save", False):
            raise NotImplementedError("AutoModel veRL checkpointing does not support async_save")
        self.checkpoint_manager = BaseCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
        )
        supported_contents = {"model", "optimizer", "extra", "hf_model"}
        configured_contents = set(self.checkpoint_manager.checkpoint_save_contents) | set(
            self.checkpoint_manager.checkpoint_load_contents
        )
        unsupported_contents = configured_contents - supported_contents
        if unsupported_contents:
            raise ValueError(f"AutoModel checkpoint contents are unsupported: {sorted(unsupported_contents)}")
        if self.checkpoint_manager.should_save_lora_only:
            raise NotImplementedError("AutoModel veRL checkpointing does not yet support save_lora_only")

        ckpt_config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="checkpoints/",
            model_save_format="safetensors",
            model_cache_dir=HF_HUB_CACHE,
            model_repo_id=self.model_config.path,
            save_consolidated="every" if self.checkpoint_manager.should_save_hf_model else False,
            is_peft=False,
        )
        self.checkpointer = Checkpointer(
            config=ckpt_config,
            dp_rank=self.get_data_parallel_rank(),
            tp_rank=get_tp_rank(self.device_mesh),
            pp_rank=get_pp_rank(self.device_mesh),
            moe_mesh=self.moe_mesh,
        )

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Save the contents selected by veRL using AutoModel as the tensor backend."""
        if hdfs_path is not None:
            raise NotImplementedError("AutoModel veRL checkpointing does not support HDFS paths")
        policy = self.checkpoint_manager
        _, checkpoint_path = policy.checkpath(local_path, hdfs_path)
        save_model = policy.should_save_model or policy.should_save_hf_model
        save_optimizer = policy.should_save_optimizer
        if save_optimizer and self.optimizer is None:
            raise ValueError("checkpoint save_contents includes optimizer, but this engine has no optimizer")

        policy.previous_global_step = global_step
        if policy.rank == 0:
            policy.ensure_checkpoint_capacity(max_ckpt_to_keep)
            os.makedirs(checkpoint_path, exist_ok=True)
        torch.distributed.barrier()

        stage_model = save_model or save_optimizer
        if stage_model:
            origin_module_device = next(self.module.parameters()).device.type
            if self._is_offload_param or origin_module_device == "cpu":
                load_automodel_model_to_gpu(self.module)
        try:
            if save_model:
                self.checkpointer.save_model(
                    self.module,
                    checkpoint_path,
                    tokenizer=policy.processing_class,
                )
            if save_optimizer:
                # veRL keeps scheduler state in the independently selectable extra payload.
                self.checkpointer.save_optimizer(self.optimizer, self.module, checkpoint_path, scheduler=None)
            if policy.should_save_extra:
                extra_state = {
                    "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                    "rng": policy.get_rng_state(),
                    "global_step": global_step,
                }
                torch.save(
                    extra_state,
                    os.path.join(
                        checkpoint_path,
                        f"extra_state_world_size_{policy.world_size}_rank_{policy.rank}.pt",
                    ),
                )
            torch.distributed.barrier()
            if policy.rank == 0:
                policy.register_checkpoint(checkpoint_path, max_ckpt_to_keep)
            torch.distributed.barrier()
        finally:
            if stage_model and self._is_offload_param:
                offload_automodel_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """Load the contents selected by veRL using AutoModel as the tensor backend."""
        if hdfs_path is not None:
            raise NotImplementedError("AutoModel veRL checkpointing does not support HDFS paths")
        policy = self.checkpoint_manager
        _, checkpoint_path = policy.checkpath(local_path, hdfs_path)
        load_model = policy.should_load_model or policy.should_load_hf_model
        load_optimizer = policy.should_load_optimizer
        if load_optimizer and self.optimizer is None:
            raise ValueError("checkpoint load_contents includes optimizer, but this engine has no optimizer")

        stage_model = load_model or load_optimizer
        if stage_model and self._is_offload_param:
            load_automodel_model_to_gpu(self.module)
        try:
            if load_model:
                if policy.should_load_model:
                    model_path = os.path.join(checkpoint_path, "model")
                    if not os.path.isdir(model_path):
                        model_path = checkpoint_path
                else:
                    model_path = os.path.join(checkpoint_path, "model", "consolidated")
                self.checkpointer.load_model(self.module, model_path)
            if load_optimizer:
                # veRL restores its scheduler from the extra payload below.
                self.checkpointer.load_optimizer(self.optimizer, self.module, checkpoint_path, scheduler=None)
            if policy.should_load_extra:
                extra_path = os.path.join(
                    checkpoint_path,
                    f"extra_state_world_size_{policy.world_size}_rank_{policy.rank}.pt",
                )
                # RNG payloads contain trusted Python and NumPy state that cannot use weights-only loading.
                extra_state = torch.load(extra_path, map_location="cpu", weights_only=False)
                if "rng" in extra_state:
                    policy.load_rng_state(extra_state["rng"])
                scheduler_state = extra_state.get("lr_scheduler")
                if scheduler_state is not None:
                    if self.lr_scheduler is None:
                        raise ValueError("checkpoint extra contains an LR scheduler, but this engine has none")
                    self.lr_scheduler.load_state_dict(scheduler_state)
                policy.previous_global_step = extra_state.get("global_step")
            torch.distributed.barrier()
        finally:
            if stage_model and self._is_offload_param:
                offload_automodel_model_to_cpu(self.module)
            if self._is_offload_optimizer and self.optimizer is not None:
                offload_automodel_optimizer(self.optimizer)

    def get_per_tensor_param(self, **kwargs):
        def param_generator():
            load_automodel_model_to_gpu(self.module)
            try:
                # DCP's public state API preserves local DTensors while normalizing
                # DDP/FSDP wrapper prefixes before custom-model adapter conversion.
                params = get_model_state_dict(self.module)
                model = (
                    self.module.module
                    if isinstance(self.module, torch.nn.parallel.DistributedDataParallel)
                    else self.module
                )
                adapter = getattr(model, "state_dict_adapter", None)
                if adapter is None:
                    params = convert_weight_keys(params, model)
                for name, param in params.items():
                    if not torch.is_tensor(param) or name.endswith("_extra_state"):
                        continue
                    if (
                        getattr(getattr(self, "engine_config", None), "ep_size", 1) > 1
                        and "expert" in name.lower()
                        and not isinstance(param, DTensor)
                    ):
                        raise RuntimeError(
                            f"EP expert weight {name!r} is not a DTensor; streaming it would export only one "
                            "rank's local experts"
                        )
                    # full_tensor is collective: every model-parallel rank must
                    # materialize and convert in the same state-dict order.
                    full_tensor = param.full_tensor() if isinstance(param, DTensor) else param
                    if adapter is None:
                        hf_pairs = [(name, full_tensor)]
                    elif callable(getattr(adapter, "convert_single_tensor_to_hf", None)):
                        hf_pairs = adapter.convert_single_tensor_to_hf(
                            name,
                            full_tensor,
                            exclude_key_regex=r".*_extra_state.*",
                            quantization=False,
                        )
                    else:
                        raise RuntimeError(
                            f"{type(model).__name__} has a state_dict_adapter without "
                            "convert_single_tensor_to_hf; streaming export cannot safely convert it"
                        )
                    yield from hf_pairs
            finally:
                if self._is_offload_param:
                    offload_automodel_model_to_cpu(self.module)

        return param_generator(), None


class AutomodelEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: AutomodelEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, AutomodelEngine)
        super().__enter__()
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, AutomodelEngine)
        # Reshard the root FSDP module
        if hasattr(self.engine.module, "reshard"):
            self.engine.module.reshard()
        super().__exit__(exc_type, exc_value, traceback)


class AutomodelTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: AutomodelEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, AutomodelEngine)
        super().__enter__()
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, AutomodelEngine)
        if self.zero_grad_on_exit or exc_type is not None:
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend=["automodel"], device=["cuda"])
class AutomodelEngineWithLMHead(AutomodelEngine):
    """Automodel engine for language model with LM head training."""

    def prepare_model_inputs(self, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        configured_remove_padding = getattr(self.engine_config, "use_remove_padding", use_remove_padding)
        if use_remove_padding != configured_remove_padding:
            raise ValueError(
                "batch use_remove_padding must match the AutoModel engine configuration used to load attention"
            )
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        temperature = micro_batch["temperature"]
        if use_fused_kernels:
            raise NotImplementedError("AutoModel veRL has not integrated VERL fused log-prob kernels")
        unsupported_outputs = [
            name
            for name in ("calculate_sum_pi_squared", "distillation_use_topk", "distillation_only")
            if tu.get_non_tensor_data(data=micro_batch, key=name, default=False)
        ]
        if unsupported_outputs:
            raise NotImplementedError(f"AutoModel veRL has not integrated output processing for {unsupported_outputs}")
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]

        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

        temperature = temperature.to(torch.float32)
        assert temperature.shape[0] == input_ids.shape[0]

        output_args = {
            "input_ids_rmpad_rolled": torch.roll(input_ids.values(), shifts=-1, dims=0),
            "temperature_rmpad": verl_F.expand_as_nested(temperature, input_ids).values(),
        }

        if use_remove_padding:
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids_rmpad = input_ids.values().unsqueeze(0)
                if position_ids.dim() == 3:
                    position_ids_rmpad = position_ids.values().unsqueeze(1)
                else:
                    position_ids_rmpad = position_ids.values().unsqueeze(0)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

            output_args["token_count"] = input_ids_rmpad.shape[1]

            model_inputs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": None,
                "position_ids": position_ids_rmpad,
            }

            # Engine owns final THD metadata and any HybridEP token equalization.
            if self._packing_layout == "thd":
                seq_lens = input_ids.offsets().diff().to(torch.int32).unsqueeze(0)
                model_inputs["qkv_format"] = "thd"
                model_inputs["seq_lens"] = seq_lens
                model_inputs["seq_lens_padded"] = seq_lens.clone()
            elif self._use_indexed_packing:
                seq_lens = input_ids.offsets().diff()
                packed_seq_ids = torch.repeat_interleave(
                    torch.arange(1, seq_lens.numel() + 1, device=input_ids_rmpad.device),
                    seq_lens.to(input_ids_rmpad.device),
                ).unsqueeze(0)
                model_inputs["attention_mask"] = packed_seq_ids

        else:
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids = micro_batch["input_ids"]
                position_ids = micro_batch["position_ids"]
                pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
                batch_size = micro_batch.batch_size[0]
                seq_len_effective = input_ids.offsets().diff()
                max_seq_len = max(seq_len_effective)

                input_ids = torch.nested.to_padded_tensor(
                    input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
                )

                if position_ids.dim() == 3:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
                    ).transpose(0, 1)
                else:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, max_seq_len)
                    )

                attention_mask = build_attention_mask_from_nested(
                    input_ids=micro_batch["input_ids"], max_seq_len=max_seq_len
                )

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
                output_args["padded_width"] = input_ids.shape[1]

            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        model_inputs["use_cache"] = False
        model_inputs.update(multi_modal_inputs)

        return model_inputs, output_args

    def _token_statistics(self, logits, targets, temperature, calculate_entropy):
        temperature = temperature.clamp(min=1e-8)
        if tuple(logits.shape[:-1]) != tuple(targets.shape) or tuple(targets.shape) != tuple(temperature.shape):
            raise ValueError(
                "logits, targets, and temperature must share token axes; "
                f"got {tuple(logits.shape)}, {tuple(targets.shape)}, and {tuple(temperature.shape)}"
            )

        if isinstance(logits, DTensor):
            local_logits = logits.to_local()
            scaled_local = local_logits / temperature.unsqueeze(-1).to(local_logits.dtype)
            scaled_logits = DTensor.from_local(
                scaled_local,
                logits.device_mesh,
                logits.placements,
                run_check=False,
                shape=logits.shape,
                stride=logits.stride(),
            )
            log_probs = vocab_parallel_log_probs(scaled_logits, targets)
            entropy = vocab_parallel_entropy(scaled_logits) if calculate_entropy else None
            return log_probs, entropy

        scaled_logits = logits / temperature.unsqueeze(-1).to(logits.dtype)
        log_probs = logprobs_from_logits(
            logits=scaled_logits,
            labels=targets,
            inplace_backward=not calculate_entropy,
        )
        if not calculate_entropy:
            return log_probs, None
        if self.engine_config.entropy_checkpointing:
            entropy = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, scaled_logits)
        elif self.engine_config.entropy_from_logits_with_chunking:
            entropy = self.compute_entropy_from_logits(
                scaled_logits,
                chunk_size=self.engine_config.entropy_from_logits_chunk_size,
            )
        else:
            entropy = self.compute_entropy_from_logits(scaled_logits)
        return log_probs, entropy

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        if use_fused_kernels:
            raise NotImplementedError("AutoModel veRL has not integrated VERL fused log-prob kernels")
        if pad_mode != DatasetPadMode.NO_PADDING:
            raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        if isinstance(output, torch.Tensor):
            from types import SimpleNamespace

            output = SimpleNamespace(logits=output)

        input_ids = micro_batch["input_ids"]
        cu_seqlens = input_ids.offsets()
        if use_remove_padding:
            token_count = output_args["token_count"]
            logits = output.logits.squeeze(0)[:token_count]
            log_probs_rmpad, entropy_rmpad = self._token_statistics(
                logits,
                output_args["input_ids_rmpad_rolled"],
                output_args["temperature_rmpad"],
                calculate_entropy,
            )
            log_probs = torch.nested.nested_tensor_from_jagged(log_probs_rmpad, cu_seqlens)
            entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens) if calculate_entropy else None
        else:
            original_width = output_args["padded_width"]
            logits = output.logits[:, :original_width]
            seq_lengths = cu_seqlens.diff()
            valid_mask = torch.arange(original_width, device=logits.device).unsqueeze(0) < seq_lengths.to(
                logits.device
            ).unsqueeze(1)
            if isinstance(logits, DTensor):
                local_logits = logits.to_local()[valid_mask]
                flat_placements = []
                for placement in logits.placements:
                    if isinstance(placement, Shard):
                        if placement.dim % logits.ndim != logits.ndim - 1:
                            raise NotImplementedError(
                                "unpacked AutoModel logits only support sharding on the vocabulary dimension"
                            )
                        flat_placements.append(Shard(1))
                    elif isinstance(placement, Replicate):
                        flat_placements.append(placement)
                    else:
                        raise NotImplementedError(
                            f"unpacked AutoModel logits do not support DTensor placement {placement!r}"
                        )
                vocab_size = logits.shape[-1]
                logits = DTensor.from_local(
                    local_logits,
                    logits.device_mesh,
                    tuple(flat_placements),
                    run_check=False,
                    shape=(output_args["input_ids_rmpad_rolled"].numel(), vocab_size),
                    stride=(vocab_size, 1),
                )
            else:
                logits = logits[valid_mask]
            log_probs_rmpad, entropy_rmpad = self._token_statistics(
                logits,
                output_args["input_ids_rmpad_rolled"],
                output_args["temperature_rmpad"],
                calculate_entropy,
            )
            log_probs = torch.nested.nested_tensor_from_jagged(log_probs_rmpad, cu_seqlens)
            entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens) if calculate_entropy else None

        model_output = {"log_probs": log_probs}
        if calculate_entropy:
            model_output["entropy"] = entropy
        return model_output
