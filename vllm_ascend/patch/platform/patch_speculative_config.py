from typing import Literal, get_args

from transformers import DeepseekV2Config, PretrainedConfig
from vllm.config import speculative as _speculative_config
from vllm.config.speculative import SpeculativeConfig

_orig_post_init = SpeculativeConfig.__post_init__
_orig_hf_config_override = SpeculativeConfig.hf_config_override

# GLM-5.3 checkpoints carry an MTP layer (``num_nextn_predict_layers``), but
# their text model_type is registered downstream and is not in upstream's
# ``MTPModelTypes`` tuple.  ``SpeculativeConfig`` therefore rejects
# ``method="mtp"`` before the draft model is ever built.  Extend the runtime
# tuple so the generic MTP path (and the fork's Glm5NextMTP head) can be used.
_GLM5_NEXT_MTP_MODEL_TYPES = ("glm5_next", "glm5_next_text")
_known_mtp_types = get_args(_speculative_config.MTPModelTypes)
_missing_mtp_types = tuple(
    name for name in _GLM5_NEXT_MTP_MODEL_TYPES if name not in _known_mtp_types
)
if _missing_mtp_types:
    _speculative_config.MTPModelTypes = Literal[
        (*_known_mtp_types, *_missing_mtp_types)
    ]


# Transformers 5.14 inherited a hidden_size % num_heads check from Llama in
# DeepseekV2Config. K3 MLA has independent projection/head dimensions (e.g.
# hidden_size=7168, num_heads=96), so that MHA constraint does not apply.
# strict stores unbound validators; patch that entry, not all config validation.
if hasattr(DeepseekV2Config, "__class_validators__"):
    _orig_validate_architecture = DeepseekV2Config.validate_architecture

    def _validate_dspark_architecture(config):
        if config.model_type != "k3_dspark":
            _orig_validate_architecture(config)

    DeepseekV2Config.__class_validators__ = [
        _validate_dspark_architecture if validator is _orig_validate_architecture else validator
        for validator in DeepseekV2Config.__class_validators__
    ]


def _normalize_legacy_qwen3_dspark_config(hf_config: PretrainedConfig) -> PretrainedConfig:
    hf_config = _orig_hf_config_override(hf_config)
    # The draft config is built from the target checkpoint, so the MTP path
    # has to replace the target architecture with the registered MTP head.
    # ``Glm5NextConfig`` mirrors this onto ``text_config`` and leaves
    # ``architectures`` unmirrored.
    if hf_config.model_type in _GLM5_NEXT_MTP_MODEL_TYPES:
        text_config = getattr(hf_config, "text_config", hf_config)
        if getattr(text_config, "num_nextn_predict_layers", 0):
            hf_config.update({"architectures": ["Glm5NextMTPModel"]})
    architectures = hf_config.architectures or ()
    if hf_config.model_type == "qwen3" and "DSparkDraftModel" in architectures:
        dflash_config = hf_config.dflash_config
        hf_config.update(
            {
                "architectures": ["Qwen3DSparkModel"],
                "mask_token_id": dflash_config["mask_token_id"],
                "target_layer_ids": dflash_config["target_layer_ids"],
            }
        )
    return hf_config


def _dspark_post_init(self):
    _orig_post_init(self)
    if self.use_dspark():
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


SpeculativeConfig.hf_config_override = staticmethod(_normalize_legacy_qwen3_dspark_config)
SpeculativeConfig.__post_init__ = _dspark_post_init
