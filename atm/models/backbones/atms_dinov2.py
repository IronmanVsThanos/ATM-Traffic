from mmseg.models.builder import BACKBONES, MODELS
from .atms import atms
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train


@BACKBONES.register_module()
class atmsDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        atms_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.atms: atms = MODELS.build(atms_config)

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape  #4,3,512,512
        H, W = h // self.patch_size, w // self.patch_size # patch_size=16  H,W=32,32
        # print(x.shape)
        x = self.prepare_tokens_with_masks(x, masks)  # 4,1025,1024
        # print(x.shape)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x) # 4,1025,1024
            # print(x.shape)
            x = self.atms.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )# 4,1025,1024
            # print(x.shape)
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
            # print(x.shape)
        return self.atms.return_auto(outs)

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["atms"])
        set_train(self, ["atms"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "atm" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
