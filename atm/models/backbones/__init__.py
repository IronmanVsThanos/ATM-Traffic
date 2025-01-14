from .dino_v2 import DinoVisionTransformer
from .atms_dinov2 import atmsDinoVisionTransformer
from .atms_eva_02 import atmsEVA2
try:
    from .atms_convnext import atmsConvNeXt
except:
    print('Fail to import atmsConvNeXt, if you need to use it, please install mmpretrain')
from .clip import CLIPVisionTransformer
from .atms_sam_vit import atmsSAMViT
from .sam_vit import SAMViT
from .atms_clip import atmsCLIPVisionTransformer
