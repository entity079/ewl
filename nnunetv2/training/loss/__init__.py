from .compound_losses import DC_and_CE_loss, DC_and_BCE_loss, DC_and_topk_loss, DC_EWL_and_CE_loss
from .dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from .EWL import Euclidean_Weighted_Loss
from .robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from .deep_supervision import DeepSupervisionWrapper