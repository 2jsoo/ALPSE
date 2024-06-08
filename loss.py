## Import Library
from import_library import *
from utils import *
import warnings
warnings.filterwarnings(action="ignore")


class YoloLoss(nn.Module):
    def __init__(self, S = 9, B = 2, C = 1, lambda_class=0, lambda_noobj=0.5, lambda_obj=1, lambda_iou=1, iou_type='diou', reduction='sum'):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C    

        self.lambda_noobj = lambda_noobj
        self.lambda_iou = lambda_iou
        self.lambda_class = lambda_class
        self.lambda_obj = lambda_obj

        self.iou_type = iou_type

        self.reduction = reduction


    def forward(self, predictions, target):
        # predictions : (#, 9, 7) -> 7 : class0 confidence_score1 xmid1 width1 confidence_score2 xmid2 width2

        if self.iou_type == 'diou':
            iou1 = cal_diou(predictions[..., self.C+1:self.C+3], target[..., self.C+1:self.C+3]) 
            iou2 = cal_diou(predictions[..., self.C+4:self.C+6], target[..., self.C+1:self.C+3])
        else:
            iou1 = cal_iou(predictions[..., self.C+1:self.C+3], target[..., self.C+1:self.C+3])
            iou2 = cal_iou(predictions[..., self.C+4:self.C+6], target[..., self.C+1:self.C+3])
        

        iou12 = torch.cat([iou1.unsqueeze(0), iou2.unsqueeze(0)], dim=0) 
        iou_max, bestbox_indices = torch.max(iou12, dim=0) 
        obj_exist_box = target[..., self.C:self.C+1]

        ############ 1d DIoU loss ############
        if self.reduction == 'sum':
            iou_loss = ((1-iou_max) * obj_exist_box).sum()
        elif self.reduction in ['mean']:
            iou_loss = ((1-iou_max) * obj_exist_box).mean()


        ############ Confidence loss ############
        ## Conf1 (object) ##
        predict_conf = obj_exist_box * (bestbox_indices * predictions[..., self.C+3:self.C+4] + (1 - bestbox_indices) * predictions[..., self.C:self.C+1])
        target_conf = obj_exist_box * target[..., self.C:self.C+1]
        
        object_loss = F.binary_cross_entropy(predict_conf, target_conf, reduction=self.reduction)

        ## Conf2 (no object) ##
        target_conf_noobj = (1 - obj_exist_box) * target[..., self.C:self.C+1]

        noobject_loss = F.binary_cross_entropy((1-obj_exist_box) * predictions[..., self.C+3:self.C+4], target_conf_noobj, reduction=self.reduction)
        noobject_loss = noobject_loss + F.binary_cross_entropy((1-obj_exist_box) * predictions[..., self.C:self.C+1], target_conf_noobj, reduction=self.reduction)
        
        ############ Classification loss (If Single class, not used) ############
        predict_class = obj_exist_box * predictions[..., :self.C]
        target_class = obj_exist_box * target[..., :self.C]

        classification_loss = F.binary_cross_entropy(predict_class, target_class, reduction=self.reduction)

        final_loss = object_loss * self.lambda_obj + noobject_loss * self.lambda_noobj + classification_loss * self.lambda_class + iou_loss * self.lambda_iou
        
        return final_loss