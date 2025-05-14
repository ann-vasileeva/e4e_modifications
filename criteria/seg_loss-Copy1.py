import torch
import importlib
import facer
import os
from face_parsing.model import BiSeNet
from face_parsing.test_net import vis_parsing_maps

from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from configs.paths_config import model_paths
import metrics
from . import intersection_metrics

class SegLoss(nn.Module):

    def __init__(self):
        super(SegLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        self.net.to(device)
        save_pth = os.path.join('/home/ayavasileva/face_parsing/res/cp', '79999_iter.pth')
        self.net.load_state_dict(torch.load(save_pth))
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.resize_transform = transforms.Resize((512, 512))
        self.inv_resize_transform = transforms.Resize((256, 256))
        
        # facer
        self.face_detector = facer.face_detector(name='retinaface/mobilenet', model_path='/home/ayavasileva/mobilenet0.25_Final.pth', device=device, threshold=0.7)
        self.face_parser = facer.face_parser('farl/lapa/448', model_path='/home/ayavasileva/face_parsing.farl.lapa.main_ema_136500_jit191.pt', device=device) # optional "farl/celebm/448"

        # self.seg_transforms =  transforms.Compose(
        #     [transforms.Normalize([0.0, 0.0, 0.0], [2, 2, 2]),
        #     transforms.Normalize([-0.5, -0.5, -0.5], [1.0, 1.0, 1.0])])   
    
    # @torch.inference_mode()
    # def val_parse_face_segments(self, y):
    #     faces = self.face_detector(y * 255)
    #     for key in faces.keys():
    #         faces[key] = faces[key][0:1]
    #     faces = self.face_parser(y, faces)
    #     return faces["seg"]['logits']
    
    # @torch.no_grad()
    def replace_classes(self, parsing):
        # print(parsing.shape)
        parsing[parsing == 3] = 2 #brow
        parsing[parsing == 5] = 4 #eye
        parsing[parsing == 8] = 7 #ear
        return parsing
    
    def get_facer_image(self, y):
        y_img = y * 255
        try:
            faces = self.face_detector(y_img) 
            for key in faces.keys():
                faces[key] = faces[key][0:1]
            faces = self.face_parser(y, faces)
            image = facer.bchw2hwc(facer.draw_bchw(y_img.detach(), faces)).to(torch.uint8)
            image = Image.fromarray(image.cpu().numpy())   
        except Exception as e:
            image = Image.fromarray(numpy.zeros(256,256)) #maybe test this?
        return faces["seg"]["logits"][0:1],image
        
    def forward(self, y_hat, y, validation):
        
        seg_loss, iou_res, dice_res = 0, 0, 0       
        b_sz = y_hat.shape[0]
        # y_hat, y are tensors of shape [b_sz,3, 256,256 ]
        
        # denormalize y, y_hat:
        y_ = self.resize_transform(y) / 2 + 0.5
        y_hat_ = self.resize_transform(y_hat) / 2 + 0.5
        y_hat_new =  self.normalize(y_hat_) 
        y_new =  self.normalize(y_)
        iou_res = torch.zeros(b_sz)
        dice_res = torch.zeros(b_sz)
        seg_img = torch.zeros_like(y_new)
        seg_orig = torch.zeros_like(y_new)
        
        farl_iou_res = torch.zeros(b_sz)
        farl_dice_res = torch.zeros(b_sz)
        
        facer_img = []
        facer_orig = []
        
        for i in range(b_sz):                
            faces_init_logits = self.net(y_new[i].unsqueeze(0))[0]
            faces_e4e_logits = self.net(y_hat_new[i].unsqueeze(0))[0]
            
            seg_loss += ((faces_e4e_logits - faces_init_logits)**2).mean()
            
            with torch.no_grad():
                iou_res[i] = intersection_metrics.metrics_torch(faces_init_logits, y_pred=faces_e4e_logits, metric_name="iou")
                dice_res[i] = intersection_metrics.metrics_torch(y_true=faces_init_logits, y_pred=faces_e4e_logits, metric_name="dice")
            with torch.no_grad():
                if validation:
                    parsing = self.replace_classes(faces_e4e_logits[0].squeeze(0).detach().cpu().numpy().argmax(0))
                    parsing_init = self.replace_classes(faces_init_logits.squeeze(0).detach().cpu().numpy().argmax(0)) # change classes
                    seg_img[i] = vis_parsing_maps(transforms.functional.to_pil_image(y_hat_[i]), parsing, stride=1, save_im=False).transpose(0,2)
                    seg_orig[i] = vis_parsing_maps(transforms.functional.to_pil_image(y_[i]), parsing_init, stride=1, save_im=False).transpose(0,2)
                    # facer part 
                    for i in range(b_sz):
                        init_logits, facer_init_im = self.get_facer_image(y_[i].unsqueeze(0))
                        new_logits, facer_new_im = self.get_facer_image(y_hat_[i].unsqueeze(0))
                        facer_orig.append(facer_init_im.resize((256,256))) #farl init images
                        facer_img.append(facer_new_im.resize((256,256))) #farl inverted images
                        farl_iou_res[i] = intersection_metrics.metrics_torch(init_logits, y_pred=new_logits, metric_name="iou")
                        farl_dice_res[i] = intersection_metrics.metrics_torch(y_true=init_logits, y_pred=new_logits, metric_name="dice")
        return seg_loss / b_sz, iou_res.mean(), dice_res.mean(), [self.inv_resize_transform(seg_orig), self.inv_resize_transform(seg_img), iou_res, dice_res], [facer_orig, facer_img, farl_iou_res, farl_dice_res]
# from sfe.modelings.farl.farl import Masker

# class SegLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.masker = Masker(trash=0.001)

#         for param in self.masker.parameters():
#             param.requires_grad = False

#         self.masker = self.masker.eval()
        

#     def forward(self, x, y):
#         x = x / 2 + 0.5
#         y = y / 2 + 0.5

#         x_scores = []
#         y_scores = []

#         for i, orig_tensor in enumerate(x):
#             long_img_tensor = (orig_tensor.detach().unsqueeze(0) * 255).long()
#             orig_img_tensor = orig_tensor.unsqueeze(0).cuda()


#             # with torch.inference_mode():
#             # try to find trashhlod for detecting face
#             # for detector_trash in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]:
#             #     masker = Masker(trash=detector_trash)
#             #     for param in masker.parameters():
#             #         param.requires_grad = False

#             #     masker = masker.eval()
                
#                 # faces = masker.face_detector(long_img_tensor)
#                 # if len(faces['image_ids']) != 0:
#                 #     break

#             with torch.no_grad():
#                 faces = self.masker.face_detector(long_img_tensor)
#             faces = {k:v[0].unsqueeze(0) for k, v in faces.items()}
#             print(faces)

#             if len(faces['image_ids']) == 0:
#                 assert 1 == 0, i
#             try:
#                 faces = self.masker.face_parser(orig_img_tensor, faces)
#             except Exception as e:
#                 print(e)
#                 assert 2 == 0, i


#             x_scores.append(faces['seg']['logits'])

#         for i, orig_tensor in enumerate(y):
#             long_img_tensor = (orig_tensor.detach().unsqueeze(0) * 255).long()
#             orig_img_tensor = orig_tensor.unsqueeze(0).cuda()


#             # with torch.inference_mode():
#             # try to find trashhlod for detecting face
#             # for detector_trash in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]:
#             #     masker = Masker(trash=detector_trash)
#             #     for param in masker.parameters():
#             #         param.requires_grad = False

#             #     masker = masker.eval()
                
#             #     faces = masker.face_detector(long_img_tensor)
#             #     if len(faces['image_ids']) != 0:
#             #         break

#             with torch.no_grad():
#                 faces = self.masker.face_detector(long_img_tensor)
#             faces = {k:v[0].unsqueeze(0) for k, v in faces.items()}

#             if len(faces['image_ids']) == 0:
#                 assert 3 == 0, i
#             try:
#                 faces = self.masker.face_parser(orig_img_tensor, faces)
#             except Exception as e:
#                 print(e)
#                 assert 4 == 0, i


#             y_scores.append(faces['seg']['logits'])

#         loss = 0
#         for x_item, y_item in zip(x_scores, y_scores):
#             loss = loss + (x_item - y_item).square().mean()
#         loss = loss / len(x)
#         return loss, None, None, None