import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
from PIL import Image
import numpy as np
import facer
from face_parsing.model import BiSeNet
from face_parsing.test_net import vis_parsing_maps
import matplotlib.pyplot as plt
import time
# from sfe.modelings.farl.farl import Masker

import metrics
from . import intersection_metrics

class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.debug = False #change for no debug
        
        # Initialize models
        # self._init_profiler()
        self._init_models()
        
        # Transforms
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.resize_transform = transforms.Resize((512, 512))
        self.inv_resize_transform = transforms.Resize((256, 256))
        # self.default_cmap = plt.get_cmap('Set3')
        
        self._freeze_models()
        self._init_buffers()  

    def _init_models(self):
        """Initialize all models with proper device placement"""
        # BiSeNet model
        self.net = BiSeNet(n_classes=19).to(self.device)
        save_pth = '/home/ayavasileva/face_parsing/res/cp/79999_iter.pth'
        self.net.load_state_dict(torch.load(save_pth, map_location=self.device))
        
        # Facer models
        self.face_detector = facer.face_detector(
            name='retinaface/mobilenet',
            model_path='/home/ayavasileva/mobilenet0.25_Final.pth',
            device=self.device,
            threshold=0.55
        )
        self.face_parser = facer.face_parser(
            'farl/celebm/448',
            model_path='/home/ayavasileva/face_parsing.farl.celebm.main_ema_181500_jit.pt',
            device=self.device
        )

    def _freeze_models(self):
        """Freeze all model parameters"""
        for model in [self.face_parser, self.face_detector, self.net]:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

    def _init_buffers(self):
        """Initialize reusable buffers to avoid memory allocation during forward pass"""
        self._buffer_size = 4  # Adjust based on expected max batch size
        self._seg_logits_buffer = torch.zeros(
            (self._buffer_size, 19, 512, 512), 
            device=self.device
        )
        self._zero_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        
    def replace_classes(self, parsing):
        parsing[parsing == 3] = 2  # brow
        parsing[parsing == 5] = 4  # eye
        parsing[parsing == 8] = 7  # ear
        return parsing
    
    def get_logits_sum_fast(self, logits):
        new_logits = logits.clone().detach()
        with torch.no_grad():
            new_logits[:, 2] = torch.max(new_logits[:, 3], new_logits[:, 2])  # Brows (2+3)
            new_logits[:, 3].zero_()

            new_logits[:, 4] = torch.max(new_logits[:, 4], new_logits[:, 5])  # Eyes (4+5)
            new_logits[:, 5].zero_()

            new_logits[:, 7] = torch.max(new_logits[:, 7], new_logits[:, 8])  # Ears (7+8)
            new_logits[:, 8].zero_()
        return new_logits
    
    def get_logits_sum_fast_facer(self, logits):
        new_logits = logits.clone().detach()
        with torch.no_grad():
            new_logits[:, 4] = torch.max(new_logits[:, 5], new_logits[:, 4])# ? (4+5)
            new_logits[:, 5].zero_()

            new_logits[:, 6] = torch.max(new_logits[:, 6], new_logits[:, 7]) # Brows (6+7)
            new_logits[:, 7].zero_()

            new_logits[:, 8] = torch.max(new_logits[:, 8], new_logits[:, 9])   # Eyes (8+9)
            new_logits[:, 9].zero_()
        return new_logits


    @torch.inference_mode()
    def get_facer_image(self, y, i):
        y_img = y * 255
        y = torch.clamp(y, min=0, max=1)
        
        try:
            with torch.no_grad():
                faces = self.face_detector(y_img)
                for key in faces:
                    faces[key] = faces[key][:1]
                faces = self.face_parser(y, faces)
                seg_logits = self.get_logits_sum_fast_facer(faces["seg"]["logits"])
                faces["seg"]["logits"] = seg_logits
                
                if i < 2:
                    image = facer.bchw2hwc(facer.draw_bchw(y_img.detach(), faces)).to(torch.uint8)
                    image = Image.fromarray(image.cpu().numpy())
                else:
                    image = None
                    
            return seg_logits, image
            
        except Exception as e:
            print(f"Facer processing failed for sample {i}: {str(e),faces['scores']}")
            print(y.max())
            print(y.min())
            numb = torch.randint(20, (1,))
            torch.save(y.detach().cpu(), f"/home/ayavasileva/strange_y_{numb.item()}.pt")
            return self._seg_logits_buffer[:1], self._zero_image

    def _process_batch(self, y_new, y_hat_new, y_, y_hat_, validation):
        b_sz = y_new.size(0)
        
        # Pre-allocate result tensors
        iou_res = torch.zeros(b_sz, device=self.device)
        dice_res = torch.zeros(b_sz, device=self.device)
        farl_iou_res = torch.zeros(b_sz, device=self.device)
        farl_dice_res = torch.zeros(b_sz, device=self.device)
        
        seg_loss = 0
        seg_img, seg_orig = [], []
        facer_img, facer_orig = [], []
        
        for i in range(b_sz):
            # Process with BiSeNet
            faces_init_logits = self.net(y_new[i].unsqueeze(0))[0]
            faces_e4e_logits = self.net(y_hat_new[i].unsqueeze(0))[0]
                
            seg_loss += F.mse_loss(faces_e4e_logits, faces_init_logits)
                
            faces_init_logits_ = self.get_logits_sum_fast(faces_init_logits)
            faces_e4e_logits_ = self.get_logits_sum_fast(faces_e4e_logits)
                
            # Calculate metrics
            # with torch.no_grad():
            iou_res[i] = intersection_metrics.metrics_torch(
                faces_init_logits_, faces_e4e_logits_, "iou"
            )
            dice_res[i] = intersection_metrics.metrics_torch(
                faces_init_logits_, faces_e4e_logits_, "dice"
            )

            if validation:
                 # Process with facer
                init_logits, facer_init_im = self.get_facer_image(y_[i].unsqueeze(0), i)
                new_logits, facer_new_im = self.get_facer_image(y_hat_[i].unsqueeze(0), i)
                
                farl_iou_res[i] = intersection_metrics.metrics_torch(init_logits, new_logits, "iou")
                farl_dice_res[i] = intersection_metrics.metrics_torch(init_logits, new_logits, "dice")

                # Prepare visualization images
                parsing = self.replace_classes(
                    faces_e4e_logits[0].squeeze(0).detach().cpu().numpy().argmax(0)
                )
                parsing_init = self.replace_classes(
                    faces_init_logits.squeeze(0).detach().cpu().numpy().argmax(0)
                )

                seg_img.append(
                    vis_parsing_maps(
                        transforms.functional.to_pil_image(y_hat_[i]),
                        parsing, stride=1, save_im=False
                    ).transpose(0, 2)
                )
                seg_orig.append(
                    vis_parsing_maps(
                        transforms.functional.to_pil_image(y_[i]),
                        parsing_init, stride=1, save_im=False
                    ).transpose(0, 2)
                )

                if i < 2:
                    facer_orig.append(facer_init_im.resize((256, 256)))
                    facer_img.append(facer_new_im.resize((256, 256)))

        if validation and seg_img:
            seg_img = torch.stack(seg_img).to(self.device)
            seg_orig = torch.stack(seg_orig).to(self.device)
        else:
            seg_img = seg_orig = None
            
        return (
            seg_loss / b_sz,
            iou_res.mean(),
            dice_res.mean(),
            [seg_orig, seg_img, iou_res.cpu(), dice_res.cpu()],
            [facer_orig, facer_img, farl_iou_res.cpu(), farl_dice_res.cpu()]
        )

    def forward(self, y_hat, y, validation=False):
        
        # Denormalize and resize
        y_ = self.resize_transform(y) / 2 + 0.5
        y_hat_ = self.resize_transform(y_hat) / 2 + 0.5
        
        # Normalize for network input
        y_new = self.normalize(y_)
        y_hat_new = self.normalize(y_hat_)
        
        result = self._process_batch(y_new, y_hat_new, y_, y_hat_, validation)
        
        return result