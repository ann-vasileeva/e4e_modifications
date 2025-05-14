import torch
import torch.nn as nn
from face_parsing.model import BiSeNet

class SegHelper(nn.Module):
    def __init__(self):
        super(SegHelper, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = BiSeNet(n_classes=19).to(self.device)
        save_pth = '/home/ayavasileva/face_parsing/res/cp/79999_iter.pth'
        self.net.load_state_dict(torch.load(save_pth, map_location=self.device))
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()
        
    def get_logits(self, y):
        b_sz, _, h, w = y.shape
        logits = self.net(y)[0]
        return logits
        