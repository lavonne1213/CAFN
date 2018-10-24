import torch
import torch.nn as nn
from network import Conv2d


class cafn(nn.Module):
    
    def __init__(self, bn=False):
        super(cafn, self).__init__()

        self.branch1 = nn.Sequential(Conv2d(1, 16, 9, same_padding=True, dilation=1, bn=bn),
                                     nn.MaxPool2d(2),  
                                     Conv2d(16, 32, 7, same_padding=True, dilation=1, bn=bn),
                                     nn.MaxPool2d(2), 
                                     Conv2d(32, 16, 7, same_padding=True, dilation=1, bn=bn),
                                     Conv2d(16, 8, 7, same_padding=True, dilation=1, bn=bn)
                                     )
       
        self.branch2 = nn.Sequential(Conv2d(1, 20, 7, same_padding=True, dilation=2, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, dilation=2, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, dilation=2, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, dilation=2, bn=bn))
        '''
        self.branch3 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, dilation=3,bn=bn),
                                         nn.MaxPool2d(2),
                                         Conv2d(24, 48, 3, same_padding=True, dilation=3, bn=bn),
                                         nn.MaxPool2d(2),
                                         Conv2d(48, 24, 3, same_padding=True, dilation=3, bn=bn),
                                         Conv2d(24, 12, 3, same_padding=True, dilation=3, bn=bn))
        '''
        self.t_stage = nn.Sequential(Conv2d(18, 24, 3, same_padding=True, bn=bn),
                                        Conv2d(24, 32, 3, same_padding=True, bn=bn),
                                        nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, output_padding=0, bias=True),
                                        nn.PReLU(),
                                        # nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, output_padding=0, bias=True),
                                        # nn.PReLU(),
                                        Conv2d(16, 1, 1, same_padding=True, bn=bn),
                                        nn.MaxPool2d(2))

        # self.com = nn.Sequential(Conv2d( 30,  1, 1, same_padding=True, bn=bn))

        # self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        # x3 = self.branch3(im_data)
        x = torch.cat((x1, x2), 1)
        # x = self.com(x)
        x = self.t_stage(x)

        return x
