import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class DynamicUNet(nn.Module):
    """ This is the Pytorch version of U-Net Architecture.
    This is not the vanilla version of U-Net.
    For more information about U-Net Architecture check the paper here.
    Link :- https://arxiv.org/abs/1505.04597

    This network is modified to have only 4 blocks depth because of
    computational limitations. 
    The input and output of this network is of the same shape.
    Input Size of Network - (1,512,512)
    Output Size of Network - (1,512,512)
        Shape Format :  (Channel, Width, Height)
    """

    def __init__(self, filters, input_channels=1, output_channels=1):
        """ Constructor for UNet class.
        Parameters:
            filters(list): Five filter values for the network.
            input_channels(int): Input channels for the network. Default: 1
            output_channels(int): Output channels for the final network. Default: 1
        """
        super(DynamicUNet, self).__init__()

        if len(filters) != 5:
            raise Exception(f"Filter list size {len(filters)}, expected 5!")

        padding = 1
        ks = 3
        # Encoding Part of Network.
        #   Block 1
        self.conv1_1 = nn.Conv2d(input_channels, filters[0], kernel_size=ks, padding=padding)
        self.conv1_2 = nn.Conv2d(filters[0], filters[0], kernel_size=ks, padding=padding)
        self.maxpool1 = nn.MaxPool2d(2)
        #   Block 2
        self.conv2_1 = nn.Conv2d(filters[0], filters[1], kernel_size=ks, padding=padding)
        self.conv2_2 = nn.Conv2d(filters[1], filters[1], kernel_size=ks, padding=padding)
        self.maxpool2 = nn.MaxPool2d(2)
        #   Block 3
        self.conv3_1 = nn.Conv2d(filters[1], filters[2], kernel_size=ks, padding=padding)
        self.conv3_2 = nn.Conv2d(filters[2], filters[2], kernel_size=ks, padding=padding)
        self.maxpool3 = nn.MaxPool2d(2)
        #   Block 4
        self.conv4_1 = nn.Conv2d(filters[2], filters[3], kernel_size=ks, padding=padding)
        self.conv4_2 = nn.Conv2d(filters[3], filters[3], kernel_size=ks, padding=padding)
        self.maxpool4 = nn.MaxPool2d(2)
        
        # Bottleneck Part of Network.
        self.conv5_1 = nn.Conv2d(filters[3], filters[4], kernel_size=ks, padding=padding)
        self.conv5_2 = nn.Conv2d(filters[4], filters[4], kernel_size=ks, padding=padding)
        self.conv5_t = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)

        # Decoding Part of Network.
        #   Block 4
        self.conv6_1 = nn.Conv2d(filters[4], filters[3], kernel_size=ks, padding=padding)
        self.conv6_2 = nn.Conv2d(filters[3], filters[3], kernel_size=ks, padding=padding)
        self.conv6_t = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        #   Block 3
        self.conv7_1 = nn.Conv2d(filters[3], filters[2], kernel_size=ks, padding=padding)
        self.conv7_2 = nn.Conv2d(filters[2], filters[2], kernel_size=ks, padding=padding)
        self.conv7_t = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        #   Block 2
        self.conv8_1 = nn.Conv2d(filters[2], filters[1], kernel_size=ks, padding=padding)
        self.conv8_2 = nn.Conv2d(filters[1], filters[1], kernel_size=ks, padding=padding)
        self.conv8_t = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        #   Block 1
        self.conv9_1 = nn.Conv2d(filters[1], filters[0], kernel_size=ks, padding=padding)
        self.conv9_2 = nn.Conv2d(filters[0], filters[0], kernel_size=ks, padding=padding)

        # Output Part of Network.
        self.conv10 = nn.Conv2d(filters[0], output_channels, kernel_size=ks, padding=padding)

    def forward(self, x):
        """ Method for forward propagation in the network.
        Parameters:
            x(torch.Tensor): Input for the network of size (1, 512, 512).

        Returns:
            output(torch.Tensor): Output after the forward propagation 
                                    of network on the input.
        """

        # Encoding Part of Network.
        #   Block 1
        conv1 = F.relu(self.conv1_1(x))
        conv1 = F.relu(self.conv1_2(conv1))
        pool1 = self.maxpool1(conv1)
        #   Block 2
        conv2 = F.relu(self.conv2_1(pool1))
        conv2 = F.relu(self.conv2_2(conv2))
        pool2 = self.maxpool2(conv2)
        #   Block 3
        conv3 = F.relu(self.conv3_1(pool2))
        conv3 = F.relu(self.conv3_2(conv3))
        pool3 = self.maxpool3(conv3)
        #   Block 4
        conv4 = F.relu(self.conv4_1(pool3))
        conv4 = F.relu(self.conv4_2(conv4))
        pool4 = self.maxpool4(conv4)

        # Bottleneck Part of Network.
        conv5 = F.relu(self.conv5_1(pool4))
        conv5 = F.relu(self.conv5_2(conv5))

        # Decoding Part of Network.
        #   Block 4
        up6 = torch.cat((self.conv5_t(conv5), conv4), dim=1)
        conv6 = F.relu(self.conv6_1(up6))
        conv6 = F.relu(self.conv6_2(conv6))
        #   Block 3
        up7 = torch.cat((self.conv6_t(conv6), conv3), dim=1)
        conv7 = F.relu(self.conv7_1(up7))
        conv7 = F.relu(self.conv7_2(conv7))
        #   Block 2
        up8 = torch.cat((self.conv7_t(conv7), conv2), dim=1)
        conv8 = F.relu(self.conv8_1(up8))
        conv8 = F.relu(self.conv8_2(conv8))
        #   Block 1
        up9 = torch.cat((self.conv8_t(conv8), conv1), dim=1)
        conv9 = F.relu(self.conv9_1(up9))
        conv9 = F.relu(self.conv9_2(conv9))

        # Output Part of Network.
        output = F.sigmoid(self.conv10(conv9))

        return output

    def summary(self, input_size=(1, 512, 512), batch_size=-1, device='cuda'):
        """ Get the summary of the network in a chart like form
        with name of layer size of the inputs and parameters 
        and some extra memory details.
        This method uses the torchsummary package.
        For more information check the link.
        Link :- https://github.com/sksq96/pytorch-summary

        Parameters:
            input_size(tuple): Size of the input for the network in
                                 format (Channel, Width, Height).
                                 Default: (1,512,512)
            batch_size(int): Batch size for the network.
                                Default: -1
            device(str): Device on which the network is loaded.
                            Device can be 'cuda' or 'cpu'.
                            Default: 'cuda'

        Returns:
            A printed output for IPython Notebooks.
            Table with 3 columns for Layer Name, Input Size and Parameters.
            torchsummary.summary() method is used.
        """
        return summary(self, input_size, batch_size, device)


#ResUnet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )
        """ Residual block과 channel size를 맞추기 위한 conv operation """
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
        ) 
    
    def forward(self, inputs):
        r = self.conv_block(inputs)
        s = self.shortcut(inputs)
        
        skip = r + s
        return skip

class ResUNet(nn.Module):
    def __init__(self, num_classes):
        super(ResUNet, self).__init__()
        self.num_classes = num_classes
        
        """ Encoder input layer """
        self.contl_1 = self.input_block(in_channels=1, out_channels=64)
        self.contl_2 = self.input_skip(in_channels=1, out_channels=64)
        
        """ Residual encoder block """
        self.resdl_1 = ResidualBlock(64, 128, 2, 1)
        self.resdl_2 = ResidualBlock(128, 256, 2, 1)
        #self.resdl_3 = ResidualBlock(256, 512, 2, 1)
        
        """ Encoder decoder skip connection """
        self.middle = ResidualBlock(256, 512, 2, 1)
        
        """ Decoder block """
        self.expnl_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, 
                                          kernel_size=2, stride=2, padding=0)
        self.expnl_1_cv = ResidualBlock(256+256, 256, 1, 1)
        self.expnl_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, 
                                          kernel_size=2, stride=2, padding=0)
        self.expnl_2_cv = ResidualBlock(128+128, 128, 1, 1)
        self.expnl_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, 
                                          kernel_size=2, stride=2, padding=0)
        self.expnl_3_cv = ResidualBlock(64+64, 64, 1, 1)
        # self.expnl_4 = nn.ConvTranspose2d(in_channels=128, out_channels=128, 
        #                                   kernel_size=2, stride=2, padding=0)
        # self.expnl_4_cv = ResidualBlock(128+64, 64, 1, 1)
        
        self.output = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
          nn.Sigmoid(),
        )
        
    def input_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    )
        return block
    
    def input_skip(self, in_channels, out_channels):
        skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        return skip                         
    
    def forward(self, X):
        contl_1_out = self.contl_1(X) # 64
        contl_2_out = self.contl_2(X) # 64
        input_out = contl_1_out + contl_2_out
        
        resdl_1_out = self.resdl_1(input_out) # 128
        resdl_2_out = self.resdl_2(resdl_1_out) # 256
        #resdl_3_out = self.resdl_3(resdl_2_out) # 512
        
        middle_out = self.middle(resdl_2_out) # 512
        
        expnl_1_out = self.expnl_1(middle_out)
        expnl_1_cv_out = self.expnl_1_cv(torch.cat((expnl_1_out, resdl_2_out), dim=1)) # 512
        expnl_2_out = self.expnl_2(expnl_1_cv_out) # 256
        expnl_2_cv_out = self.expnl_2_cv(torch.cat((expnl_2_out, resdl_1_out), dim=1))
        expnl_3_out = self.expnl_3(expnl_2_cv_out)
        expnl_3_cv_out = self.expnl_3_cv(torch.cat((expnl_3_out, contl_1_out), dim=1))
        # expnl_4_out = self.expnl_4(expnl_3_cv_out)
        # expnl_4_cv_out = self.expnl_4_cv(torch.cat((expnl_4_out, input_out), dim=1))
        
        out = self.output(expnl_3_cv_out)
        return out