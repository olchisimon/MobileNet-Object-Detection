import torch
import torchvision


class MmpNet(torch.nn.Module):
    def __init__(self, num_widths: int, num_aspect_ratios: int):
        super(MmpNet, self).__init__()
        self.num_widths = num_widths
        self.num_aspect_ratios = num_aspect_ratios

        #mobilenet V3
        mobile_v3 = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone = mobile_v3.features
        
        num_anchors = num_widths * num_aspect_ratios
        dropout_rate = 0.25
        backbone_out_channels = 960
        # final layer
        self.anchor_layers = torch.nn.Sequential(
            torch.nn.Conv2d(backbone_out_channels, backbone_out_channels, kernel_size=3, padding=1, groups=backbone_out_channels),  # depthwise
            torch.nn.Conv2d(backbone_out_channels, 512, kernel_size=1),  # pointwise
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout2d(p=dropout_rate),
            torch.nn.Conv2d(512, num_anchors * 2, kernel_size=1)
        )

        self.bbr_layers = torch.nn.Sequential(
            torch.nn.Conv2d(backbone_out_channels, backbone_out_channels, kernel_size=3, padding=1, groups=backbone_out_channels),  # depthwise
            torch.nn.Conv2d(backbone_out_channels, 512, kernel_size=1),  # pointwise
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout2d(p=dropout_rate),
            torch.nn.Conv2d(512, num_anchors * 4, kernel_size=1)
        )



    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.backbone(x)
        output = self.anchor_layers(x)
        bbr_output = self.bbr_layers(x)

        batch_size = output.shape[0]
        height = output.shape[2]
        width = output.shape[3]

        # reformat output anchor
        output = output.permute(0,2,3,1)
        output = output.reshape(batch_size, height, width, self.num_widths, self.num_aspect_ratios, 2)
        output = output.permute(0,5,3,4,1,2)

        # reformat output bbr
        bbr_output = bbr_output.permute(0,2,3,1)
        bbr_output = bbr_output.reshape(batch_size, height, width, self.num_widths, self.num_aspect_ratios, 4)
        bbr_output = bbr_output.permute(0,3,4,1,2,5)


        # output shape (B, 2, num_widths, num_ratios, H, W)
        # output bbr shape (B, num_widths, num_ratios, H, W, 4) <- same as label_grid, anchor grid
        return output, bbr_output
