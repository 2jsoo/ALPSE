## Import Library
from import_library import *

class ALPSE(nn.Module):
    def __init__(self, in_channels, out_channels, S = 9, B = 2, C = 1, init_weights=True, device=None, activation='leakyrelu'):
        super(ALPSE, self).__init__()

        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'silu': # = swish
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels//16, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm1d(num_features=out_channels//16),
            self.activation,
            nn.MaxPool1d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels//16, out_channels=out_channels//16, kernel_size=1, stride=1, padding=0), 
            nn.BatchNorm1d(num_features=out_channels//16),
            self.activation,
            nn.Conv1d(in_channels=out_channels//16, out_channels=out_channels//8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channels//8),
            self.activation,
            nn.MaxPool1d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels//8, out_channels=out_channels//8, kernel_size=1, stride=1, padding=0), 
            nn.BatchNorm1d(num_features=out_channels//8),
            self.activation,
            nn.Conv1d(in_channels=out_channels//8, out_channels=out_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channels//4),
            self.activation,
            nn.Conv1d(in_channels=out_channels//4, out_channels=out_channels//8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=out_channels//8),
            self.activation,
            nn.Conv1d(in_channels=out_channels//8, out_channels=out_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channels//4),
            self.activation,
            nn.Conv1d(in_channels=out_channels//4, out_channels=out_channels//8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=out_channels//8),
            self.activation,
            nn.Conv1d(in_channels=out_channels//8, out_channels=out_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channels//4),
            self.activation,
            nn.MaxPool1d(2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels//4, out_channels=out_channels//4, kernel_size=1, stride=1, padding=0), 
            nn.BatchNorm1d(num_features=out_channels//4),
            self.activation,
            nn.Conv1d(in_channels=out_channels//4, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channels//2),
            self.activation,
            nn.Conv1d(in_channels=out_channels//2, out_channels=out_channels//4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=out_channels//4),
            self.activation,
            nn.Conv1d(in_channels=out_channels//4, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channels//2),
            self.activation,
            nn.Conv1d(in_channels=out_channels//2, out_channels=out_channels//4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=out_channels//4),
            self.activation,
            nn.Conv1d(in_channels=out_channels//4, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channels//2),
            self.activation,
            nn.AdaptiveMaxPool1d(S), 
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm1d(num_features=out_channels),
            self.activation,
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm1d(num_features=out_channels),
            self.activation,
        )

        self.device=device
        self.S = S
        self.B = B
        self.C = C

        self.fpn1 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels//4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=out_channels//4),
            self.activation,
        )
        self.fpn2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels//2, out_channels=out_channels//4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(num_features=out_channels//4),
            self.activation,
        )

        self.fpn_conv3x3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels//4, out_channels=out_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channels//4),
            self.activation,
            nn.Conv1d(in_channels=out_channels//4, out_channels=out_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channels//4),
            self.activation,
        )

        self.bottomup = nn.Sequential(
            nn.Conv1d(in_channels=out_channels//4, out_channels=out_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channels//4),
            self.activation,
        )


        self.final_detect = nn.Sequential(
            nn.Conv1d(out_channels//4, out_channels//4, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm1d(out_channels//4),
            self.activation,
            nn.Conv1d(out_channels//4, out_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels//4),
            self.activation,
            nn.Conv1d(out_channels//4, (self.B*3 + self.C), kernel_size=1, stride=1, padding=0),
        )
        self.final_detect2 = nn.Sequential(
            nn.Conv1d((self.B*3 + self.C), (self.B*3 + self.C), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d((self.B*3 + self.C)),
            self.activation,
            nn.Conv1d((self.B*3 + self.C), (self.B*3 + self.C), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d((self.B*3 + self.C)),
            self.activation,
            nn.Conv1d((self.B*3 + self.C), (self.B*3 + self.C), kernel_size=1, stride=1, padding=0),
        )

        self.flatten = nn.Flatten()
        
        self.apply(self._init_weights)


        if device != None:
            self.layer1 = self.layer1.to(device)
            self.layer2 = self.layer2.to(device)
            self.layer3 = self.layer3.to(device)
            self.layer4 = self.layer4.to(device)
            self.layer5 = self.layer5.to(device)           

            self.fpn1 = self.fpn1.to(device)
            self.fpn2 = self.fpn2.to(device)
            self.fpn_conv3x3 = self.fpn_conv3x3.to(device)

            self.bottomup = self.bottomup.to(device)

            self.final_detect = self.final_detect.to(device)
            self.final_detect2 = self.final_detect2.to(device)
            self.flatten = self.flatten.to(device)

    def forward(self, x):
        #### Backbone ####
        # x = torch.Size([64, 1, 2688]) 
        out_layer1 = self.layer1(x) # torch.Size([64, 16, 671]) 
        
        out_layer2 = self.layer2(out_layer1) # torch.Size([64, 32, 335]) 

        out_layer3 = self.layer3(out_layer2) # torch.Size([64, 64, 167]) 

        out_layer4 = self.layer4(out_layer3) # torch.Size([64, 128, S])

        out_layer5 = self.layer5(out_layer4) # torch.Size([64, 256, S])


        #### Top-down pathway #### 
        out_layer5_fpn = self.fpn1(out_layer5) # (64, 64, S)
        out_layer4_fpn = self.fpn2(out_layer4) # (64, 64, S)
        out_layer45 = out_layer4_fpn + F.upsample(out_layer5_fpn, size=out_layer4.shape[2]) # (64, 64, S)

        out_layer345 = out_layer3 + F.upsample(out_layer45, size=out_layer3.shape[2]).to(self.device) # (64, 64, 167)

        out_layer5_fpn_alias = self.fpn_conv3x3(out_layer5_fpn)
        out_layer4_fpn_alias = self.fpn_conv3x3(out_layer4_fpn)
        out_layer345_alias = self.fpn_conv3x3(out_layer345) # (64, 64, 167)

        #### Bottom-up path augmentation ####
        out_layer345_bottomup = out_layer4_fpn_alias + F.adaptive_max_pool1d(self.bottomup(out_layer345_alias), output_size=out_layer4.shape[2]).to(self.device) # (64, 64, 167) -> (64, 64, S)
        out_layer345_bottomup_alias = self.bottomup(out_layer345_bottomup)
        out_layer45_bottomup = out_layer5_fpn_alias + F.adaptive_max_pool1d(out_layer345_bottomup_alias, output_size=out_layer5.shape[2]).to(self.device) # (64, 64, S) -> (64, 64, S)
        out_layer45_bottomup_alias = self.bottomup(out_layer45_bottomup)


        output_1 = self.final_detect(out_layer345_alias) # (64, 7, 167)
        output_2 = self.final_detect(out_layer345_bottomup_alias) # (64, 7, S)
        output_3 = self.final_detect(out_layer45_bottomup_alias) # (64, 7, S)

        output = F.adaptive_max_pool1d(output_1, self.S).to(self.device) + output_2 + output_3 # (64, 7, S)
        output = self.final_detect2(output) # (64, 7, S)
        output = output.permute(0, 2, 1) # (64, S, 7)

        ## class, confidence -> sigmoid
        ## coordinate -> linear
        output[..., :self.C] = F.sigmoid(output[..., :self.C]).to(self.device)
        output[..., self.C:self.C+1] = F.sigmoid(output[..., self.C:self.C+1]).to(self.device) 
        output[..., self.C+3:self.C+1+3] = F.sigmoid(output[..., self.C+3:self.C+1+3]).to(self.device)      
        return output


    def _init_weights(self, m):
        if isinstance(self.activation, nn.LeakyReLU):
            nonlinearity_ = 'leaky_relu'
        elif isinstance(self.activation, nn.ReLU):
            nonlinearity_ = 'relu'
        elif isinstance(self.activation, nn.Mish):
            nonlinearity_ = 'relu'
        elif isinstance(self.activation, nn.SELU):
            nonlinearity_ = 'selu'
        elif isinstance(self.activation, nn.SiLU):
            nonlinearity_ = 'relu'
        elif isinstance(self.activation, nn.GELU):
            nonlinearity_ = 'relu'

        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity_) 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity=nonlinearity_)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)