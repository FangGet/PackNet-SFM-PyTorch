# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class UnPackDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(UnPackDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128,128])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i+1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("unpack", i)] = UnPackBlock(num_ch_in, num_ch_out)

            # upconv_1
            if i > 0:
                num_ch_in = num_ch_out + self.num_ch_enc[i-1]
            else:
                num_ch_in = num_ch_out + 0

            if i in self.scales:
                self.convs[("dispconv", i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)
            if i < 3:
                num_ch_in += 1
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, idx):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("unpack", i)](x) #12 #14 #17 #20 #23
            if i > 0:
                x = [input_features[i], x]
            else:
                x = [x]
            if i < 3:
                x += [upsample(self.outputs[("disp", idx, i+1)])]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 0)](x) #13 #15 #18 #21 #24
            if i in self.scales:
                self.outputs[("disp", idx, i)] = self.sigmoid(self.convs[("dispconv", i)](x)) #16 #19 #21 #24

        return self.outputs
