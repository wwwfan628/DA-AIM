#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch.nn as nn
from torch.autograd import Function
from torch.nn.init import *
from . import TRNmodule


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None


class Video_Head(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(self, dropout_i=0.5, dropout_v=0.5, add_fc=1, share_params='Y', feat_shared_dim=512,
                 adv=['Y', 'Y', 'Y', 'Y'], use_attn='TransAttn', use_attn_frame='TransAttn'):
        super(Video_Head, self).__init__()
        self.dropout_rate_i = dropout_i
        self.dropout_rate_v = dropout_v
        self.add_fc = add_fc
        self.share_params = share_params
        self.feat_shared_dim = feat_shared_dim
        self.feature_dim = 2048
        self.feat_aggregated_dim = 256 * 14 * 14
        self.adv = adv
        self.use_attn_frame = use_attn_frame
        self.use_attn = use_attn
        self.num_segments = 8
        self.frame_aggregation = 'trn-m'
        self._construct_video_head()

    def _construct_video_head(self):
        std = 0.001
        self.relu = nn.ReLU(inplace=True)
        self.dropout_i = nn.Dropout(p=self.dropout_rate_i)
        self.dropout_v = nn.Dropout(p=self.dropout_rate_v)
        self.softmax = nn.Softmax(dim=1)
        ################################################################################
        # feature extraction
        feat_shared_dim = self.feat_shared_dim
        feat_frame_dim = feat_shared_dim

        feat_aggregated_dim = self.feat_aggregated_dim
        feat_video_dim = feat_shared_dim

        self.fc_feature_shared_source = nn.Linear(self.feature_dim, feat_shared_dim)
        normal_(self.fc_feature_shared_source.weight, 0, std)
        constant_(self.fc_feature_shared_source.bias, 0)

        if self.add_fc > 1:
            self.fc_feature_shared_2_source = nn.Linear(feat_shared_dim, feat_shared_dim)
            normal_(self.fc_feature_shared_2_source.weight, 0, std)
            constant_(self.fc_feature_shared_2_source.bias, 0)

        if self.add_fc > 2:
            self.fc_feature_shared_3_source = nn.Linear(feat_shared_dim, feat_shared_dim)
            normal_(self.fc_feature_shared_3_source.weight, 0, std)
            constant_(self.fc_feature_shared_3_source.bias, 0)

        if self.share_params == 'N':
            self.fc_feature_shared_target = nn.Linear(self.feature_dim, feat_shared_dim)
            normal_(self.fc_feature_shared_target.weight, 0, std)
            constant_(self.fc_feature_shared_target.bias, 0)

            if self.add_fc > 1:
                self.fc_feature_shared_2_target = nn.Linear(feat_shared_dim, feat_shared_dim)
                normal_(self.fc_feature_shared_2_target.weight, 0, std)
                constant_(self.fc_feature_shared_2_target.bias, 0)
            if self.add_fc > 2:
                self.fc_feature_shared_3_target = nn.Linear(feat_shared_dim, feat_shared_dim)
                normal_(self.fc_feature_shared_3_target.weight, 0, std)
                constant_(self.fc_feature_shared_3_target.bias, 0)

        ###############################################################################
        # relation + video
        if self.frame_aggregation == 'trn-m':  # 4. TRN (ECCV 2018) ==> fix segment # for both train/val
            self.num_bottleneck = 256
            self.TRN = TRNmodule.RelationModuleMultiScale(feat_shared_dim, self.num_bottleneck, self.num_segments)

        if 'trn' in self.frame_aggregation:
            feat_aggregated_dim_att = self.num_bottleneck

        feat_video_dim_att = feat_aggregated_dim_att
        self.relation_domain_classifier_all = nn.ModuleList()
        for i in range(self.num_segments - 1):
            relation_domain_classifier = nn.Sequential(
                nn.Linear(feat_aggregated_dim_att, feat_video_dim_att),
                nn.ReLU(),
                nn.Linear(feat_video_dim_att, 2)
            )
            self.relation_domain_classifier_all += [relation_domain_classifier]

        self.fc_feature_domain_video = nn.Linear(feat_aggregated_dim_att, feat_video_dim_att)
        normal_(self.fc_feature_domain_video.weight, 0, std)
        constant_(self.fc_feature_domain_video.bias, 0)

        self.fc_classifier_domain_video = nn.Linear(feat_video_dim_att, 2)
        normal_(self.fc_classifier_domain_video.weight, 0, std)
        constant_(self.fc_classifier_domain_video.bias, 0)

        ################################################################################
        # slow
        self.fc_feature_domain_slow = nn.Linear(feat_shared_dim, feat_frame_dim)
        normal_(self.fc_feature_domain_slow.weight, 0, std)
        constant_(self.fc_feature_domain_slow.bias, 0)

        self.fc_classifier_domain_slow = nn.Linear(feat_frame_dim, 2)
        normal_(self.fc_classifier_domain_slow.weight, 0, std)
        constant_(self.fc_classifier_domain_slow.bias, 0)

        #################################################################################
        # fast
        self.fc_feature_domain_fast = nn.Linear(feat_aggregated_dim, feat_video_dim)
        normal_(self.fc_feature_domain_fast.weight, 0, std)
        constant_(self.fc_feature_domain_fast.bias, 0)

        self.fc_classifier_domain_fast = nn.Linear(feat_video_dim, 2)
        normal_(self.fc_classifier_domain_fast.weight, 0, std)
        constant_(self.fc_classifier_domain_fast.bias, 0)

    def transform_data_slow(self, inputs):
        batch_size = inputs.size()[0]
        num_frames = inputs.size()[2]
        inputs = inputs.permute(0, 2, 1, 3, 4)
        temporal_pool = nn.AdaptiveAvgPool3d((2048, 1, 1))
        out_put = temporal_pool(inputs)
        out_put = out_put.view(batch_size, num_frames, -1)
        return out_put

    def transform_data_fast(self, inputs):
        batch_size = inputs.size()[0]
        num_channel = inputs.size()[1]
        temporal_pool = nn.AdaptiveAvgPool3d((1, 14, 14))
        out_put = temporal_pool(inputs)
        out_put = out_put.view(batch_size, num_channel, -1)
        return out_put

    ################################################################################
    # slow
    def domain_classifier_frame_slow(self, feat, beta):
        feat_fc_domain_frame = GradReverse.apply(feat, beta[0])
        feat_fc_domain_frame = self.fc_feature_domain_slow(feat_fc_domain_frame)
        feat_fc_domain_frame = self.relu(feat_fc_domain_frame)
        pred_fc_domain_frame = self.fc_classifier_domain_slow(feat_fc_domain_frame)
        pred_fc_domain_frame = self.softmax(pred_fc_domain_frame)
        return pred_fc_domain_frame

    ################################################################################
    # fast
    def domain_classifier_fast(self, feat_video, beta):
        feat_fc_domain_video = GradReverse.apply(feat_video, beta[1])
        feat_fc_domain_video = self.fc_feature_domain_fast(feat_fc_domain_video)
        feat_fc_domain_video = self.relu(feat_fc_domain_video)
        pred_fc_domain_video = self.fc_classifier_domain_fast(feat_fc_domain_video)
        pred_fc_domain_video = self.softmax(pred_fc_domain_video)
        return pred_fc_domain_video

    #################################################################################
    # attentive entropy
    def get_trans_attn(self, pred_domain):
        softmax = nn.Softmax(dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
        weights = 1 - entropy
        return weights

    def get_attn_feat_frame(self, feat_fc, pred_domain):
        weights_attn = self.get_trans_attn(pred_domain)
        weights_attn = weights_attn.view(-1, 1).repeat(1, feat_fc.size()[-1])  # reshape & repeat weights (e.g. 16 x 512)
        feat_fc_attn = (weights_attn + 1) * feat_fc
        return feat_fc_attn

    def get_attn_feat_relation(self, feat_fc, pred_domain, num_segments):
        weights_attn = self.get_trans_attn(pred_domain)
        weights_attn = weights_attn.view(-1, num_segments - 1, 1).repeat(1, 1, feat_fc.size()[-1])  # reshape & repeat weights (e.g. 16 x 4 x 256)
        feat_fc_attn = (weights_attn + 1) * feat_fc
        return feat_fc_attn

    #################################################################################
    # relation
    def domain_classifier_relation(self, feat_relation, beta):
        # 128x4x256 --> (128x4)x2
        pred_fc_domain_relation_video = None
        for i in range(len(self.relation_domain_classifier_all)):
            feat_relation_single = feat_relation[:, i, :].squeeze(1)  # 128x1x256 --> 128x256
            feat_fc_domain_relation_single = GradReverse.apply(feat_relation_single, beta[2])  # the same beta for all relations (for now)
            pred_fc_domain_relation_single = self.relation_domain_classifier_all[i](feat_fc_domain_relation_single)
            # pred_fc_domain_relation_single = [batch_size, 2]
            if pred_fc_domain_relation_video is None:
                pred_fc_domain_relation_video = pred_fc_domain_relation_single.view(-1, 1, 2)
            else:
                pred_fc_domain_relation_video = torch.cat(
                    (pred_fc_domain_relation_video, pred_fc_domain_relation_single.view(-1, 1, 2)), 1)
        pred_fc_domain_relation_video = pred_fc_domain_relation_video.view(-1, 2)
        return pred_fc_domain_relation_video

    #################################################################################
    # video
    def domain_classifier_video(self, feat_video, beta):
        feat_fc_domain_video = GradReverse.apply(feat_video, beta[3])
        feat_fc_domain_video = self.fc_feature_domain_video(feat_fc_domain_video)
        feat_fc_domain_video = self.relu(feat_fc_domain_video)
        pred_fc_domain_video = self.fc_classifier_domain_video(feat_fc_domain_video)
        return pred_fc_domain_video

    def forward(self, input_source, input_target, beta):
        if input_target[0].size()[0] == 0:
            batch_size = int(input_source[0].size()[0] / 2)
            input_target[0] = input_source[0][batch_size:, :, :, :, :]
            input_target[1] = input_source[1][batch_size:, :, :, :, :]
            input_source[0] = input_source[0][0:batch_size, :, :, :, :]
            input_source[1] = input_source[1][0:batch_size, :, :, :, :]

        pred_domain_all_source = []
        pred_domain_all_target = []

        input_source_slow = self.transform_data_slow(input_source[0])  # (batch_size, 8, 2048)
        input_target_slow = self.transform_data_slow(input_target[0])  # (batch_size, 8, 2048)

        input_source_fast = self.transform_data_fast(input_source[1])  # (batch_size, 256, 14x14)
        input_target_fast = self.transform_data_fast(input_target[1])  # (batch_size, 256, 14x14)

        feat_base_source_slow = input_source_slow.view(-1, input_source_slow.size()[
            -1])  # e.g. batch_size x 8 x 2048 --> (batch_size x 8) x 2048
        feat_base_target_slow = input_target_slow.view(-1, input_target_slow.size()[
            -1])  # e.g. batch_size x 8 x 2048 --> (batch_size x 8) x 2048

        if self.add_fc < 1:
            raise ValueError('not enough fc layer')

        # feature_fc_source = [batch * # of frames, dimension_output_fc = 512]
        feat_fc_source_slow = self.fc_feature_shared_source(feat_base_source_slow)
        feat_fc_target_slow = self.fc_feature_shared_target(
            feat_base_target_slow) if self.share_params == 'N' else self.fc_feature_shared_source(feat_base_target_slow)

        feat_fc_source_slow = self.relu(feat_fc_source_slow)
        feat_fc_target_slow = self.relu(feat_fc_target_slow)
        feat_fc_source_slow = self.dropout_i(feat_fc_source_slow)
        feat_fc_target_slow = self.dropout_i(feat_fc_target_slow)

        if self.add_fc > 1:
            # feat_fc_source = [batch *  # of frames, dimension_output_fc = 512]
            feat_fc_source_slow = self.fc_feature_shared_2_source(feat_fc_source_slow)
            feat_fc_target_slow = self.fc_feature_shared_2_target(
                feat_fc_target_slow) if self.share_params == 'N' else self.fc_feature_shared_2_source(
                feat_fc_target_slow)

            feat_fc_source_slow = self.relu(feat_fc_source_slow)
            feat_fc_target_slow = self.relu(feat_fc_target_slow)
            feat_fc_source_slow = self.dropout_i(feat_fc_source_slow)
            feat_fc_target_slow = self.dropout_i(feat_fc_target_slow)

        if self.add_fc > 2:
            # feat_fc_source = [batch *  # of frames, dimension_output_fc = 512]
            feat_fc_source_slow = self.fc_feature_shared_3_source(feat_fc_source_slow)
            feat_fc_target_slow = self.fc_feature_shared_3_target(
                feat_fc_target_slow) if self.share_params == 'N' else self.fc_feature_shared_3_source(
                feat_fc_target_slow)

            feat_fc_source_slow = self.relu(feat_fc_source_slow)
            feat_fc_target_slow = self.relu(feat_fc_target_slow)
            feat_fc_source_slow = self.dropout_i(feat_fc_source_slow)
            feat_fc_target_slow = self.dropout_i(feat_fc_target_slow)

        ###########################################################
        # slow
        if self.adv[0] == 'Y':
            pred_fc_domain_frame_source_slow = self.domain_classifier_frame_slow(feat_fc_source_slow, beta)
            pred_fc_domain_frame_target_slow = self.domain_classifier_frame_slow(feat_fc_target_slow, beta)

            pred_domain_all_source.append(self.softmax(pred_fc_domain_frame_source_slow))
            pred_domain_all_target.append(self.softmax(pred_fc_domain_frame_target_slow))
        else:
            pred_domain_all_source.append(0)
            pred_domain_all_target.append(0)

        ############################################################
        # relation module
        if self.adv[1] == 'Y':
            if self.use_attn_frame != 'none':  # attend the frame-level features only
                # input:pred_fc_domain_frame_source = [batch *  # of frames, 2]
                # input:feat_fc_source = [batch *  # of frames, dimension_output_fc]
                if self.adv[0] == 'N':
                    pred_fc_domain_frame_source_slow = self.domain_classifier_frame_slow(feat_fc_source_slow, beta)
                    pred_fc_domain_frame_target_slow = self.domain_classifier_frame_slow(feat_fc_target_slow, beta)

                feat_fc_source_slow = self.get_attn_feat_frame(feat_fc_source_slow, pred_fc_domain_frame_source_slow)
                feat_fc_target_slow = self.get_attn_feat_frame(feat_fc_target_slow, pred_fc_domain_frame_target_slow)
            #######################################
            # relation module
            feat_fc_video_source_slow = feat_fc_source_slow.view((-1, self.num_segments) + feat_fc_source_slow.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)
            feat_fc_video_target_slow = feat_fc_target_slow.view((-1, self.num_segments) + feat_fc_target_slow.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)

            feat_fc_video_relation_source = self.TRN(feat_fc_video_source_slow)  # 128x5x512 --> 128x5x256 (256-dim. relation feature vectors x 5)
            feat_fc_video_relation_target = self.TRN(feat_fc_video_target_slow)
            # feat_fc_video_relation_source = [batch,  # of frames - 1, dimension_output]

            # adversarial branch
            pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source, beta)
            pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target, beta)
            # pred_fc_domain_video_relation_source = [batch * # of frames - 1, 2]

            pred_domain_all_source.append(self.softmax(pred_fc_domain_video_relation_source))
            pred_domain_all_target.append(self.softmax(pred_fc_domain_video_relation_target))
        else:
            pred_domain_all_source.append(0)
            pred_domain_all_target.append(0)

        ############################################################
        # video module with attentive entropy
        if self.adv[2] == 'Y':
            if self.adv[1] == 'N':
                if self.use_attn_frame != 'none':  # attend the frame-level features only
                    # input:pred_fc_domain_frame_source = [batch *  # of frames, 2]
                    # input:feat_fc_source = [batch *  # of frames, dimension_output_fc]
                    if self.adv[0] == 'N':
                        pred_fc_domain_frame_source_slow = self.domain_classifier_frame_slow(feat_fc_source_slow, beta)
                        pred_fc_domain_frame_target_slow = self.domain_classifier_frame_slow(feat_fc_target_slow, beta)

                    feat_fc_source_slow = self.get_attn_feat_frame(feat_fc_source_slow, pred_fc_domain_frame_source_slow)
                    feat_fc_target_slow = self.get_attn_feat_frame(feat_fc_target_slow, pred_fc_domain_frame_target_slow)
                #######################################
                # relation module
                feat_fc_video_source_slow = feat_fc_source_slow.view((-1, self.num_segments) + feat_fc_source_slow.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)
                feat_fc_video_target_slow = feat_fc_target_slow.view((-1, self.num_segments) + feat_fc_target_slow.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)

                feat_fc_video_relation_source = self.TRN(feat_fc_video_source_slow)  # 128x5x512 --> 128x5x256 (256-dim. relation feature vectors x 5)
                feat_fc_video_relation_target = self.TRN(feat_fc_video_target_slow)
                # feat_fc_video_relation_source = [batch,  # of frames - 1, dimension_output]
            # transferable attention
            if self.use_attn != 'none':  # get the attention weighting
                # adversarial branch
                if self.adv[1] == 'N':
                    pred_fc_domain_video_relation_source = self.domain_classifier_relation(
                        feat_fc_video_relation_source, beta)
                    pred_fc_domain_video_relation_target = self.domain_classifier_relation(
                        feat_fc_video_relation_target, beta)
                # pred_fc_domain_video_relation_source = [batch * # of frames - 1, 2]
                feat_fc_video_relation_source = self.get_attn_feat_relation(feat_fc_video_relation_source,
                                                                            pred_fc_domain_video_relation_source,
                                                                            self.num_segments)
                feat_fc_video_relation_target = self.get_attn_feat_relation(feat_fc_video_relation_target,
                                                                            pred_fc_domain_video_relation_target,
                                                                            self.num_segments)
                # feat_fc_video_relation_target = [batch, # of frames - 1, dimension_output]
                # attn_relation_target = [batch, # of frames - 1]

            # sum up relation features (ignore 1-relation)
            feat_fc_video_source = torch.sum(feat_fc_video_relation_source, 1)
            feat_fc_video_target = torch.sum(feat_fc_video_relation_target, 1)
            # feat_fc_video_source = [batch_size, dimension_output]

            # === source layers (video-level) ===#
            feat_fc_video_source = self.dropout_v(feat_fc_video_source)
            feat_fc_video_target = self.dropout_v(feat_fc_video_target)

            pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, beta)
            pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, beta)
            # pred_fc_domain_video_source = [batch_size, 2]

            pred_domain_all_source.append(self.softmax(pred_fc_domain_video_source))
            pred_domain_all_target.append(self.softmax(pred_fc_domain_video_target))
        else:
            pred_domain_all_source.append(0)
            pred_domain_all_target.append(0)

        ############################################################
        # fast
        if self.adv[3] == 'Y':

            feat_base_source_fast = input_source_fast.view(input_source_fast.size()[0], -1)  # e.g. batch_size x 256 x (14 x 14) --> batch_size x (256 x 14 x 14)
            feat_base_target_fast = input_target_fast.view(input_target_fast.size()[0], -1)  # e.g. batch_size x 256 x (14 x 14) --> batch_size x (256 x 14 x 14)

            feat_fc_video_source_fast = self.dropout_v(feat_base_source_fast)
            feat_fc_video_target_fast = self.dropout_v(feat_base_target_fast)

            pred_fc_domain_video_source = self.domain_classifier_fast(feat_fc_video_source_fast, beta)
            pred_fc_domain_video_target = self.domain_classifier_fast(feat_fc_video_target_fast, beta)

            pred_domain_all_source.append(pred_fc_domain_video_source)
            pred_domain_all_target.append(pred_fc_domain_video_target)
        else:
            pred_domain_all_source.append(0)
            pred_domain_all_target.append(0)

        return pred_domain_all_source, pred_domain_all_target