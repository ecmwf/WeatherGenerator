# ruff: noqa: N801, N806
# pylint: disable=W0201

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def lossfunc(t, s, temp):
    return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)


class iBOTPatchTargetProcessing(nn.Module):
    """
    Code taken and adapted from the official DINOv2 implementation
    https://github.com/facebookresearch/dinov2/tree/main

    Needs to be nn.Module because of the registered_buffer, it means we should have a forward
    function, previously was the softmax computation, maybe we can make it the
    softmax_center_teacher, etc
    """

    def __init__(
        self,
        patch_out_dim,
        student_temp=0.1,
        teacher_temp=0.1,
        center_momentum=0.9,
        teacher_style="softmax_center",
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None
        self.teacher_style = teacher_style
        # self.center cannot be initialised with None then the code breaks hence the disabled pylint
        assert teacher_style in ["softmax_center", "sinkhorn_knopp"], f"{teacher_style} is unknown"

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        #
        # WARNING:
        #   as self.center is a float32, everything gets casted to float32 afterwards
        #
        # teacher_patch_tokens = teacher_patch_tokens.float()
        # return F.softmax((teacher_patch_tokens.sub_(self.center.to(
        #        teacher_patch_tokens.dtype))).mul_(1 / teacher_temp), dim=-1)

        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

        # this is experimental, keep everything in float16 and let's see what happens:
        # return F.softmax((teacher_patch_tokens.sub_(self.center)) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        batch_size, C, K = teacher_output.shape  # C = num of tokens
        teacher_output = teacher_output.view(batch_size * C, K)
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        # B = n_masked_patches_tensor
        dist.all_reduce(B)
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for _it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t().view(batch_size, C, K)

    def forward(self, teacher_output):
        if self.teacher_style == "softmax_center":
            processed_teacher_output = self.softmax_center_teacher(
                teacher_output, self.teacher_temp
            )
            self.update_center(teacher_output)
            return processed_teacher_output
        elif self.teacher_style == "sinkhorn_knopp":
            return self.sinkhorn_knopp_teacher(teacher_output, self.teacher_temp)
        else:
            # this code should never be reached, see assert in __init__
            return teacher_output

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True


class DINOTargetProcessing(nn.Module):
    """
    Code taken and adapted from the official DINOv2 implementation
    https://github.com/facebookresearch/dinov2/tree/main

    Needs to be nn.Module because of the registered_buffer, it means we should have a forward
    function, previously was the softmax computation, maybe we can make it the
    softmax_center_teacher, etc
    """

    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
        teacher_temp=0.1,
        teacher_style="softmax_center",
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None
        self.teacher_style = teacher_style
        # self.center cannot be initialised with None then the code breaks hence the disabled pylint
        assert teacher_style in ["softmax_center", "sinkhorn_knopp"], f"{teacher_style} is unknown"

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        B, C, K = teacher_output.shape
        batch_size, C, K = teacher_output.shape  # C = num of tokens
        teacher_output = teacher_output.view(batch_size * C, K)
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for _it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t().view(batch_size, C, K)

    def forward(self, teacher_output):
        if self.teacher_style == "softmax_center":
            processed_teacher_output = self.softmax_center_teacher(
                teacher_output, self.teacher_temp
            )
            self.update_center(teacher_output)
            return processed_teacher_output
        elif self.teacher_style == "sinkhorn_knopp":
            return self.sinkhorn_knopp_teacher(teacher_output, self.teacher_temp)
        else:
            # this code should never be reached, see assert in __init__
            return teacher_output

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True


class JEPATargetProcessing(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return z
