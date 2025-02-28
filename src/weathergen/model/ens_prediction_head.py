# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import code
import torch

from weathergen.model.norms import RMSNorm


class EnsPredictionHead(torch.nn.Module):

    #########################################
    def __init__(
        self,
        dim_embed,
        dim_out,
        ens_num_layers,
        ens_size,
        norm_type="LayerNorm",
        hidden_factor=2,
    ):
        """Constructor"""

        super(EnsPredictionHead, self).__init__()

        dim_internal = dim_embed * hidden_factor
        norm = torch.nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm
        enl = ens_num_layers

        self.pred_heads = torch.nn.ModuleList()
        for i in range(ens_size):

            self.pred_heads.append(torch.nn.ModuleList())

            # self.pred_heads[-1].append( norm( dim_embed))
            self.pred_heads[-1].append(
                torch.nn.Linear(dim_embed, dim_out if 1 == enl else dim_internal)
            )

            for i in range(ens_num_layers - 1):
                self.pred_heads[-1].append(torch.nn.GELU())
                self.pred_heads[-1].append(
                    torch.nn.Linear(
                        dim_internal, dim_out if enl - 2 == i else dim_internal
                    )
                )

    #########################################
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, toks):

        preds = []
        for pred_head in self.pred_heads:
            cpred = toks
            for block in pred_head:
                cpred = block(cpred)
            preds.append(cpred)
        preds = torch.stack(preds, 0)

        return preds
