import os
from code.box.box_wrapper import CenterSigmoidBoxTensor
from code.box.modules import BCEWithLogProbLoss, HighwayNetwork, LinearProjection, SimpleFeedForwardLayer
from code.config import Config
from code.models.context_encoder import ContextEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

config = Config()


class CBERTProto(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.context_encoder = ContextEncoder(args)

    def forward(self, batch, device):
        query_ids = batch['query_ids']
        query_spans = batch['query_spans']
        support_ids = batch['support_ids']
        support_spans = batch['support_spans']
        target_ids = batch['target_ids1']

        prototype_reps = []
        for s_ids, s_spans in zip(support_ids, support_spans):
            s_ids = torch.tensor(s_ids).to(device)
            prototype_rep = self.context_encoder(s_ids, s_spans).mean(dim=0)
            prototype_reps.append(prototype_rep)
        prototype_reps = torch.stack(prototype_reps)
        query_reps = self.context_encoder(query_ids, query_spans)
        sims = query_reps @ prototype_reps.t()

        preds = torch.argmax(sims, dim=1)
        loss = F.cross_entropy(sims, target_ids)
        return loss, preds == target_ids

    def forward_wsd(self, batch, device):
        example = batch['example']
        query_ids = batch['query_ids']
        query_spans = batch['query_spans']
        support_ids = batch['support_ids']
        support_spans = batch['support_spans']
        support_sensekeys = batch['support_sensekeys']

        with torch.no_grad():
            prototype_reps = []
            for s_ids, s_spans in zip(support_ids, support_spans):
                s_ids = torch.tensor(s_ids).to(device)
                prototype_rep = self.context_encoder(s_ids, s_spans).mean(dim=0)
                prototype_reps.append(prototype_rep)
            prototype_reps = torch.stack(prototype_reps)

            query_rep = self.context_encoder(query_ids, query_spans)

            sims = query_rep @ prototype_reps.t()
            pred_support_idx = torch.argmax(sims).item()
            pred_sensekey = support_sensekeys[pred_support_idx]

            return pred_sensekey

    def forward_nsc(self, batch, device):
        example = batch['example']
        query_ids = batch['query_ids']
        query_spans = batch['query_spans']
        support_ids = batch['support_ids']
        support_spans = batch['support_spans']
        support_sensekeys = batch['support_sensekeys']

        with torch.no_grad():
            prototype_reps = []
            for s_ids, s_spans in zip(support_ids, support_spans):
                s_ids = torch.tensor(s_ids).to(device)
                prototype_rep = self.context_encoder(s_ids, s_spans).mean(dim=0)
                prototype_reps.append(prototype_rep)
            prototype_reps = torch.stack(prototype_reps)

            query_rep = self.context_encoder(query_ids, query_spans)

            sims = query_rep @ prototype_reps.t()
            sims = sims / config.BERT_DIM

            return sims, support_sensekeys

    def forward_rep(self, batch, device):
        examples = batch['examples']
        ids = batch['ids']
        spans = batch['spans']
        with torch.no_grad():
            ids = torch.tensor(ids).to(device)
            prototype_rep = self.context_encoder(ids, spans).mean(dim=0)
            return prototype_rep


class CBERTProtoBox(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.box = CenterSigmoidBoxTensor
        self.context_encoder = ContextEncoder(args)
        self.euler_gamma = 0.57721566490153286060
        self.gumbel_beta = 0.00036026463511690845
        self.inv_softplus_temp = 1.2471085395024732
        # self.linear = LinearProjection(config.BERT_DIM, args.box_dim * 2)
        self.linear = HighwayNetwork(config.BERT_DIM, args.box_dim * 2, 2)
        self.loss_func = BCEWithLogProbLoss()
        self.softplus_scale = 1.0

    def forward(self, batch, device):
        query_ids = batch['query_ids']
        query_spans = batch['query_spans']
        support_ids = batch['support_ids']
        support_spans = batch['support_spans']
        target_ids1 = batch['target_ids1']
        target_ids2 = batch['target_ids2']

        with torch.autocast(device_type='cuda'):
            prototype_reps = []
            for s_ids, s_spans in zip(support_ids, support_spans):
                s_ids = torch.tensor(s_ids).to(device)
                s_reps = self.context_encoder(s_ids, s_spans)
                prototype_rep = self.linear(s_reps).mean(dim=0)
                prototype_reps.append(prototype_rep)
            prototype_reps = torch.stack(prototype_reps)
            prototype_reps = self.box.from_split(prototype_reps)

            query_reps = self.context_encoder(query_ids, query_spans)
            query_reps = self.linear(query_reps)
            query_reps = self.box.from_split(query_reps)

            log_probs1 = self.calc_log_prob(prototype_reps, query_reps)
            log_probs2 = self.calc_log_prob(query_reps, prototype_reps).t()

            loss1 = self.loss_func(log_probs1, target_ids1)
            loss2 = self.loss_func(log_probs2, target_ids2)
            r = 0.5
            loss = r * loss1 + (1 - r) * loss2

            return loss

    def forward_wsd(self, batch, device):
        example = batch['example']
        query_ids = batch['query_ids']
        query_spans = batch['query_spans']
        support_ids = batch['support_ids']
        support_spans = batch['support_spans']
        support_sensekeys = batch['support_sensekeys']

        with torch.no_grad():
            prototype_reps = []
            for s_ids, s_spans in zip(support_ids, support_spans):
                s_ids = torch.tensor(s_ids).to(device)
                s_reps = self.context_encoder(s_ids, s_spans)
                prototype_rep = self.linear(s_reps).mean(dim=0)
                prototype_reps.append(prototype_rep)
            prototype_reps = torch.stack(prototype_reps)
            prototype_reps = self.box.from_split(prototype_reps)

            query_rep = self.context_encoder(query_ids, query_spans)
            query_rep = self.linear(query_rep)
            query_rep = self.box.from_split(query_rep)

            log_probs = self.calc_log_prob(prototype_reps, query_rep)
            probs1 = torch.exp(log_probs)

            log_probs = self.calc_log_prob(query_rep, prototype_reps).t()
            probs2 = torch.exp(log_probs)

            sims = (probs1 * probs2 * 2) / (probs1 + probs2)
            pred_support_idx = torch.argmax(sims).item()
            pred_sensekey = support_sensekeys[pred_support_idx]

            return pred_sensekey

    def forward_nsc(self, batch, device):
        example = batch['example']
        query_ids = batch['query_ids']
        query_spans = batch['query_spans']
        support_ids = batch['support_ids']
        support_spans = batch['support_spans']
        support_sensekeys = batch['support_sensekeys']

        with torch.no_grad():
            prototype_reps = []
            for s_ids, s_spans in zip(support_ids, support_spans):
                s_ids = torch.tensor(s_ids).to(device)
                s_reps = self.context_encoder(s_ids, s_spans)
                prototype_rep = self.linear(s_reps).mean(dim=0)
                prototype_reps.append(prototype_rep)
            prototype_reps = torch.stack(prototype_reps)
            prototype_reps = self.box.from_split(prototype_reps)

            query_rep = self.context_encoder(query_ids, query_spans)
            query_rep = self.linear(query_rep)
            query_rep = self.box.from_split(query_rep)

            log_probs = self.calc_log_prob(prototype_reps, query_rep)
            probs1 = torch.exp(log_probs)

            log_probs = self.calc_log_prob(query_rep, prototype_reps).t()
            probs2 = torch.exp(log_probs)

            sims = (probs1 * probs2 * 2) / (probs1 + probs2)

            return sims, support_sensekeys

    def forward_rep(self, batch, device):
        examples = batch['examples']
        ids = batch['ids']
        spans = batch['spans']
        with torch.no_grad():
            ids = torch.tensor(ids).to(device)
            reps = self.context_encoder(ids, spans)
            prototype_rep = self.linear(reps).mean(dim=0)
            return prototype_rep

    def log_soft_volume(
            self,
            z: torch.Tensor,
            Z: torch.Tensor,
            temp: float = 1.,
            scale: float = 1.,
            gumbel_beta: float = 0.) -> torch.Tensor:
        eps = torch.finfo(z.dtype).tiny  # type: ignore
        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale
        if gumbel_beta <= 0.:
            return (torch.sum(torch.log(F.softplus(Z - z, beta=temp).clamp_min(eps)), dim=-1) + torch.log(s))
        else:
            return (torch.sum(
                torch.log(F.softplus(Z - z - 2 * self.euler_gamma * gumbel_beta, beta=temp).clamp_min(eps)), dim=-1) + torch.log(s))

    def calc_log_prob(self, support_boxes, query_boxes):
        min_point = torch.stack(
            [query_boxes.z.unsqueeze(1).expand(-1, support_boxes.z.size(0), -1),
             support_boxes.z.unsqueeze(0).expand(query_boxes.z.size(0), -1, -1)]
        )
        min_point = torch.max(
            self.gumbel_beta * torch.logsumexp(min_point / self.gumbel_beta, 0),
            torch.max(min_point, 0)[0]
        )
        max_point = torch.stack(
            [query_boxes.Z.unsqueeze(1).expand(-1, support_boxes.Z.size(0), -1),
             support_boxes.Z.unsqueeze(0).expand(query_boxes.Z.size(0), -1, -1)]
        )
        max_point = torch.min(
            -self.gumbel_beta * torch.logsumexp(-max_point / self.gumbel_beta, 0),
            torch.min(max_point, 0)[0]
        )
        vol1 = self.log_soft_volume(min_point,
                                    max_point,
                                    temp=self.inv_softplus_temp,
                                    scale=self.softplus_scale,
                                    gumbel_beta=self.gumbel_beta)
        vol2 = self.log_soft_volume(query_boxes.z,
                                    query_boxes.Z,
                                    temp=self.inv_softplus_temp,
                                    scale=self.softplus_scale,
                                    gumbel_beta=self.gumbel_beta)
        log_probs = vol1 - vol2.unsqueeze(-1)
        return log_probs
