from dataclasses import dataclass

import torch
from pytorch3d.ops import knn_points
from torch import nn

from utils.cloth_and_material import FaceNormals
from utils.common import gather


@dataclass
class Config:
    weight_start: float = 1e+3          # minimal weight
    weight_max: float = 1e+5            # maximal weight
    start_rampup_iteration: int = 50000 # the iteration to start ramping up the weight, for iterations < this value, the weight is weight_start
    n_rampup_iterations: int = 100000   # the number of iterations to ramp up the weight, for iterations > start_rampup_iteration + n_rampup_iterations, the weight is weight_max
    eps: float = 1e-3                   # penetration threshold


def create(mcfg):
    return Criterion(weight_start=mcfg.weight_start, weight_max=mcfg.weight_max,
                     start_rampup_iteration=mcfg.start_rampup_iteration, n_rampup_iterations=mcfg.n_rampup_iterations,
                     eps=mcfg.eps)


class Criterion(nn.Module):
    def __init__(self, weight_start, weight_max, start_rampup_iteration, n_rampup_iterations, eps=1e-3):
        """
        Collision penalty
        See section 1.4 of the HOOD Supplementary Material for details
        https://dolorousrtur.github.io/hood/static/suppmat.pdf

        We ramp up the weight of this penalty from weight_start to weight_max
        """


        super().__init__()
        self.weight_start = weight_start
        self.weight_max = weight_max
        self.start_rampup_iteration = start_rampup_iteration
        self.n_rampup_iterations = n_rampup_iterations
        self.eps = eps
        self.f_normals_f = FaceNormals()
        self.name = 'collision_penalty'

    def get_weight(self, iter):
        iter = iter - self.start_rampup_iteration
        iter = max(iter, 0)
        progress = iter / self.n_rampup_iterations
        progress = min(progress, 1.)
        weight = self.weight_start + (self.weight_max - self.weight_start) * progress
        return weight

    def _calc_external_loss(self, example):
        """L_e: garment-body external penetration loss."""
        obstacle_pos = example['obstacle'].pos
        obstacle_prev_pos = example['obstacle'].prev_pos
        obstacle_faces = example['obstacle'].faces_batch.T

        prev_pos = example['cloth'].pos
        pred_pos = example['cloth'].pred_pos

        obstacle_face_pos = gather(obstacle_pos, obstacle_faces, 0, 1, 1).mean(dim=-2)

        obstacle_face_prev_pos = gather(obstacle_prev_pos, obstacle_faces, 0, 1, 1).mean(dim=-2)
        obstacle_fn = self.f_normals_f(obstacle_pos.unsqueeze(0), obstacle_faces.unsqueeze(0))[0]

        prev_pos = prev_pos.unsqueeze(0)
        obstacle_face_prev_pos = obstacle_face_prev_pos.unsqueeze(0)
        _, nn_idx, nn_points_prev = knn_points(prev_pos, obstacle_face_prev_pos, return_nn=True)
        nn_idx = nn_idx[0]

        nn_points = gather(obstacle_face_pos, nn_idx, 0, 1, 1)
        nn_normals = gather(obstacle_fn, nn_idx, 0, 1, 1)

        nn_points = nn_points[:, 0]
        nn_normals = nn_normals[:, 0]
        device = pred_pos.device

        distance = ((pred_pos - nn_points) * nn_normals).sum(dim=-1)
        interpenetration = torch.maximum(self.eps - distance, torch.FloatTensor([0]).to(device))
        loss = interpenetration.pow(3).sum(-1)
        return loss

    def _calc_self_loss(self, example):
        """L_s: garment self-penetration loss."""
        pred_pos = example['cloth'].pred_pos
        cloth_normals = example['cloth'].normals

        # KNN k=2: first neighbor is self, take second
        _, nn_idx, _ = knn_points(pred_pos.unsqueeze(0), pred_pos.unsqueeze(0),
                                  k=2, return_nn=True)
        nn_idx = nn_idx[0, :, 1]

        nn_pos = pred_pos[nn_idx]
        nn_normals = cloth_normals[nn_idx]
        device = pred_pos.device

        distance = ((pred_pos - nn_pos) * nn_normals).sum(dim=-1)

        # Filter mesh-adjacent vertices by Euclidean distance threshold
        pair_dist = torch.norm(pred_pos - nn_pos, dim=-1)
        neighbor_mask = pair_dist > self.eps

        interpenetration = torch.maximum(self.eps - distance, torch.FloatTensor([0]).to(device))
        interpenetration = interpenetration * neighbor_mask.float()
        loss = interpenetration.pow(3).sum(-1)
        return loss

    def calc_loss_prev_obstacle(self, example):
        external_loss = self._calc_external_loss(example)
        self_loss = self._calc_self_loss(example)
        loss = external_loss + self_loss
        return loss

    def forward(self, sample):
        B = sample.num_graphs
        iter_num = sample['cloth'].iter[0].item()
        weight = self.get_weight(iter_num)

        loss_list = []
        for i in range(B):
            loss_list.append(
                self.calc_loss_prev_obstacle(sample.get_example(i)))

        loss = sum(loss_list) / B * weight

        return dict(loss=loss)
