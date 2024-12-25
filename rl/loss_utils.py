import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.pytorch_utils import ActionLookupTable, ActionLookupTable_for9Drot, ActionLookupTable_for18Drot, ActionLookupTable_for16D
from rl.so3 import compute_err_deg_from_quats, batch_torch_A_to_R, quaternion_to_matrix, \
    so3_relative_angle, axis_angle_to_quaternion, rotate2xyz_torch, axis_angle_to_matrix
from rl.fisher_laplace import fisher_nll, laplace_nll
loss_str_to_function = {"l2":F.mse_loss, "l1":F.l1_loss, "huber":F.huber_loss, "laplace":laplace_nll, "fisher": fisher_nll}

class ActionLoss(nn.Module):
    def __init__(self,
                 rotation_dim=9,
                 probability_weight=1e0,
                 open_close_weight=1e0,
                 translation_weight=1e0,
                 rotation_weight=1e0,
                 open_close_loss="l2",
                 translation_loss="l2",
                 rotation_loss="l2",
                 translation_unit=1e0,):
        super().__init__()
        self.rotation_dim = rotation_dim
        if rotation_dim == 3:
            self.alt = ActionLookupTable
            self.action_type_num = 4
        elif rotation_dim == 9:
            self.alt = ActionLookupTable_for9Drot
            self.action_type_num = 4
        elif rotation_dim == 6:
            self.alt = ActionLookupTable_for16D
            self.action_type_num = 5
        elif rotation_dim == 18:
            self.alt = ActionLookupTable_for18Drot
            self.action_type_num = 5
        else:
            raise ValueError(f"rotation_dim: {rotation_dim}")

        self.probability_weight = probability_weight
        self.open_close_weight = open_close_weight
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.translation_unit = translation_unit

        self.open_close_loss = loss_str_to_function[open_close_loss]
        self.translation_loss = loss_str_to_function[translation_loss]
        self.rotation_loss = loss_str_to_function[rotation_loss]

    def forward(self, pred: torch.Tensor, target: torch.Tensor, reduce: bool=True):
        """
        Input:
            pred: [B, self.action_type_num + self.action_value_dim] 
                first self.action_type_num dim are probability
                last self.action_value_dim = 1 + 1 + 3 + rotation_dim is each action's dim
            target: [B, 1 + 3]
        """
        B, P = pred.shape
        assert P == self.action_type_num + 1 + 1 + 3 + self.rotation_dim

        B_, T = target.shape
        assert B == B_ and T == 4

        info = {}

        action_idx = torch.round(target[:, 0]).long()
        log_prob = F.log_softmax(pred[:, :self.action_type_num], dim=1)

        if reduce:
            prob_loss = F.nll_loss(log_prob, action_idx, reduction="mean") * self.probability_weight
        else:
            prob_loss = F.nll_loss(log_prob, action_idx, reduction="none") * self.probability_weight
        info["class_loss"] = prob_loss.detach().cpu()
        action_loss = torch.zeros_like(prob_loss)

        for i in range(self.action_type_num):
            idx1 = torch.where(action_idx == i)[0][:, None]
            cnt = idx1.shape[0]
            
            if i == self.alt["Open"]["ActionID"]:
                open_loss = torch.zeros_like(action_loss)
                if cnt > 0:
                    idx2t = torch.tensor(self.alt["Open"]["TargetIndices"], dtype=torch.int64)[None, :]
                    idx2p = torch.tensor(self.alt["Open"]["PredIndices"], dtype=torch.int64)[None, :]
                    t = target[idx1, idx2t]
                    p = pred[idx1, idx2p]
                    if reduce:
                        open_loss = self.open_close_loss(p, t, reduction="mean") 
                        action_loss += open_loss * (cnt * self.open_close_weight / B)
                        info["open_err"] = torch.rad2deg(torch.mean(torch.abs(p - t)).detach().cpu())
                    else:
                        open_loss[idx1.squeeze(1)] = torch.mean(self.open_close_loss(p, t, reduction="none"), dim=1)
                        action_loss += open_loss * self.open_close_weight
                    info["open_loss"] = open_loss.detach().cpu()
                
            elif i == self.alt["Close"]["ActionID"]:
                close_loss = torch.zeros_like(action_loss)
                if cnt > 0:
                    idx2t = torch.tensor(self.alt["Close"]["TargetIndices"], dtype=torch.int64)[None, :]
                    idx2p = torch.tensor(self.alt["Close"]["PredIndices"], dtype=torch.int64)[None, :]
                    t = target[idx1, idx2t]
                    p = pred[idx1, idx2p]
                    if reduce:
                        close_loss = self.open_close_loss(p, t, reduction="mean")
                        action_loss += close_loss * (cnt * self.open_close_weight / B)
                        info["close_err"] = torch.rad2deg(torch.mean(torch.abs(p - t)).detach().cpu())
                    else:
                        close_loss[idx1.squeeze(1)] = torch.mean(self.open_close_loss(p, t, reduction="none"), dim=1)
                        action_loss += close_loss * self.open_close_weight
                    info["close_loss"] = close_loss.detach().cpu()
                
            elif i == self.alt["Translate"]["ActionID"]:
                translation_loss = torch.zeros_like(action_loss)
                if cnt > 0:
                    idx2t = torch.tensor(self.alt["Translate"]["TargetIndices"], dtype=torch.int64)[None, :]
                    idx2p = torch.tensor(self.alt["Translate"]["PredIndices"], dtype=torch.int64)[None, :]
                    t = target[idx1, idx2t]
                    p = pred[idx1, idx2p]
                    if reduce:
                        translation_loss = self.translation_loss(p, t, reduction="mean")
                        action_loss += translation_loss * (cnt * self.translation_weight / B)
                        info["translation_err"] = torch.mean(torch.abs(p - t)).detach().cpu() * self.translation_unit
                    else:
                        translation_loss[idx1.squeeze(1)] = torch.mean(self.translation_loss(p, t, reduction="none"), dim=1)
                        action_loss += translation_loss * self.translation_weight
                    info["translation_loss"] = translation_loss.detach().cpu()
                
            elif i == self.alt["Rotate"]["ActionID"]:
                rotation_loss = torch.zeros_like(action_loss)
                if cnt > 0:
                    idx2t = torch.tensor(self.alt["Rotate"]["TargetIndices"], dtype=torch.int64)[None, :]
                    idx2p = torch.tensor(self.alt["Rotate"]["PredIndices"], dtype=torch.int64)[None, :]
                    if self.rotation_dim == 3:
                        t = axis_angle_to_quaternion(rotate2xyz_torch(target[idx1, idx2t]))
                        p = axis_angle_to_quaternion(rotate2xyz_torch(pred[idx1, idx2p]))
                    elif self.rotation_dim in [9, 18]:
                        t = quaternion_to_matrix(axis_angle_to_quaternion(rotate2xyz_torch(target[idx1, idx2t])))
                        p = batch_torch_A_to_R(pred[idx1, idx2p])
                    elif self.rotation_dim == 6:
                        t = axis_angle_to_quaternion(target[idx1, idx2t])
                        p = axis_angle_to_quaternion(pred[idx1, idx2p])
                    else:
                        raise NotImplementedError
                    if reduce:
                        rotation_loss = self.rotation_loss(p, t, reduction="mean")
                        action_loss += rotation_loss * (cnt * self.rotation_weight / B)
                        if self.rotation_dim in [3, 6]:
                            info["rotation_err"] = torch.mean(torch.abs(compute_err_deg_from_quats(p, t))).detach().cpu()
                        elif self.rotation_dim in [9, 18]:
                            info["rotation_err"] = torch.mean(torch.abs(torch.rad2deg(so3_relative_angle(p, t)))).detach().cpu()
                        else:
                            raise NotImplementedError
                    else:
                        rotation_loss[idx1.squeeze(1)] = torch.mean(self.rotation_loss(p, t, reduction="none"), dim=1)
                        action_loss += rotation_loss * self.rotation_weight
                    info["rotation_loss"] = rotation_loss.detach().cpu()

            elif i == self.alt["RotateSmall"]["ActionID"]:
                rotation_loss = torch.zeros_like(action_loss)
                if cnt > 0:
                    idx2t = torch.tensor(self.alt["RotateSmall"]["TargetIndices"], dtype=torch.int64)[None, :]
                    idx2p = torch.tensor(self.alt["RotateSmall"]["PredIndices"], dtype=torch.int64)[None, :]
                    if self.rotation_dim == 3:
                        t = axis_angle_to_quaternion(rotate2xyz_torch(target[idx1, idx2t]))
                        p = axis_angle_to_quaternion(rotate2xyz_torch(pred[idx1, idx2p]))
                    elif self.rotation_dim in [9, 18]:
                        t = quaternion_to_matrix(axis_angle_to_quaternion(rotate2xyz_torch(target[idx1, idx2t])))
                        p = batch_torch_A_to_R(pred[idx1, idx2p])
                    elif self.rotation_dim == 6:
                        t = axis_angle_to_quaternion(target[idx1, idx2t])
                        p = axis_angle_to_quaternion(pred[idx1, idx2p])
                    else:
                        raise NotImplementedError
                    if reduce:
                        rotation_loss = self.rotation_loss(p, t, reduction="mean")
                        action_loss += rotation_loss * (cnt * self.rotation_weight / B)
                        if self.rotation_dim in [3, 6]:
                            info["rotation_small_err"] = torch.mean(torch.abs(compute_err_deg_from_quats(p, t))).detach().cpu()
                        elif self.rotation_dim in [9, 18]:
                            info["rotation_small_err"] = torch.mean(torch.abs(torch.rad2deg(so3_relative_angle(p, t)))).detach().cpu()
                        else:
                            raise NotImplementedError
                    else:
                        rotation_loss[idx1.squeeze(1)] = torch.mean(self.rotation_loss(p, t, reduction="none"), dim=1)
                        action_loss += rotation_loss * self.rotation_weight
                    info["rotation_small_loss"] = rotation_loss.detach().cpu()

            else:
                raise ValueError(f"{i}")

        return prob_loss + action_loss, info


def action_gt_4D_to_18D(ac_gt: torch.Tensor, dtype=torch.float32):
    assert len(ac_gt.shape) == 2 and ac_gt.shape[1] == 4
    B = ac_gt.shape[0]
    device = ac_gt.device
    ret = torch.zeros((B, 18), dtype=dtype, device=device)
    action_num = 4
    alt = ActionLookupTable_for9Drot
    
    action_idx = torch.round(ac_gt[:, 0]).long()
    ret[:, :action_num] = F.one_hot(action_idx, action_num).float()

    for i in range(action_num):
        idx1 = torch.where(action_idx == i)[0][:, None]
        if i == alt["Open"]["ActionID"]:
            idx2t = torch.tensor(alt["Open"]["TargetIndices"], dtype=torch.int64)[None, :]
            idx2p = torch.tensor(alt["Open"]["PredIndices"], dtype=torch.int64)[None, :]
            ret[idx1, idx2p] = ac_gt[idx1, idx2t]

        elif i == alt["Close"]["ActionID"]:
            idx2t = torch.tensor(alt["Close"]["TargetIndices"], dtype=torch.int64)[None, :]
            idx2p = torch.tensor(alt["Close"]["PredIndices"], dtype=torch.int64)[None, :]
            ret[idx1, idx2p] = ac_gt[idx1, idx2t]

        elif i == alt["Translate"]["ActionID"]:
            idx2t = torch.tensor(alt["Translate"]["TargetIndices"], dtype=torch.int64)[None, :]
            idx2p = torch.tensor(alt["Translate"]["PredIndices"], dtype=torch.int64)[None, :]
            ret[idx1, idx2p] = ac_gt[idx1, idx2t]
                        
        elif i == alt["Rotate"]["ActionID"]:
            idx2t = torch.tensor(alt["Rotate"]["TargetIndices"], dtype=torch.int64)[None, :]
            idx2p = torch.tensor(alt["Rotate"]["PredIndices"], dtype=torch.int64)[None, :]
            ret[idx1, idx2p] = quaternion_to_matrix(axis_angle_to_quaternion(rotate2xyz_torch(ac_gt[idx1, idx2t]))).view(-1, 9)
            
        else:
            raise ValueError
        
    return ret

def action_gt_4D_to_long_vector(ac_gt: torch.Tensor, alt: dict, dtype=torch.float32) -> torch.Tensor:
    assert len(ac_gt.shape) == 2 and ac_gt.shape[1] == 4
    B = ac_gt.shape[0]
    device = ac_gt.device
    ret = torch.zeros((B, 18), dtype=dtype, device=device)
    action_num = 5

    action_idx = torch.round(ac_gt[:, 0]).long()
    ret = torch.zeros((B, 11), dtype=torch.float32, device=ac_gt.device)

    for i in range(action_num):
        idx1 = torch.where(action_idx == i)[0][:, None]
        if i == alt["Open"]["ActionID"]:
            idx2t = torch.tensor(alt["Open"]["TargetIndices"], dtype=torch.int64)[None, :]
            idx2p = torch.tensor([0], dtype=torch.int64)[None, :]
            ret[idx1, idx2p] = ac_gt[idx1, idx2t]

        elif i == alt["Close"]["ActionID"]:
            idx2t = torch.tensor(alt["Close"]["TargetIndices"], dtype=torch.int64)[None, :]
            idx2p = torch.tensor([1], dtype=torch.int64)[None, :]
            ret[idx1, idx2p] = ac_gt[idx1, idx2t]

        elif i == alt["Translate"]["ActionID"]:
            idx2t = torch.tensor(alt["Translate"]["TargetIndices"], dtype=torch.int64)[None, :]
            idx2p = torch.tensor([2, 3, 4], dtype=torch.int64)[None, :]
            ret[idx1, idx2p] = ac_gt[idx1, idx2t]
                        
        elif i == alt["Rotate"]["ActionID"]:
            idx2t = torch.tensor(alt["Rotate"]["TargetIndices"], dtype=torch.int64)[None, :]
            idx2p = torch.tensor([5, 6, 7], dtype=torch.int64)[None, :]
            ret[idx1, idx2p] = rotate2xyz_torch(ac_gt[idx1, idx2t])

        elif i == alt["RotateSmall"]["ActionID"]:
            idx2t = torch.tensor(alt["RotateSmall"]["TargetIndices"], dtype=torch.int64)[None, :]
            idx2p = torch.tensor([8, 9, 10], dtype=torch.int64)[None, :]
            ret[idx1, idx2p] = rotate2xyz_torch(ac_gt[idx1, idx2t])
            
        else:
            raise ValueError
        
    return ret

def chamfer_distance(x: torch.Tensor, y: torch.Tensor, p=1, reduce=True):
    """
    Args:
        x: [B, N, D]
        y: [B, M, D]
    Return:
        2-way CD
        if reduce == True:
            return scalar
        else:
            return [B, ]
    """
    assert x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2], f"invalid shape {x.shape}, {y.shape}"
    B, N, D = x.shape
    _, M, _ = y.shape
    dist = torch.norm(x.view(B, N, 1, D) - y.view(B, 1, M, D), dim=3) ** p
    x_loss = torch.mean(torch.min(dist, dim=2)[0], dim=1)
    y_loss = torch.mean(torch.min(dist, dim=1)[0], dim=1)
    loss = x_loss + y_loss
    if reduce:
        return torch.mean(loss)
    else:
        return loss
    

def earth_mover_distance(x: torch.Tensor, y: torch.Tensor, reduce=True):
    """
    Args:
        xyz1 (torch.Tensor): (b, n1, 3)
        xyz2 (torch.Tensor): (b, n1, 3)

    Returns:
        cost (torch.Tensor): (b)
    """
    import external.emd.emd as emd
    assert len(x.shape) == 3 and x.shape[2] == 3 and x.shape == y.shape
    cost = emd.EarthMoverDistanceFunction.apply(x, y)
    if reduce:
        return torch.mean(cost)
    else:
        return cost
    

class HelperPointLoss(nn.Module):
    def __init__(self, seq_len: int, xyz_unit: float, loss_name="chamfer_distance", weight=1e0, chamfer_distance_p=1) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.xyz_unit = xyz_unit
        self.weight = weight
        self.loss_name = loss_name
        self.chamfer_distance_p = chamfer_distance_p

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, reduce_over_batch=True) -> torch.Tensor:
        """
        Args:
            pred: [B, T, N, D]
            gt: [B, T, M, D]
        Return:
            if reduce:
                return scalar
            else:
                return [B, ]
        """
        B,  N, D = pred.shape
        B_, M, D_ = gt.shape
        assert B == B_ and D == D_, f"{pred.shape}, {gt.shape}"
        info = {}
        if self.loss_name == "chamfer_distance":
            loss = chamfer_distance(pred.view(-1, N, D), gt.view(-1, M, D), p=self.chamfer_distance_p, reduce=reduce_over_batch)
        elif self.loss_name == "earth_mover_distance":
            loss = earth_mover_distance(pred.view(-1, N, D), gt.view(-1, M, D), reduce=reduce_over_batch)
        else:
            raise NotImplementedError
        if not reduce_over_batch:
            loss = torch.mean(loss.view(B, T), dim=1) # always reduce over sequence
        info["helper_loss"] = loss.detach().cpu()
        info["helper_cderr"] = chamfer_distance(pred.view(-1, N, D), gt.view(-1, M, D), p=1, reduce=True).detach().cpu() * self.xyz_unit
        return self.weight * loss, info

class GoalPredLoss(nn.Module):
    def __init__(self, a_xyz_unit, b_xyz_unit, loss = 'l2', weight= 1e0) -> None:
        super().__init__()
        self.loss_func = loss_str_to_function[loss]
        self.a_xyz_unit = a_xyz_unit
        self.b_xyz_unit = b_xyz_unit
        self.weight = weight
    
    def forward(self, pred, gt, reduce_over_batch=True):
        B,  D = pred.shape
        B_, D_ = gt.shape
        assert B == B_ and D == D_, f"{pred.shape}, {gt.shape}"
        info = {}
        reduction = 'mean' if reduce_over_batch else 'none'
        loss = self.loss_func(pred, gt, reduction = reduction)
        info["goal_pred_loss"] = loss.detach().cpu()
        info["edge_a_pred_err"] = torch.mean(torch.linalg.norm((pred[:, :3] - gt[:, :3]), dim = -1)).detach().cpu() * self.a_xyz_unit
        info["edge_b_pred_err"] = torch.mean(torch.linalg.norm((pred[:, 3:] - gt[:, 3:]), dim = -1)).detach().cpu() * self.b_xyz_unit
        return self.weight* loss, info
    
class ActionLoss8D(nn.Module):
    def __init__(self,
                 close_weight = 1e0, 
                 translation_weight=1e0,
                 rotation_weight=1e0,
                 tune_weight=1e0,
                 close_loss = 'l2',
                 translation_loss="l2",
                 rotation_loss="l2",
                 tune_loss="l2",
                 translation_unit=1e0,):
        super().__init__()
        
        self.close_weight = close_weight
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.tune_weight = tune_weight

        self.translation_unit = translation_unit

        self.close_loss = loss_str_to_function[close_loss]
        self.translation_loss = loss_str_to_function[translation_loss]
        self.rotation_loss = loss_str_to_function[rotation_loss]
        self.tune_loss = loss_str_to_function[tune_loss]

    def forward(self, pred: torch.Tensor, target: torch.Tensor, reduce: bool=True):
        """
        Input:
            pred: [B, 1+ 1 + 3 + 3] # close+ push + rotate + tune
            target: [B, 1 + 3]
        """
        B, P = pred.shape
        assert P == 8, f"{pred.shape} {target.shape}"

        B_, T = target.shape
        assert B == B_ and T == 4, f"{pred.shape} {target.shape}"

        info = {}

        action_idx = torch.round(target[:, 0]).long()

        if reduce:
            action_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
        else:
            action_loss = torch.zeros((B, ), dtype=pred.dtype, device=pred.device)

        def relative_angle(a_vec3: torch.Tensor, b_vec3: torch.Tensor, eps=1e-7):
            return torch.acos(torch.sum(a_vec3 * b_vec3, dim=1) / (torch.norm(a_vec3, dim=1) * torch.norm(b_vec3, dim=1) + eps))

        for i in range(4):
            idx1 = torch.where(action_idx == i)[0][:, None]
            cnt = idx1.shape[0]

            if i == 0: # close:
                close_loss = torch.zeros_like(action_loss)
                if cnt > 0:
                    idx2t = torch.tensor([1], dtype=torch.int64)[None, :]
                    idx2p = torch.tensor([0], dtype=torch.int64)[None, :]
                    t = target[idx1, idx2t]
                    p = pred[idx1, idx2p]
                    if reduce:
                        close_loss = self.close_loss(p, t, reduction="mean")
                        action_loss += close_loss * (cnt * self.close_weight / B)
                        info["close_err"] = torch.dist(p, t).detach().cpu()
                    else:
                        close_loss[idx1.squeeze(1)] = torch.mean(self.close_loss(p, t, reduction="none"), dim=1)
                        action_loss += close_loss * self.close_weight
                    info["close_loss"] = close_loss.detach().cpu()

            elif i == 1: # translation:
                translation_loss = torch.zeros_like(action_loss)
                if cnt > 0:
                    idx2t = torch.tensor([1], dtype=torch.int64)[None, :]
                    idx2p = torch.tensor([1], dtype=torch.int64)[None, :]
                    t = target[idx1, idx2t]
                    p = pred[idx1, idx2p]
                    if reduce:
                        translation_loss = self.translation_loss(p, t, reduction="mean")
                        action_loss += translation_loss * (cnt * self.translation_weight / B)
                        info["translation_err"] = torch.dist(p, t).detach().cpu()
                    else:
                        translation_loss[idx1.squeeze(1)] = torch.mean(self.translation_loss(p, t, reduction="none"), dim=1)
                        action_loss += translation_loss * self.translation_weight
                    info["translation_loss"] = translation_loss.detach().cpu()
                
            elif i == 2: # rotation
                rotation_loss = torch.zeros_like(action_loss)
                if cnt > 0:
                    idx2t = torch.tensor([1, 2, 3], dtype=torch.int64)[None, :]
                    idx2p = torch.tensor([2, 3, 4], dtype=torch.int64)[None, :]
                    t = target[idx1, idx2t]
                    p = pred[idx1, idx2p]
                    if reduce:
                        rotation_loss = self.rotation_loss(p, t, reduction="mean")
                        action_loss += rotation_loss * (cnt * self.rotation_weight / B)
                        info["rotation_err"] = torch.mean(torch.abs(torch.rad2deg(relative_angle(p, t)))).detach().cpu()
                    else:
                        rotation_loss[idx1.squeeze(1)] = torch.mean(self.rotation_loss(p, t, reduction="none"), dim=1)
                        action_loss += rotation_loss * self.rotation_weight
                    info["rotation_loss"] = rotation_loss.detach().cpu()

            elif i == 3: # tune
                tune_loss = torch.zeros_like(action_loss)
                if cnt > 0:
                    idx2t = torch.tensor([1, 2, 3], dtype=torch.int64)[None, :]
                    idx2p = torch.tensor([5, 6, 7], dtype=torch.int64)[None, :]
                    t = target[idx1, idx2t]
                    p = pred[idx1, idx2p]
                    if reduce:
                        tune_loss = self.tune_loss(p, t, reduction="mean")
                        action_loss += tune_loss * (cnt * self.tune_weight / B)
                        info["tune_err"] = torch.mean(torch.abs(torch.rad2deg(relative_angle(p, t)))).detach().cpu()
                    else:
                        tune_loss[idx1.squeeze(1)] = torch.mean(self.tune_loss(p, t, reduction="none"), dim=1)
                        action_loss += tune_loss * self.tune_weight
                    info["tune_loss"] = tune_loss.detach().cpu()

            else:
                raise ValueError(f"{i}")

        return action_loss, info

class ActionLossDelta(nn.Module):
    def __init__(self,
                 close_weight = 1e0, 
                 translation_weight=1e0,
                 rotation_weight=1e0,
                 tune_weight=1e0,
                 close_loss = 'l2',
                 translation_loss="l2",
                 rotation_loss="l2",
                 tune_loss="l2",
                 translation_unit=1e0,):
        super().__init__()
        
        self.close_weight = close_weight
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.tune_weight = tune_weight

        self.translation_unit = translation_unit

        self.close_loss = loss_str_to_function[close_loss]
        self.translation_loss = loss_str_to_function[translation_loss]
        self.rotation_loss = loss_str_to_function[rotation_loss]
        self.tune_loss = loss_str_to_function[tune_loss]

    def forward(self, pred: torch.Tensor, target: torch.Tensor, pose:torch.Tensor, reduce: bool=True):
        """
        Input:
            pred: [B, 1+ 1 + 3 + 3] # close+ push + rotate + tune
            target: [B, 1 + 3]
        """
        B, P = pred.shape
        assert P == 20, f"{pred.shape} {target.shape}"

        B_, T = target.shape
        assert B == B_ and T == 4, f"{pred.shape} {target.shape}"

        info = {}

        action_idx = torch.round(target[:, 0]).long()

        if reduce:
            action_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
        else:
            action_loss = torch.zeros((B, ), dtype=pred.dtype, device=pred.device)

        def relative_angle(a_vec3: torch.Tensor, b_vec3: torch.Tensor, eps=1e-7):
            return torch.acos(torch.sum(a_vec3 * b_vec3, dim=1) / (torch.norm(a_vec3, dim=1) * torch.norm(b_vec3, dim=1) + eps))

        for i in torch.unique(action_idx):
            idx1 = torch.where(action_idx == i)[0][:, None]
            cnt = idx1.shape[0]

            if i == 0: # close:
                close_loss = torch.zeros_like(action_loss)
                if cnt > 0:
                    idx2t = torch.tensor([1], dtype=torch.int64)[None, :]
                    idx2p = torch.tensor([0], dtype=torch.int64)[None, :]
                    t = target[idx1, idx2t]
                    p = pred[idx1, idx2p]
                    if reduce:
                        close_loss = self.close_loss(p, t, reduction="mean")
                        action_loss += close_loss * (cnt * self.close_weight / B)
                        info["close_err"] = torch.dist(p, t).detach().cpu()
                    else:
                        close_loss[idx1.squeeze(1)] = torch.mean(self.close_loss(p, t, reduction="none"), dim=1)
                        action_loss += close_loss * self.close_weight
                    info["close_loss"] = close_loss.detach().cpu()

            elif i == 1: # translation:
                translation_loss = torch.zeros_like(action_loss)
                if cnt > 0:
                    idx2t = torch.tensor([1], dtype=torch.int64)[None, :]
                    idx2p = torch.tensor([1], dtype=torch.int64)[None, :]
                    t = target[idx1, idx2t]
                    p = pred[idx1, idx2p]
                    if reduce:
                        translation_loss = self.translation_loss(p, t, reduction="mean")
                        action_loss += translation_loss * (cnt * self.translation_weight / B)
                        info["translation_err"] = torch.dist(p, t).detach().cpu()
                    else:
                        translation_loss[idx1.squeeze(1)] = torch.mean(self.translation_loss(p, t, reduction="none"), dim=1)
                        action_loss += translation_loss * self.translation_weight
                    info["translation_loss"] = translation_loss.detach().cpu()
                
            elif i == 2: # rotation
                rotation_loss = torch.zeros_like(action_loss)
                if cnt > 0:
                    idx2t = torch.tensor([1, 2, 3], dtype=torch.int64)[None, :]
                    idx2p = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64)[None, :]
                    t = target[idx1, idx2t]
                    cut_direc = pose[idx1, 3:].squeeze(1)
                    gt_rotation = delta_rotation_batch(cut_direc, t)
                    p = pred[idx1, idx2p]
                    pred_rotation = batch_torch_A_to_R(p)
                    if reduce:
                        rotation_loss = self.rotation_loss(pred_rotation, gt_rotation, reduction="mean")
                        action_loss += rotation_loss * (cnt * self.rotation_weight / B)
                        info["rotation_err"] = torch.mean(torch.abs(torch.rad2deg(so3_relative_angle(pred_rotation, gt_rotation)))).detach().cpu()
                    else:
                        rotation_loss[idx1.squeeze(1)] = torch.mean(self.rotation_loss(pred_rotation, gt_rotation, reduction="none"), dim=1)
                        action_loss += rotation_loss * self.rotation_weight
                    info["rotation_loss"] = rotation_loss.detach().cpu()

            elif i == 3: # tune
                tune_loss = torch.zeros_like(action_loss)
                if cnt > 0:
                    idx2t = torch.tensor([1, 2, 3], dtype=torch.int64)[None, :]
                    idx2p = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=torch.int64)[None, :]
                    t = target[idx1, idx2t]
                    cut_direc = pose[idx1, 3:].squeeze(1)
                    gt_rotation = delta_rotation_batch(cut_direc, t)
                    p = pred[idx1, idx2p]
                    pred_rotation = batch_torch_A_to_R(p)
                    if reduce:
                        tune_loss = self.tune_loss(pred_rotation, gt_rotation, reduction="mean")
                        action_loss += tune_loss * (cnt * self.tune_weight / B)
                        info["tune_err"] = torch.mean(torch.abs(torch.rad2deg(relative_angle(pred_rotation, gt_rotation)))).detach().cpu()
                    else:
                        tune_loss[idx1.squeeze(1)] = torch.mean(self.tune_loss(pred_rotation, gt_rotation, reduction="none"), dim=1)
                        action_loss += tune_loss * self.tune_weight
                    info["tune_loss"] = tune_loss.detach().cpu()

            elif i == -2:
                continue
            
            else:
                raise ValueError(f"{i}")

        return action_loss, info

class ActionChunkingLoss(nn.Module):
    def __init__(self,
                 close_weight = 1e0, 
                 translation_weight=1e0,
                 rotation_weight=1e0,
                 tune_weight=1e0,
                 close_loss = 'l2',
                 translation_loss="l2",
                 rotation_loss="l2",
                 tune_loss="l2",
                 translation_unit=1e0,
                 chunking_weight=[1e0, 1e0, 1e0, 1e0]):
        super().__init__()
        
        self.action_loss = ActionLossDelta(close_weight, translation_weight, rotation_weight, 
                                           tune_weight, close_loss, translation_loss, rotation_loss, tune_loss, translation_unit)
        self.chunking_weight = chunking_weight
        self.chunking_size = len(chunking_weight)
    
    # fix bug for List[torch.Tensor] and List[torch.Tensor]
    def forward(self, pred: torch.Tensor, target: torch.Tensor, pose: torch.Tensor, reduce: bool=True):
        """
        Input:
            pred: N * [B, 1+ 1 + 3 + 3] # close+ push + rotate + tune   
            target: N * [B, 1 + 3]  
        """
        assert target.shape[1] == target.shape[1] == pose.shape[1] == self.chunking_size
        loss_all = torch.zeros((), dtype=pred[0].dtype, device=pred[0].device)
        info_all = dict()
        for i, weight in enumerate(self.chunking_weight):
            loss, info = self.action_loss(pred[:, i, :], target[:, i, :], pose[:, i, :], reduce)
            loss_all += weight * loss
            # integrate info into info_all and change their key into "chunking_{i}_{key}"
            for key, value in info.items():
                info_all[f"future_{i}_{key}"] = value
        return loss_all, info_all

class PoseLoss(nn.Module):
    def __init__(self,
                 angle_weight = 1e0, 
                 translation_weight=1e0,
                 rotation_weight=1e0,
                 angle_loss = 'l2',
                 translation_loss="l2",
                 rotation_loss="l2",
                 translation_unit = 1e0,):
        super().__init__()
        self.angle_weight = angle_weight
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight

        self.angle_loss = loss_str_to_function[angle_loss]
        self.translation_loss = loss_str_to_function[translation_loss]
        self.rotation_loss = loss_str_to_function[rotation_loss]

    def forward(self, pred: torch.Tensor, target: torch.Tensor, reduce: bool = True):
        """
        Input:
            pred: [B, 1 + 3 + 3] # angle + fp + cd
            target: [B, 1 + 3 + 3] # angle + fp + cd
        """
        pose2idx = dict(angle = torch.tensor([0]).to(pred.device), front_point = torch.tensor([1, 2, 3]).to(pred.device), 
                        cut_direction = torch.tensor([4, 5, 6]).to(pred.device))
        B, P = pred.shape
        assert P == 7, f"{pred.shape} {target.shape}"

        B_, T = target.shape
        assert B == B_ and T == 7, f"{pred.shape} {target.shape}"

        info = {}

        if reduce:
            action_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
        else:
            action_loss = torch.zeros((B, ), dtype=pred.dtype, device=pred.device)

        def relative_angle(a_vec3: torch.Tensor, b_vec3: torch.Tensor, eps=1e-7):
            return torch.acos(torch.sum(a_vec3 * b_vec3, dim=1) / (torch.norm(a_vec3, dim=1) * torch.norm(b_vec3, dim=1) + eps))
        
        # detect valid tensor which is not all zero in target batch
        valid_idx = torch.sum(torch.abs(target), dim=1) > 1e-7

        # rotation loss
        pred_rot = torch.index_select(pred[valid_idx], 1, pose2idx["cut_direction"])
        target_rot = torch.index_select(target[valid_idx], 1, pose2idx["cut_direction"])
        if reduce:
            rotation_loss = self.rotation_loss(pred_rot, target_rot, reduction="mean")
            action_loss += rotation_loss * self.rotation_weight
            info["rotation_err"] = torch.mean(torch.abs(torch.rad2deg(relative_angle(pred_rot, target_rot)))).detach().cpu()
        else:
            rotation_loss = torch.mean(self.rotation_loss(pred_rot, target_rot, reduction="none"), dim=1)
            action_loss += rotation_loss * self.rotation_weight
        info["rotation_loss"] = rotation_loss.detach().cpu()

        # translation loss
        pred_trans = torch.index_select(pred[valid_idx], 1, pose2idx["front_point"])
        target_trans = torch.index_select(target[valid_idx], 1, pose2idx["front_point"])
        if reduce:
            translation_loss = self.translation_loss(pred_trans, target_trans, reduction="mean")
            action_loss += translation_loss * self.translation_weight 
            info["translation_err"] = torch.norm(pred_trans - target_trans, dim = -1).mean().detach().cpu()
        else:
            translation_loss = torch.mean(self.translation_loss(pred_trans, target_trans, reduction="none"), dim=1)
            action_loss += translation_loss * self.translation_weight
        info["translation_loss"] = translation_loss.detach().cpu()

        # angle loss
        pred_angle = torch.index_select(pred[valid_idx], 1, pose2idx["angle"])
        target_angle = torch.index_select(target[valid_idx], 1, pose2idx["angle"])
        if reduce:
            angle_loss = self.angle_loss(pred_angle, target_angle, reduction="mean")
            action_loss += angle_loss *  self.angle_weight 
            info["angle_err"] = torch.norm(pred_angle - target_angle, dim = -1).mean().detach().cpu()
        else:
            angle_loss= torch.mean(self.angle_loss(pred_angle, target_angle, reduction="none"), dim=1)
            action_loss += angle_loss * self.angle_weight
        info["angle_loss"] = angle_loss.detach().cpu()

        return action_loss, info

class ActionChunkingPoseLoss(nn.Module):
    def __init__(self,
                angle_weight = 1e0, 
                translation_weight=1e0,
                rotation_weight=1e0,
                angle_loss = 'l2',
                translation_loss="l2",
                rotation_loss="l2",
                translation_unit=1e0,
                chunking_weight=[1e0, 1e0, 1e0, 1e0]):
        super().__init__()
        
        self.action_loss = PoseLoss(angle_weight, translation_weight, rotation_weight, 
                                           angle_loss, translation_loss, rotation_loss,translation_unit)
        self.chunking_weight = chunking_weight
        self.chunking_size = len(chunking_weight)
    
    # fix bug for List[torch.Tensor] and List[torch.Tensor]
    def forward(self, pred: torch.Tensor, target: torch.Tensor, reduce: bool=True):
        """
        Input:
            pred: N * [B, 1+ 1 + 3 + 3] # close+ push + rotate + tune   
            target: N * [B, 1 + 3]  
        """
        assert target.shape[1] == target.shape[1]== self.chunking_size
        loss_all = torch.zeros((), dtype=pred[0].dtype, device=pred[0].device)
        info_all = dict()
        for i, weight in enumerate(self.chunking_weight):
            loss, info = self.action_loss(pred[:, i, :], target[:, i, :], reduce)
            loss_all += weight * loss
            # integrate info into info_all and change their key into "chunking_{i}_{key}"
            for key, value in info.items():
                info_all[f"future_{i}_{key}"] = value
        return loss_all, info_all
        
def delta_rotation_batch(current, target):
    """
    current: torch.Tensor (batch_size, 3)
    target: torch.Tensor (batch_size, 3)
    """
    current = current / torch.norm(current, dim=-1, keepdim=True)
    target = target / torch.norm(target, dim=-1, keepdim=True)

    axis = torch.cross(current, target, dim=-1)
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)

    dot_product = torch.clamp(torch.sum(current * target, dim=-1), -1, 1)
    angle = torch.acos(dot_product).reshape(-1, 1)

    axis_angle = axis * angle
    rot = axis_angle_to_matrix(axis_angle)
    return rot