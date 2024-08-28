import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from .loss import alignment_loss_func

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, use_center_mask, cfg):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        self.with_type_embedding = cfg.MODEL.MOTION.MULTI_TYPE 

        self.use_center_mask = use_center_mask
        self.cfg = cfg
        
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.motion_cfg = cfg.MODEL.MOTION
        self.detr_cfg = cfg.LOSS.DETR
        self.use_attr_gmm = cfg.MODEL.SCENE.INIT_CFG.DECODER.ATTR_GMM_ENABLE

    def loss_labels(self, outputs, data, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        use_background = self.detr_cfg.PRED_BACKGROUND
        full_cls = self.num_classes if use_background else 0

        targets = data['targets']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).long()
        target_classes = torch.full(src_logits.shape[:2], full_cls,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if self.use_center_mask:
            # mask out the non-center lanes
            line_mask = data['center_mask']
            
            if use_background:
                bg_mask = torch.ones_like(line_mask[:, :1])
                line_mask = torch.cat([line_mask,bg_mask], dim=1)

            line_mask = line_mask.unsqueeze(1).repeat(1, src_logits.shape[1], 1)
            src_logits[~line_mask] = -1e9

        if use_background:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, reduce=False)
            loss_ce[~data['agent_mask']] *= 0
            cnt = torch.sum(data['agent_mask'])
            if cnt > 0:
                loss_ce = loss_ce.sum() / cnt
            else:
                loss_ce = loss_ce.sum()

        losses = {'labels': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

            if use_background:
                target_classes = target_classes.flatten(0, 1)
                bg_mask = target_classes == self.num_classes
                bg_logits = src_logits.flatten(0, 1)[bg_mask]
                bg_target = target_classes[bg_mask]
                losses['background_error'] = 100 - accuracy(bg_logits, bg_target)[0]
        return losses
    
    def pos_rel(self, ego_heading, ego_pos, other_pos):
        # angle_init = ego_heading[0]
        pos_rel = other_pos - ego_pos
        dis_rel = torch.linalg.norm(pos_rel)
        degree_pos_rel = 0 # int(torch.clip(dis_rel/2.5, a_min=0, a_max=8))
        deg_other = torch.arctan2(pos_rel[1], pos_rel[0])
        deg_rel = deg_other - ego_heading
        if deg_rel > torch.pi:
            deg_rel -= 2 * torch.pi
        elif deg_rel < -1 * torch.pi:
            deg_rel += 2*torch.pi

        if dis_rel <= 0.1:
            ang = -1        
        elif deg_rel < torch.pi/9 and deg_rel > -1 * torch.pi / 9:
            ang = 0
        elif deg_rel <= -1 * torch.pi / 9 and deg_rel > -1 * torch.pi / 2:
            ang = 1
        elif  deg_rel <= -1 * torch.pi / 2 and deg_rel > -8 * torch.pi/9:
            ang = 2
        elif deg_rel >= torch.pi/9 and deg_rel < torch.pi/2:
            ang = 5
        elif deg_rel >= torch.pi/2 and deg_rel < 8*torch.pi/9:
            ang = 4
        else:
            ang = 3

        return [degree_pos_rel, ang]
    
    def _compute_motion_loss(self, src_motion, src_probs, target_motion, target_motion_mask, loss_func, motion_attrs, traj_type, gt_pos_f, gt_pos_i, pos_attrs, gt_type_pos, veh_type = None):
        pred_other_attr = self.motion_cfg.PRED_HEADING_VEL
        type_frequency = [0.42, 0.50, 0.02, 0.02, 0.02, 0.02] 
        weight_frequency = [1.0 / t for t in type_frequency]
        loss_attr = []
        motion_attr_loss = {'motion_pos': []}
        if pred_other_attr:
            motion_attr_loss['motion_heading'] = []
            motion_attr_loss['motion_vel'] = []

        CLS = torch.nn.CrossEntropyLoss()
        MSE = torch.nn.MSELoss(reduction='none')
        L1 = torch.nn.L1Loss(reduction='none')

        if src_probs is None:
            src_probs = [None] * len(src_motion)
        
        if veh_type is None:
            b_idx = 0
            for src, src_prob, tgt, mask, traj, pos_f, pos_i, pos_0, gt_tp in zip(src_motion, src_probs, target_motion, target_motion_mask, traj_type, gt_pos_f, gt_pos_i, pos_attrs, gt_type_pos):
                if self.motion_cfg.PRED_MODE == 'mlp':
                    if tgt.shape[0] > 0 and mask.shape[0] > 0:
                        loss_attr.append(loss_func(src[mask], tgt[mask]))
                    else:
                        loss_attr.append([])
                elif self.motion_cfg.PRED_MODE in ['mlp_gmm', 'mtf']:
                    if tgt.shape[0] > 0 and mask.shape[0] > 0:
                        K = src.shape[1]
                        tgt_gt = tgt.unsqueeze(1).repeat(1, K, 1, 1)

                        dists = []
                        for i in range(len(src)):
                            tgt_gt_i = tgt_gt[i]
                            src_i = src[i]
                            mask_i = mask[i][:, 0]
                            # get the last idx of mask_i that is 1
                            last_idx = torch.where(mask_i)[0]
                            if len(last_idx) == 0:
                                dists.append(torch.zeros(K, device=src_i.device))
                                continue
                            last_idx = last_idx[-1]
                            tgt_end = tgt_gt_i[:, last_idx]
                            src_end = src_i[:, last_idx]
                            dist = MSE(tgt_end, src_end).mean(-1)
                            dists.append(dist)
                        dists = torch.stack(dists, dim=0)
                        min_index = traj.flatten()

                        pos_ego = src[0, min_index[0], -1, :]
                        pos_ego_init = src[0, min_index[0], 0, :]
                        init_pos_loss = []
                        final_pos_loss = []
                        type_pos_loss = []
                        for i in range(len(src)):
                            idx_traj = min_index[i]
                            pos_veh = src[i, idx_traj, -1, :]
                            pos_veh_init = src[i, idx_traj, 0, :]
                            dist_veh = torch.sqrt((pos_veh[0] - pos_ego[0])**2 + (pos_veh[1] - pos_ego[1])**2)
                            dist_veh_init = torch.sqrt((pos_veh_init[0] - pos_ego_init[0])**2 + (pos_veh_init[1] - pos_ego_init[1])**2)
                            loss_veh = MSE(dist_veh, pos_f[i])
                            loss_veh_init = MSE(dist_veh_init, pos_i[i])

                            type_pos = self.pos_rel(0.0, pos_ego, pos_veh)[1]
                            type_pos = torch.tensor(type_pos, dtype=torch.int64).to(gt_tp[i].device)
                            type_pos = type_pos.view((-1,1))
                            gt_tp[i] = torch.tensor(gt_tp[i], dtype=torch.int64).view((-1,1))
                            loss_pos_type = MSE(type_pos, gt_tp[i])

                            final_pos_loss.append(loss_veh)
                            init_pos_loss.append(loss_veh_init)
                            type_pos_loss.append(loss_pos_type)
                        
                        k_mask = mask.unsqueeze(1).repeat(1, K, 1, 1)
                        
                        pos_loss = MSE(tgt_gt, src)
                        pos_loss[~k_mask] *= 0

                        final_pos_loss = torch.tensor(final_pos_loss).to(pos_loss.device)
                        init_pos_loss = torch.tensor(init_pos_loss).to(pos_loss.device)
                        type_pos_loss = torch.tensor(type_pos_loss).to(pos_loss.device)
                        pos_loss = pos_loss.sum(-1).sum(-1) / (k_mask.sum(-1).sum(-1) + 1e-6)

                        for rel_idx in range(pos_loss.shape[0]):
                            rel_type = traj[rel_idx, 0].cpu().int()
                            pos_loss[rel_idx, rel_type] *= weight_frequency[rel_type]
                            final_pos_loss[rel_idx] *= weight_frequency[rel_type]
                            init_pos_loss[rel_idx] *= weight_frequency[rel_type]

                        pos_loss = torch.gather(pos_loss, dim=1, index=min_index.unsqueeze(-1)).mean()
                        final_pos_loss = final_pos_loss.mean() * 1.0
                        init_pos_loss = init_pos_loss.mean() * 0.01
                        type_pos_loss = type_pos_loss.mean() * 0.01
                        pos_loss *= 2.0
                        cls_loss = CLS(src_prob, min_index) * self.motion_cfg.CLS_WEIGHT
                        cls_loss *= 1.0
                        
                        motion_loss = pos_loss + cls_loss + final_pos_loss + pos_loss # + init_pos_loss
                        motion_attr_loss['motion_pos'].append(pos_loss + cls_loss + final_pos_loss + pos_loss) # + init_pos_loss) 

                        if pred_other_attr:
                            src_heading = motion_attrs['heading']['src'][b_idx]
                            src_vel = motion_attrs['vel']['src'][b_idx]
                            tgt_heading = motion_attrs['heading']['tgt'][b_idx][:, None, :, None].repeat(1, K, 1, 1)
                            tgt_vel = motion_attrs['vel']['tgt'][b_idx][:, None].repeat(1, K, 1, 1)

                            velo_loss = MSE(tgt_vel, src_vel)
                            velo_loss[~k_mask] *= 0
                            velo_loss = velo_loss.sum(-1).sum(-1) / (k_mask.sum(-1).sum(-1) + 1e-6)
                            
                            for rel_idx in range(velo_loss.shape[0]):
                                rel_type = traj[rel_idx, 0].cpu().int()
                                velo_loss[rel_idx, rel_type] *= weight_frequency[rel_type]
                            
                            velo_loss = torch.gather(velo_loss, dim=1, index=min_index.unsqueeze(-1)).mean()
                            velo_loss *= 0.1
                            heading_loss = L1(tgt_heading, src_heading)
                            heading_loss[~k_mask[...,:1]] *= 0
                            heading_loss = heading_loss.sum(-1).sum(-1) / (k_mask[...,:1].sum(-1).sum(-1) + 1e-6)
                            
                            for rel_idx in range(heading_loss.shape[0]):
                                rel_type = traj[rel_idx, 0].cpu().int()
                                heading_loss[rel_idx, rel_type] *= weight_frequency[rel_type]
                            
                            heading_loss = torch.gather(heading_loss, dim=1, index=min_index.unsqueeze(-1)).mean()
                        
                            motion_loss += velo_loss + heading_loss * 10.0

                            motion_attr_loss['motion_vel'].append(velo_loss)
                            motion_attr_loss['motion_heading'].append(heading_loss * 10.0)

                        loss_attr.append(motion_loss)
                    else:
                        loss_attr.append([])
                        motion_attr_loss['motion_pos'].append([])
                        if pred_other_attr:
                            motion_attr_loss['motion_vel'].append([])
                            motion_attr_loss['motion_heading'].append([])
                b_idx += 1
        else:
            b_idx = 0
            for src, src_prob, tgt, mask, traj, pos_f, pos_i, pos_0, gt_tp, vt in zip(src_motion, src_probs, target_motion, target_motion_mask, traj_type, gt_pos_f, gt_pos_i, pos_attrs, gt_type_pos, veh_type):
                if self.motion_cfg.PRED_MODE == 'mlp':
                    if tgt.shape[0] > 0 and mask.shape[0] > 0:
                        loss_attr.append(loss_func(src[mask], tgt[mask]))
                    else:
                        loss_attr.append([])
                elif self.motion_cfg.PRED_MODE in ['mlp_gmm', 'mtf']:
                    
                    src_temp = []
                    for j in range(src.shape[0]):
                        src_temp.append(src[j, vt[j, 0].int()-1, :, :, :])
                    src = torch.stack(src_temp, dim = 0)

                    if tgt.shape[0] > 0 and mask.shape[0] > 0:                        
                        K = src.shape[1]
                        tgt_gt = tgt.unsqueeze(1).repeat(1, K, 1, 1)

                        dists = []
                        for i in range(len(src)):
                            tgt_gt_i = tgt_gt[i]
                            src_i = src[i]
                            mask_i = mask[i][:, 0]
                            # get the last idx of mask_i that is 1
                            last_idx = torch.where(mask_i)[0]
                            if len(last_idx) == 0:
                                dists.append(torch.zeros(K, device=src_i.device))
                                continue
                            last_idx = last_idx[-1]
                            tgt_end = tgt_gt_i[:, last_idx]
                            src_end = src_i[:, last_idx]
                            dist = MSE(tgt_end, src_end).mean(-1)
                            dists.append(dist)
                        dists = torch.stack(dists, dim=0)
                        min_index = traj.flatten()
                        pos_ego = src[0, min_index[0], -1, :]
                        pos_ego_init = src[0, min_index[0], 0, :]
                        init_pos_loss = []
                        final_pos_loss = []
                        type_pos_loss = []
                        for i in range(len(src)):
                            idx_traj = min_index[i]
                            pos_veh = src[i, idx_traj, -1, :]
                            pos_veh_init = src[i, idx_traj, 0, :]
                            dist_veh = torch.sqrt((pos_veh[0] - pos_ego[0])**2 + (pos_veh[1] - pos_ego[1])**2)
                            dist_veh_init = torch.sqrt((pos_veh_init[0] - pos_ego_init[0])**2 + (pos_veh_init[1] - pos_ego_init[1])**2)
                            loss_veh = MSE(dist_veh, pos_f[i])
                            loss_veh_init = MSE(dist_veh_init, pos_i[i])

                            type_pos = self.pos_rel(0.0, pos_ego, pos_veh)[1]
                            type_pos = torch.tensor(type_pos, dtype=torch.int64).to(gt_tp[i].device)
                            type_pos = type_pos.view((-1,1))
                            gt_tp[i] = torch.tensor(gt_tp[i], dtype=torch.int64).view((-1,1))
                            loss_pos_type = MSE(type_pos, gt_tp[i])

                            final_pos_loss.append(loss_veh)
                            init_pos_loss.append(loss_veh_init)
                            type_pos_loss.append(loss_pos_type)
                        
                        k_mask = mask.unsqueeze(1).repeat(1, K, 1, 1)
                        
                        pos_loss = MSE(tgt_gt, src)
                        pos_loss[~k_mask] *= 0

                        final_pos_loss = torch.tensor(final_pos_loss).to(pos_loss.device)
                        init_pos_loss = torch.tensor(init_pos_loss).to(pos_loss.device)
                        type_pos_loss = torch.tensor(type_pos_loss).to(pos_loss.device)
                        pos_loss = pos_loss.sum(-1).sum(-1) / (k_mask.sum(-1).sum(-1) + 1e-6)

                        for rel_idx in range(pos_loss.shape[0]):
                            rel_type = traj[rel_idx, 0].cpu().int()
                            pos_loss[rel_idx, rel_type] *= weight_frequency[rel_type]
                            final_pos_loss[rel_idx] *= weight_frequency[rel_type]
                            init_pos_loss[rel_idx] *= weight_frequency[rel_type]

                        pos_loss = torch.gather(pos_loss, dim=1, index=min_index.unsqueeze(-1)).mean()
                        final_pos_loss = final_pos_loss.mean() * 0.05
                        init_pos_loss = init_pos_loss.mean() * 0.01
                        type_pos_loss = type_pos_loss.mean() * 0.01
                        pos_loss *= 0.1
                        cls_loss = CLS(src_prob, min_index) * self.motion_cfg.CLS_WEIGHT
                        cls_loss *= 0.05
                        motion_loss = pos_loss + cls_loss + final_pos_loss + pos_loss # + init_pos_loss
                        motion_attr_loss['motion_pos'].append(pos_loss + cls_loss + final_pos_loss + pos_loss) # + init_pos_loss) 

                        if pred_other_attr:
                            src_heading = motion_attrs['heading']['src'][b_idx]
                            src_vel = motion_attrs['vel']['src'][b_idx]
                            tgt_heading = motion_attrs['heading']['tgt'][b_idx][:, None, :, None].repeat(1, K, 1, 1)
                            tgt_vel = motion_attrs['vel']['tgt'][b_idx][:, None].repeat(1, K, 1, 1)
                            
                            heading_temp = []
                            for j in range(src_heading.shape[0]):
                                heading_temp.append(src_heading[j, vt[j, 0].int()-1, :, :, :])
                            src_heading = torch.stack(heading_temp, dim = 0)
 
                            
                            vel_temp = []
                            for j in range(src_vel.shape[0]):
                                vel_temp.append(src_vel[j, vt[j, 0].int()-1, :, :, :])
                            src_vel = torch.stack(vel_temp, dim = 0)

                            
                            velo_loss = MSE(tgt_vel, src_vel)
                            velo_loss[~k_mask] *= 0
                            velo_loss = velo_loss.sum(-1).sum(-1) / (k_mask.sum(-1).sum(-1) + 1e-6)
                            
                            for rel_idx in range(velo_loss.shape[0]):
                                rel_type = traj[rel_idx, 0].cpu().int()
                                velo_loss[rel_idx, rel_type] *= weight_frequency[rel_type]
                            
                            velo_loss = torch.gather(velo_loss, dim=1, index=min_index.unsqueeze(-1)).mean()
                            velo_loss *= 0.1
                            heading_loss = L1(tgt_heading, src_heading)
                            heading_loss[~k_mask[...,:1]] *= 0
                            heading_loss = heading_loss.sum(-1).sum(-1) / (k_mask[...,:1].sum(-1).sum(-1) + 1e-6)
                            
                            for rel_idx in range(heading_loss.shape[0]):
                                rel_type = traj[rel_idx, 0].cpu().int()
                                heading_loss[rel_idx, rel_type] *= weight_frequency[rel_type]
                            
                            heading_loss = torch.gather(heading_loss, dim=1, index=min_index.unsqueeze(-1)).mean()
                        
                            motion_loss += velo_loss + heading_loss * 10.0

                            motion_attr_loss['motion_vel'].append(velo_loss)
                            motion_attr_loss['motion_heading'].append(heading_loss * 10.0)

                        loss_attr.append(motion_loss)
                    else:
                        loss_attr.append([])
                        motion_attr_loss['motion_pos'].append([])
                        if pred_other_attr:
                            motion_attr_loss['motion_vel'].append([])
                            motion_attr_loss['motion_heading'].append([])
                b_idx += 1

        return loss_attr, motion_attr_loss

    def loss_attributes(self, outputs, data, indices, num_boxes, log=True):
        """Attribute loss
        """ 
        attributes = ['speed', 'pos', 'vel_heading', 'bbox', 'heading']
        targets = data['targets']
        traj_type = data['traj_type']

        if self.with_type_embedding:
            veh_type = data['veh_type']
        
        rel_pos_f = data['nei_pos_f']
        rel_pos_i = data['nei_pos_i']
        gt_type_pos = data['type_pos']
        
        if self.motion_cfg.ENABLE:
            attributes.append('motion')

        MSE = torch.nn.MSELoss(reduction='mean')
        L1 = torch.nn.L1Loss(reduction='mean')
        
        losses = {}
        losses['attributes'] = 0
        
        attr_weights = self.cfg.LOSS.DETR.ATTR_WEIGHT
        self.cal_attr = True

        for attr in attributes:
            
            if self.use_attr_gmm and attr in ['pos', 'bbox', 'heading'] and self.cal_attr:
            # if attr == "pos":
                B, N = outputs['pred_vel_heading'].shape[:2]
                D = 1 if attr == 'heading' else 2
                log_prob_input = torch.zeros(B, N, D).to(outputs['pred_vel_heading'].device)
                for i in range(B):
                    for sidx, tidx in zip(indices[i][0], indices[i][1]):
                        log_prob_input[i, sidx] = targets[i][attr][tidx]
                neg_log_prob = -outputs[f'pred_{attr}'].log_prob(log_prob_input.squeeze())
                gmm_losses = []
                for i in range(B):
                    for sidx in indices[i][0]:
                        gmm_losses.append(neg_log_prob[i, sidx])
                loss_attr = torch.stack(gmm_losses).mean()
            

            else:
                
                src_attrs = [outputs[f'pred_{attr}'][i][indices[i][0]] for i in range(len(indices))]
                try:
                    pos_attrs = [outputs[f'pred_pos'].sample()[i,indices[i][0]] for i in range(len(indices))]
                except:
                    pos_attrs = [outputs[f'pred_pos'][i][indices[i][0]] for i in range(len(indices))]


                target_attrs = [targets[i][attr][indices[i][1]] for i in range(len(indices))]
                gt_type = [traj_type[i][indices[i][1]] for i in range(len(indices))]
                if self.with_type_embedding:
                    gt_veh_type = [veh_type[i][indices[i][1]].int() - 1 for i in range(len(indices))]
                else:
                    gt_veh_type = None
                gt_pos_f = [rel_pos_f[i][indices[i][1]] for i in range(len(indices))]
                gt_pos_i = [rel_pos_i[i][indices[i][0]] for i in range(len(indices))]
                
                if attr == 'motion':
                    masks = [targets[i]['motion_mask'][indices[i][1]] for i in range(len(indices))]
                    if self.motion_cfg.PRED_MODE == 'mlp':
                        src_probs = None
                    elif self.motion_cfg.PRED_MODE in ['mlp_gmm', 'mtf']:
                        src_probs = [outputs['motion_prob'][i][indices[i][0]] for i in range(len(indices))]
 
                    motion_attrs = {}
                    if self.motion_cfg.PRED_HEADING_VEL:
                        for k in ['vel', 'heading']:
                            motion_attrs[k] = {}
                            motion_attrs[k]['src'] = [outputs[f'pred_future_{k}'][i][indices[i][0]] for i in range(len(indices))]
                            motion_attrs[k]['tgt'] = [targets[i][f'future_{k}'][indices[i][1]] for i in range(len(indices))]
 

                if len(target_attrs[0].shape) == 1:
                    target_attrs = [tgt.unsqueeze(1) for tgt in target_attrs]

                if attr in ['vel_heading', 'heading']:
                    loss_func = L1
                else:
                    loss_func = MSE

                if attr == 'motion':
                    loss_attr, loss_motion = self._compute_motion_loss(src_attrs, src_probs, target_attrs, masks, loss_func, motion_attrs, gt_type, gt_pos_f, gt_pos_i, pos_attrs, gt_type_pos, gt_veh_type)
                else:
                    
                    loss_attr = [loss_func(src, tgt) if len(tgt) > 0 else [] for src, tgt in zip(src_attrs, target_attrs)]

                loss_attr = [loss for i, loss in enumerate(loss_attr) if num_boxes[i] > 0]
                
                if len(loss_attr) == 0:
                    loss_attr = torch.tensor(0.0).to(outputs['pred_pos'].device)
                    continue

                loss_attr = torch.stack(loss_attr).mean() 

            # record the original scale of loss_attr for easy comparison
            losses[attr] = loss_attr
            losses['attributes'] += loss_attr * attr_weights[attr]


            if attr == 'motion':
                for motion_attr, motion_loss in loss_motion.items():
                    motion_loss = [loss for i, loss in enumerate(motion_loss) if num_boxes[i] > 0]
                    if len(motion_loss) == 0:
                        motion_loss = torch.tensor(0.0).to(outputs['pred_pos'].device)
                    else:
                        mean_value = torch.stack(motion_loss).mean()
                        losses[motion_attr] = mean_value

        return losses

    def loss_heatmap(self, outputs, data, indices, num_boxes, log=True):
        '''
        debug loss function - supervise only a single query point to the heatmap
        '''
        BCE = torch.nn.BCEWithLogitsLoss(reduction='none')

        # remove the ego vehicle from the gt distribution
        ego_idx = data['agent_vec_index'][:, 0].long()
        B = ego_idx.shape[0]
        data['gt_distribution'][torch.arange(B), ego_idx] = 0
        
        bg_zero = torch.zeros_like(data['gt_distribution'][:, :1])
        gt_distribution = torch.cat([data['gt_distribution'], bg_zero], dim=1)

        src_logits = outputs['pred_logits']
        heatmap = src_logits[:, 0]
        
        prob_loss = BCE(heatmap, gt_distribution)

        line_mask = data['center_mask']
        bg_mask = torch.ones_like(line_mask[:, :1])
        line_mask = torch.cat([line_mask,bg_mask], dim=1)

        prob_loss = torch.sum(prob_loss * line_mask) / max(torch.sum(line_mask), 1)
        losses = {'heatmap': prob_loss}

        return losses

    def loss_vae(self, outputs, data, indices, num_boxes, log=True):
        pm, pv = outputs['prior_output']
        qm, qv = outputs['post_output']

        # KL divergence
        kl_loss = kl_normal(qm, qv, pm, pv).mean()

        prior_std = torch.exp(0.5 * pm).mean()
        post_std = torch.exp(0.5 * qm).mean()

        return {'vae': kl_loss, 'prior_std': prior_std, 'post_std': post_std}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, data, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'heatmap': self.loss_heatmap,
            'attributes': self.loss_attributes,
            'vae': self.loss_vae,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, data, indices, num_boxes, **kwargs)

    def forward(self, outputs, data):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             data['targets']: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        ae_modes = self.cfg.LOSS.DETR.AE_MODES
        if self.cfg.LOSS.DETR.TEXT_AE and 'text' not in ae_modes:
            ae_modes.append('text')
        
        losses = {}
        full_loss = 0

        
        for mode in ae_modes:
            mode_output = outputs['{}_decode_output'.format(mode)]
            indices = self.matcher(mode_output, data['targets'], data['agent_mask'], self.detr_cfg.PRED_BACKGROUND)

            # enforce the sequential matching order between the two sets (disable hungarian matching)
            if self.cfg.LOSS.DETR.MATCH_METHOD == 'sequential':
                for i in range(len(indices)):
                    indices[i] = (indices[i][1], indices[i][1])
            # Compute all the requested losses
            mode_losses = {}

            num_boxes = torch.tensor([len(v['labels']) for v in data['targets']])
            
            for loss in self.losses:
                mode_losses.update(self.get_loss(loss, mode_output, data, indices, num_boxes))
                full_loss += self.weight_dict[loss] * mode_losses[loss]
                for subloss in mode_losses:

                    losses['{}_{}'.format(mode, subloss)] = mode_losses[subloss]
        
        losses['full_loss'] = full_loss

        if self.cfg.LOSS.DETR.ALIGNMENT.ENABLE:
            losses.update(alignment_loss_func(outputs, self.cfg))
            losses['full_loss'] += losses['alignment_loss']

        for loss_type, value in losses.items():
            if value.isnan().any():
                print('nan loss in ', loss_type, value)
                print('nan loss data', data['file'])

        return losses
