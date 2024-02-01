import os
import cv2
import time
import numpy as np
import argparse
from typing import Tuple
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP

from .classifier import Classifier
from .model.pspnet import get_model
from .utils import (fast_intersection_and_union, setup_seed, ensure_dir, 
                  resume_random_state, find_free_port, setup, cleanup, get_cfg)

from .dataset.data import get_val_loader
from .dataset.classes import classId2className, update_novel_classes


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    return get_cfg(parser)


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    print(f"==> Running evaluation script")
    setup(args, rank, world_size)
    setup_seed(args.manual_seed)

    # ========== Data  ==========
    val_loader = get_val_loader(args)

    # ========== Model  ==========    
    print("=> Creating the model")
    model = get_model(args).to(rank) 
    
    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained), args.pretrained
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint,  strict=False) 
        print("=> Loaded weight '{}'".format(args.pretrained))
    else:
        print("=> Not loading anything")

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])
    
    # ========== Test  ==========
    validate(args=args, val_loader=val_loader, model=model)
    cleanup()


def validate(args: argparse.Namespace, val_loader: torch.utils.data.DataLoader, model: DDP) -> Tuple[torch.tensor, torch.tensor]:
    print('\n==> Start testing...', flush=True)
    base_novel_classes = classId2className
    random_state = setup_seed(args.manual_seed, return_old_state=True)
    device = torch.device('cuda:{}'.format(dist.get_rank()))
    model.eval()

    c = model.module.bottleneck_dim
    h = int(args.image_size / 8)
    w = int(args.image_size / 8)
        
    # ========== Perform the runs  ==========
    # The order of classes in the following tensors is the same as the order of classifier (novels at last)
    cls_intersection = torch.zeros(args.num_classes_tr + args.num_classes_val)
    cls_union = torch.zeros(args.num_classes_tr + args.num_classes_val)
    cls_target = torch.zeros(args.num_classes_tr + args.num_classes_val)

    runtime = 0
    features_s, gt_s = None, None
    with torch.no_grad():
        spprt_imgs, s_label = val_loader.dataset.get_support_set() # Get the support set 
        spprt_imgs = spprt_imgs.to(device, non_blocking=True)
        s_label = s_label.to(device, non_blocking=True)
        features_s = model.module.extract_features(spprt_imgs).detach().view((args.num_classes_val, args.shot, c, h, w)) 
        gt_s = s_label.view((args.num_classes_val, args.shot, args.image_size, args.image_size))

    nb_episodes = len(val_loader) # The number of images in the query set
    for _ in tqdm(range(nb_episodes), leave=True):
        t0 = time.time()
        with torch.no_grad():
            try:
                loader_output = next(iter_loader)
            except (UnboundLocalError, StopIteration):
                iter_loader = iter(val_loader)
                loader_output = next(iter_loader)
            qry_img, q_label, q_valid_pix, image_name = loader_output

            qry_img = qry_img.to(device, non_blocking=True)
            q_label = q_label.to(device, non_blocking=True)
            features_q = model.module.extract_features(qry_img).detach().unsqueeze(1) 
            valid_pixels_q = q_valid_pix.unsqueeze(1).to(device)
            gt_q = q_label.unsqueeze(1)

        # =========== Initialize the classifier and run the method ===============
        base_weight = model.module.classifier.weight.detach().clone() 
        base_weight = base_weight.permute(*torch.arange(base_weight.ndim - 1, -1, -1))
        base_bias = model.module.classifier.bias.detach().clone()
                    
        classifier = Classifier(args, base_weight, base_bias, n_tasks=features_q.size(0))
        classifier.init_prototypes(features_s, gt_s)
        classifier.compute_pi(features_q, valid_pixels_q) 
        classifier.optimize(features_s, features_q, gt_s, valid_pixels_q)

        runtime += time.time() - t0

        # =========== Perform inference and compute metrics ===============
        logits = classifier.get_logits(features_q).detach()
        probas = classifier.get_probas(logits)

        if args.save_pred_maps is True:    # Save predictions in '.png' file and submit for evaluation
            ensure_dir('results/targets')
            ensure_dir('results/preds')
            n_task, shots, num_classes, h, w = probas.size()
            H, W = gt_q.size()[-2:]
            if (h, w) != (H, W):
                probas = F.interpolate(probas.view(n_task * shots, num_classes, h, w),
                                    size=(H, W), mode='bilinear', align_corners=True).view(n_task, shots, num_classes, H, W)
            pred = probas.argmax(2)  # [n_query, shot, H, W]
            pred = np.array(pred.squeeze().cpu(), np.uint8)
            target = np.array(gt_q.squeeze().cpu(), np.uint8)
            fname = ''.join(image_name)
            cv2.imwrite(os.path.join('results/targets', fname + '.png'), target)
            cv2.imwrite(os.path.join('results/preds', fname + '.png'), pred)

        intersection, union, target = fast_intersection_and_union(probas, gt_q)  # [batch_size_val, 1, num_classes]
        intersection, union, target = intersection.squeeze(1).cpu(), union.squeeze(1).cpu(), target.squeeze(1).cpu()
        cls_intersection += intersection.sum(0)
        cls_union += union.sum(0)
        cls_target += target.sum(0)

    base_count, novel_count, sum_base_IoU, sum_novel_IoU = 4 * [0]
    results = []
    results.append('\nClass IoU Results')
    results.append('---------------------------------------')
    
    if args.novel_classes is not None:  # Update novel classnames
       update_novel_classes(base_novel_classes, args.novel_classes)
  
    for i, class_ in enumerate(val_loader.dataset.all_classes):
        if class_ == 0:
            continue
        
        IoU = cls_intersection[i] / (cls_union[i] + 1e-10)
        classname = base_novel_classes[class_].capitalize()
        if classname == '':
            classname = 'Novel class'
        results.append(f'%d %-25s \t %.2f' %(i, classname, IoU * 100))
        
        if class_ in val_loader.dataset.base_class_list:
            sum_base_IoU += IoU
            base_count += 1
        elif class_ in val_loader.dataset.novel_class_list:
            sum_novel_IoU += IoU
            novel_count += 1
            
    avg_base_IoU, avg_novel_IoU = sum_base_IoU / base_count, sum_novel_IoU / novel_count
    agg_mIoU = (avg_base_IoU + avg_novel_IoU) / 2
    wght_avg_base_IoU, wght_avg_novel_IoU = avg_base_IoU * 0.6, avg_novel_IoU * 0.4
    wght_agg_mIoU = (wght_avg_base_IoU + wght_avg_novel_IoU) / 2
    
    results.append('---------------------------------------')
    results.append(f'\n%-30s \t %.2f' %('Average of base IoU', avg_base_IoU * 100))
    results.append(f'%-30s \t %.2f' %('Average of novel IoU', avg_novel_IoU * 100))
    results.append(f'%-30s \t %.2f' %('Overall mean IoU', agg_mIoU * 100))
    results.append(f'\n%-30s \t %.2f' %('Weighted average of base IoU', wght_avg_base_IoU * 100))
    results.append(f'%-30s \t %.2f' %('Weighted average of novel IoU', wght_avg_novel_IoU * 100))
    results.append(f'%-30s \t %.2f' %('Weighted overall mean IoU', wght_agg_mIoU * 100))
    results.append(f'The weighted average is calculated using `0.6:0.4 => base:novel` based on SOA GFSS baseline.')
    iou_results = "\n".join(results)
    print(iou_results)
    
    if args.save_ious is True:  # Save class IoUs
        ensure_dir('results')
        with open(os.path.join('results', 'base_novel_ious.txt'), 'w') as f:
            f.write(iou_results)
    
    print('\n===> Runtime --- {:.1f}\n'.format(runtime))
    
    resume_random_state(random_state)
    return agg_mIoU


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    world_size = len(args.gpus)
    distributed = world_size > 1
    assert not distributed, 'Testing should not be done in a distributed way'
    args.distributed = distributed
    args.port = find_free_port()
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
