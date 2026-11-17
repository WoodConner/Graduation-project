import sys
sys.path.insert(0, '.')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from fmpose3d.common import opts, Human36mDataset, Fusion
from fmpose3d.common.utils import define_actions
from fmpose3d.common.utils import get_variable
from fmpose3d.models import get_model
from fmpose3d.aggregation_methods import aggregation_RPEA_joint_level
from tqdm import tqdm

args = opts().parse()
args.test_augmentation = 1
args.test_augmentation_flip_hypothesis = 1
args.hypothesis_num = 8
args.topk = 8
args.exp_temp = 0.002

root_path = '../FMPose3D_Improved/dataset/'
dataset_path = root_path + 'data_3d_h36m.npz'

print('Loading dataset...')
dataset = Human36mDataset(dataset_path, args)
test_data = Fusion(opt=args, train=False, dataset=dataset, root_path=root_path)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=256, shuffle=False,
    num_workers=8, pin_memory=True
)


print('Loading model...')
Model = get_model('fmpose3d_humans')
model = {'CFM': Model(args).cuda().eval()}
ckpt = torch.load('./checkpoint/261108_2203_17/CFM_10_4811_best.pth', map_location='cuda:0')
model['CFM'].load_state_dict(ckpt, strict=True)
print('Model loaded!')

print('Testing S=3 with 8 hypotheses...')
p1_list = []
p2_list = []

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader)):
        batch_cam
        [input_2d, gt_3d, batch_cam] = get_variable('test', [input_2d, gt_3d, batch_cam])
        
        if args.test_augmentation:
            input_2d_nonflip = input_2d[:, 0]
            input_2d_flip = input_2d[:, 1]
        else:
            input_2d_nonflip = input_2d
            input_2d_flip = None
        
        out_target = gt_3d.clone()
        out_target[:, :, 0] = 0
        
        def euler_sample(x2d, y_local, steps):
            dt = 1.0 / steps
            for s in range(steps):
                t_s = torch.full((gt_3d.size(0), 1, 1, 1), s*dt, device=gt_3d.device, dtype=gt_3d.dtype)
                v_s = model['CFM'](x2d, y_local, t_s)
                y_local = y_local + dt * v_s
            return y_local
        
        list_hypothesis = []
        for h in range(8):
            y = torch.randn_like(gt_3d)
            y_s = euler_sample(input_2d_nonflip, y, 3)
            
            if args.test_augmentation:
                y_flip = torch.randn_like(gt_3d)
                y_flip_s = euler_sample(input_2d_flip, y_flip, 3)
                y_flip_s[:, :, :, 0] *= -1
                y_flip_s[:, :, args.joints_left + args.joints_right, :] = y_flip_s[:, :, args.joints_right + args.joints_left, :]
                y_flip_s = y_flip_s[:, args.pad].unsqueeze(1)
                y_flip_s[:, :, 0, :] = 0
                list_hypothesis.append(y_flip_s)
            
            output_3d_s = y_s[:, args.pad].unsqueeze(1)
            output_3d_s[:, :, 0, :] = 0
            list_hypothesis.append(output_3d_s)
        
        output_3d = aggregation_RPEA_joint_level(args, list_hypothesis, batch_cam, input_2d_nonflip, gt_3d, args.topk)
        
        p1 = torch.norm(output_3d - out_target, dim=-1).mean().cpu().item()
        p1_list.append(p1)

final_p1 = sum(p1_list) / len(p1_list)
print(f'\nPathA E10 S=3 Final Result:')
print(f'P1 (MPJPE): {final_p1:.4f} mm')
