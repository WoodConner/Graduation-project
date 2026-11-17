import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from tqdm import tqdm
from fmpose3d.common import Human36mDataset, Fusion
from fmpose3d.models import get_model
from fmpose3d.common import opts

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CHECKPOINT = './checkpoint/261108_2203_17/CFM_10_4811_best.pth'
DATASET_PATH = '../FMPose3D_Improved/dataset/data_3d_h36m.npz'
ROOT_PATH = '../FMPose3D_Improved/dataset/'

print('Setting up args...')
args = opts().parse(['--model', 'fmpose3d'])

print('Loading model...')
Model = get_model('fmpose3d_humans')
model = Model(args).cuda().eval()
ckpt = torch.load(CHECKPOINT, map_location='cuda:0')
model.load_state_dict(ckpt, strict=True)
print('Model loaded!')

print('Loading dataset...')
dataset = Human36mDataset(DATASET_PATH, args)
test_data = Fusion(opt=args, train=False, dataset=dataset, root_path=ROOT_PATH)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=256, shuffle=False,
    num_workers=8, pin_memory=True
)

print(f'Test samples: {len(test_data)}')
print('Evaluating S=3 with 8 hypotheses + FHA...')

with torch.no_grad():
    results = []
    gt_all = []
    
    for batch in tqdm(test_loader):
        batch_cam, gt_3d, input_2d, action, subject, scale, bb_box, cam_ind = batch
        input_2d = input_2d.cuda()
        gt_3d = gt_3d.cuda()
        
        hypotheses = []
        dt = 1.0 / 3
        
        for i in range(8):
            y = torch.randn_like(gt_3d)
            x_t = y
            for s in range(3):
                t_s = torch.full((gt_3d.size(0), 1, 1, 1), s*dt, device=gt_3d.device, dtype=gt_3d.dtype)
                v = model(input_2d, x_t, t_s)
                x_t = x_t + dt * v
            hypotheses.append(x_t[:, 0].unsqueeze(1))
        
        input_2d_flip = input_2d.clone()
        input_2d_flip[:, :, 0] *= -1
        input_2d_flip[:, :, [1,2,3,9,10,11]] = input_2d_flip[:, :, [2,1,3,10,9,11]]
        
        for i in range(8):
            y = torch.randn_like(gt_3d)
            x_t = y
            for s in range(3):
                t_s = torch.full((gt_3d.size(0), 1, 1, 1), s*dt, device=gt_3d.device, dtype=gt_3d.dtype)
                v = model(input_2d_flip, x_t, t_s)
                x_t = x_t + dt * v
            x_t[:, :, :, 0] *= -1
            x_t[:, :, [1,2,3,9,10,11]] = x_t[:, :, [2,1,3,10,9,11]]
            hypotheses.append(x_t[:, 0].unsqueeze(1))
        
        results.append(torch.cat(hypotheses, dim=1).cpu())
        gt_all.append(gt_3d.cpu())

all_hyp = torch.cat(results, dim=0)
all_gt = torch.cat(gt_all, dim=0)

print(f'Hypotheses shape: {all_hyp.shape}')
final = all_hyp.mean(dim=1)
mpjpe = torch.mean(torch.norm(final - all_gt, dim=-1)).item()

print(f'\nPathA E10 S=3 Results:')
print(f'MPJPE (P1): {mpjpe:.4f} mm')
