"""L3 Training: G1/G2/G3 comparison"""
import argparse,os,sys,logging,json
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
from L3.dataset import H36MActionDataset,H36M_ACTIONS,H36M_TRAIN_SUBJECTS,H36M_TEST_SUBJECTS,create_vote_windows,center_root
from L3.stgcn import STGCN
logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
def load_h36m_sequences(npz_path,subjects):
    data=np.load(npz_path,allow_pickle=True)
    positions=data['positions_3d'].item()
    # FMPose3D标准17关节映射（与baseline一致）
    GT_32_TO_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    seqs,labels=[],[]
    for subj in subjects:
        key=f'S{subj}'
        if key not in positions:continue
        for action_name in positions[key]:
            base=action_name.split(' ')[0]
            if base not in H36M_ACTIONS:continue
            labels.append(H36M_ACTIONS.index(base))
            seq_32 = positions[key][action_name].astype(np.float32)  # (T, 32, 3)
            seq_17 = seq_32[:, GT_32_TO_17, :]  # 正确映射到17关节
            seqs.append(seq_17)
    logger.info(f'  Loaded {len(seqs)} sequences from {subjects}')
    return seqs,labels
def train_one_epoch(model,loader,optimizer,device):
    model.train()
    total_loss,correct,total=0.,0,0
    for poses,lbls in loader:
        poses,lbls=poses.to(device),lbls.to(device)
        optimizer.zero_grad()
        logits=model(poses)
        loss=F.cross_entropy(logits,lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        total_loss+=loss.item()*lbls.size(0)
        correct+=(logits.argmax(1)==lbls).sum().item()
        total+=lbls.size(0)
    return total_loss/total,correct/total
@torch.no_grad()
def evaluate(model,loader,device):
    model.eval()
    conf=np.zeros((15,15),dtype=int)
    correct,total=0,0
    for poses,lbls in loader:
        poses,lbls=poses.to(device),lbls.to(device)
        preds=model(poses).argmax(1)
        for p,g in zip(preds.cpu().numpy(),lbls.cpu().numpy()):
            conf[g,p]+=1
        correct+=(preds==lbls).sum().item()
        total+=lbls.size(0)
    return correct/total,conf

@torch.no_grad()
def evaluate_vote(model,val_seqs,val_lbls,mean,std,device,window_len=120,n_windows=3):
    """3-window majority vote evaluation for stable accuracy estimate."""
    model.eval()
    conf=np.zeros((15,15),dtype=int)
    correct=0
    vote_data=create_vote_windows(val_seqs,val_lbls,window_len,n_windows)
    for windows,label in vote_data:
        votes=[]
        for w in windows:
            # Preprocess: center root + zscore
            w=w.copy().astype(np.float32)
            w=w-w[:,0:1,:]
            w=(w-mean)/std
            t=torch.from_numpy(w).unsqueeze(0).to(device)
            pred=model(t).argmax(1).item()
            votes.append(pred)
        # Majority vote
        final_pred=max(set(votes),key=votes.count)
        conf[label,final_pred]+=1
        if final_pred==label:
            correct+=1
    return correct/len(vote_data),conf
def train_group(name,train_seqs,train_lbls,val_seqs,val_lbls,seeds,epochs,batch_size,lr,device,save_dir,no_aug=False,dropout=0.5,weight_decay=1e-4,early_stop_patience=15,aug_rotate=15,aug_scale=0.05):
    os.makedirs(save_dir,exist_ok=True)
    all_results=[]
    for seed in range(seeds):
        logger.info(f'[{name}] Seed {seed+1}/{seeds}')
        torch.manual_seed(seed);np.random.seed(seed)
        use_aug = not no_aug
        train_ds=H36MActionDataset(train_seqs,train_lbls,window_len=120,is_train=True,
                                   use_augmentation=use_aug,aug_rotate=aug_rotate,aug_scale=aug_scale)
        val_ds=H36MActionDataset(val_seqs,val_lbls,mean=train_ds.mean,std=train_ds.std,window_len=120,is_train=False)
        logger.info(f'  Train windows: {len(train_ds)}, Val windows: {len(val_ds)} (aug={use_aug})')
        train_ldr=DataLoader(train_ds,batch_size,shuffle=True,num_workers=2,pin_memory=True)
        model=STGCN(num_classes=15,dropout=dropout).to(device)
        logger.info(f'  Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M, dropout={dropout}')
        optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=epochs,eta_min=lr*0.01)
        best_acc,best_conf,best_epoch=0.,None,0
        no_improve_count=0
        history=[]
        for epoch in range(1,epochs+1):
            tr_loss,tr_acc=train_one_epoch(model,train_ldr,optim,device)
            val_acc,conf=evaluate_vote(model,val_seqs,val_lbls,train_ds.mean,train_ds.std,device)
            sched.step()
            history.append({'epoch':epoch,'tr_loss':round(tr_loss,4),'tr_acc':round(tr_acc,4),'val_acc':round(val_acc,4)})
            if val_acc>best_acc:
                best_acc=val_acc;best_conf=conf.copy();best_epoch=epoch
                no_improve_count=0
                torch.save(model.state_dict(),f'{save_dir}/{name}_s{seed}_best.pth')
            else:
                no_improve_count+=1
            if epoch%10==0 or epoch==epochs:
                logger.info(f'  E{epoch:3d}/{epochs} loss={tr_loss:.4f} tr={tr_acc:.4f} val={val_acc:.4f} best={best_acc:.4f} (e{best_epoch}, no_improve={no_improve_count})')
            # Early stop
            if no_improve_count>=early_stop_patience:
                logger.info(f'  Early stop at epoch {epoch} (no improvement for {early_stop_patience} epochs)')
                break
        json.dump(history,open(f'{save_dir}/{name}_s{seed}_hist.json','w'),indent=2)
        np.save(f'{save_dir}/{name}_s{seed}_conf.npy',best_conf)
        logger.info(f'[{name}] Seed {seed} = {best_acc:.4f}')
        all_results.append({'best_acc':best_acc,'conf_matrix':best_conf})
    return all_results
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gt_poses_path',default='dataset/data_3d_h36m.npz')
    p.add_argument('--l2_poses_path',default=None)
    p.add_argument('--group',default='all',choices=['G1','G2','G3','all'])
    p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--batch_size',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-3)  # raised from 0.01 to 1e-3 (AdamW)
    p.add_argument('--save_dir',default='L3/results')
    p.add_argument('--device',default='cuda')
    p.add_argument('--no_aug',action='store_true',help='Disable data augmentation for diagnosis')
    p.add_argument('--dropout',type=float,default=0.5,help='Dropout rate (default 0.5)')
    p.add_argument('--weight_decay',type=float,default=1e-4,help='Weight decay (default 1e-4)')
    p.add_argument('--early_stop_patience',type=int,default=15,help='Early stop patience (default 15)')
    p.add_argument('--aug_rotate',type=float,default=15,help='Rotation augmentation +/- degrees')
    p.add_argument('--aug_scale',type=float,default=0.05,help='Scale augmentation range (0.95-1.05 if scale=0.05)')
    args=p.parse_args()
    device=args.device if torch.cuda.is_available() else 'cpu'
    logger.info('Loading GT 3D...')
    gt_tr_seqs,gt_tr_lbls=load_h36m_sequences(args.gt_poses_path,H36M_TRAIN_SUBJECTS)
    gt_va_seqs,gt_va_lbls=load_h36m_sequences(args.gt_poses_path,H36M_TEST_SUBJECTS)
    if args.l2_poses_path and os.path.exists(args.l2_poses_path):
        d=np.load(args.l2_poses_path,allow_pickle=True)
        l2_tr_seqs_raw=list(d['train_seqs']);l2_tr_lbls=list(d['train_labels'])
        l2_va_seqs_raw=list(d['val_seqs']);l2_va_lbls=list(d['val_labels'])
        # L2序列未采样，需要用同一个create_sliding_windows（从dataset.py导入）
        from L3.dataset import create_sliding_windows
        l2_tr_seqs=[w for seq,lbl in zip(l2_tr_seqs_raw,l2_tr_lbls) for w in create_sliding_windows([seq],[lbl])[0]]
        l2_tr_lbls=[lbl for seq,lbl in zip(l2_tr_seqs_raw,l2_tr_lbls) for _ in create_sliding_windows([seq],[lbl])[0]]
        l2_va_seqs=[w for seq,lbl in zip(l2_va_seqs_raw,l2_va_lbls) for w in create_sliding_windows([seq],[lbl])[0]]
        l2_va_lbls=[lbl for seq,lbl in zip(l2_va_seqs_raw,l2_va_lbls) for _ in create_sliding_windows([seq],[lbl])[0]]
        # 检查std
        l2_std=float(np.mean([s.std() for s in l2_tr_seqs]))
        gt_std=float(np.mean([s.std() for s in gt_tr_seqs]))
        ratio=l2_std/gt_std
        logger.info(f'[check] L2 std={l2_std:.4f}, GT std={gt_std:.4f}, ratio={ratio:.3f}')
        if ratio>1.5 or ratio<0.67:
            scale=gt_std/l2_std
            logger.info(f'[check] std ratio off, scaling L2 by {scale:.3f}')
            l2_tr_seqs=[s*scale for s in l2_tr_seqs]
            l2_va_seqs=[s*scale for s in l2_va_seqs]
    else:
        logger.warning('L2 placeholder: using GT')
        l2_tr_seqs,l2_tr_lbls=gt_tr_seqs,gt_tr_lbls
        l2_va_seqs,l2_va_lbls=gt_va_seqs,gt_va_lbls
    n=min(len(gt_tr_seqs),len(l2_tr_seqs))
    g3_tr_seqs=gt_tr_seqs[:n//2]+l2_tr_seqs[:n//2]
    g3_tr_lbls=gt_tr_lbls[:n//2]+l2_tr_lbls[:n//2]
    kw=dict(seeds=args.seeds,epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,device=device,save_dir=args.save_dir,no_aug=args.no_aug,dropout=args.dropout,weight_decay=args.weight_decay,early_stop_patience=args.early_stop_patience,aug_rotate=args.aug_rotate,aug_scale=args.aug_scale)
    results={}
    if args.group in ('G1','all'):
        results['G1']=train_group('G1',gt_tr_seqs,gt_tr_lbls,gt_va_seqs,gt_va_lbls,**kw)
        a=[r['best_acc'] for r in results['G1']]
        logger.info(f'G1: {np.mean(a):.4f}+/-{np.std(a):.4f}')
    if args.group in ('G2','all'):
        results['G2']=train_group('G2',l2_tr_seqs,l2_tr_lbls,l2_va_seqs,l2_va_lbls,**kw)
        a=[r['best_acc'] for r in results['G2']]
        logger.info(f'G2: {np.mean(a):.4f}+/-{np.std(a):.4f}')
    if args.group in ('G3','all'):
        results['G3']=train_group('G3',g3_tr_seqs,g3_tr_lbls,l2_va_seqs,l2_va_lbls,**kw)
        a=[r['best_acc'] for r in results['G3']]
        logger.info(f'G3: {np.mean(a):.4f}+/-{np.std(a):.4f}')
    if 'G1' in results and 'G2' in results:
        a1=np.mean([r['best_acc'] for r in results['G1']])
        a2=np.mean([r['best_acc'] for r in results['G2']])
        print('Results: G1={:.4f} G2={:.4f} Delta={:.2f}%'.format(a1,a2,(a1-a2)*100))
    summary={k:[{'best_acc':r['best_acc']} for r in v] for k,v in results.items()}
    os.makedirs(args.save_dir,exist_ok=True)
    json.dump(summary,open(f'{args.save_dir}/summary.json','w'),indent=2)
    logger.info(f'Results: {args.save_dir}/')
if __name__=='__main__':
    main()
