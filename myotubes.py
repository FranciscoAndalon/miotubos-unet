"""Small, explicit binary segmentation pipeline. Original data are read-only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Cannot read {path}')
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f'Expected a 2D image: {path}')
    return image


def normalize(image):
    # Per-image percentiles use no statistics from other images or partitions.
    lo, hi = np.percentile(image, (1, 99))
    return np.clip((image.astype(np.float32) - lo) / max(float(hi-lo), 1), 0, 1)


def prepare(args):
    data, out = Path(args.data), Path(args.out)
    files = sorted((data / 'images').glob('*.png'))
    if not files:
        raise ValueError('No PNG images found')
    rows, seen, mask_hashes, mask_conflicts = [], {}, {}, set()
    coco = json.loads((data/'coco/annotations.json').read_text())
    coco_images = {i['file_name']: i['id'] for i in coco['images']}
    coco_annotations = {}
    for a in coco['annotations']:
        coco_annotations.setdefault(a['image_id'], []).append(a)
    parent = {}

    def find(g):
        parent.setdefault(g, g)
        if parent[g] != g:
            parent[g] = find(parent[g])
        return parent[g]

    for f in files:
        mask = gray(data / 'masks_png' / f.name)
        image = gray(f)
        if mask.shape != image.shape or not set(np.unique(mask)).issubset({0, 1, 255}):
            raise ValueError(f'Invalid mask for {f.name}')
        label = data / 'labels' / (f.stem + '.txt')
        lines = [s.split() for s in label.read_text().splitlines() if s.strip()]
        reconstructed = np.zeros_like(mask, dtype=np.uint8)
        for line in lines:
            coords = np.array(line[1:], dtype=float).reshape(-1, 2)
            if line[0] != '0' or not np.isfinite(coords).all():
                raise ValueError(f'Invalid annotation in {label}')
            pts = (coords * [image.shape[1], image.shape[0]]).astype(np.int32)
            cv2.fillPoly(reconstructed, [pts], 1)
        # Names contain an export prefix, then a source UUID. All cellN images
        # are kept together because their biological provenance is unknown.
        source = f.stem.split('_')[0][9:]
        group = 'cell_unknown_origin' if source.startswith('cell') else source
        find(group)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        mask_digest = hashlib.sha256((mask>0).tobytes()).hexdigest()
        if digest in mask_hashes and mask_hashes[digest] != mask_digest:
            mask_conflicts.add(digest)
        mask_hashes[digest] = mask_digest
        duplicate = seen.get(digest)
        if duplicate:
            parent[find(group)] = find(duplicate)
        seen[digest] = group
        exported = np.zeros_like(mask, dtype=np.uint8)
        annotations = coco_annotations.get(coco_images[f.name], [])
        for annotation in annotations:
            for polygon in annotation['segmentation']:
                cv2.fillPoly(exported, [np.array(polygon).reshape(-1,2).astype(np.int32)], 1)
        rows.append(dict(file=f.name, group=group, sha256=digest,
                         height=image.shape[0], width=image.shape[1],
                         dtype=str(image.dtype), minimum=int(image.min()), maximum=int(image.max()),
                         foreground_fraction=float((mask > 0).mean()), instances=len(lines),
                         exported_instances=len(annotations),
                         coco_mask_disagreement=float(((mask > 0) != exported).mean()),
                         polygon_mask_disagreement=float(((mask > 0) != reconstructed).mean())))
    for row in rows:
        row['group'] = find(row['group'])
    groups = sorted({r['group'] for r in rows})
    counts = {g: sum(r['group'] == g for r in rows) for g in groups}
    rng = np.random.default_rng(args.seed)
    best = None
    # Match sizes using group counts only, never masks or measured performance.
    for _ in range(3000):
        shuffled = list(rng.permutation(groups))
        n = len(shuffled)
        a, b = max(1, round(n*.7)), max(1, round(n*.15))
        splits = dict(train=shuffled[:a], val=shuffled[a:a+b], test=shuffled[a+b:])
        score = sum((sum(counts[g] for g in splits[k])/len(rows)-v)**2
                    for k,v in [('train', .7), ('val', .15), ('test', .15)])
        if best is None or score < best[0]:
            best = (score, splits)
    assignment = {g: k for k, gs in best[1].items() for g in gs}
    for r in rows:
        r['split'] = assignment[r['group']]
    out.mkdir(parents=True, exist_ok=True)
    with (out/'manifest.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for split in ['train', 'val', 'test']:
        (out/f'{split}.txt').write_text(''.join(r['file']+'\n' for r in rows if r['split']==split))
    orphan = sorted(f.name for f in (data/'labels').glob('*.txt') if not (data/'images'/(f.stem+'.png')).exists())
    audit = dict(seed=args.seed, images=len(rows), groups=len(groups), empty_masks=sum(r['instances']==0 for r in rows),
                 foreground_fraction_mean=float(np.mean([r['foreground_fraction'] for r in rows])),
                 duplicate_images=len(rows)-len(seen), orphan_labels=orphan,
                 duplicate_groups_with_different_masks=[[r['file'] for r in rows if r['sha256']==d] for d in sorted(mask_conflicts)],
                 masks_above_20_percent=[r['file'] for r in rows if r['foreground_fraction']>.2],
                 max_polygon_mask_disagreement=max(r['polygon_mask_disagreement'] for r in rows),
                 max_coco_mask_disagreement=max(r['coco_mask_disagreement'] for r in rows),
                 images_with_exported_instance_count_difference=[r['file'] for r in rows if r['instances']!=r['exported_instances']],
                 split_counts={k:sum(r['split']==k for r in rows) for k in assignment.values()},
                 group_policy='Source UUID; all cellN together; exact decoded-image duplicates merged. Biological provenance unverified.')
    save_json(out/'audit.json', audit)
    print(json.dumps(audit, indent=2))


def read_split(out, split):
    return (Path(out)/f'{split}.txt').read_text().splitlines()


def cached_pairs(data, out, split, width, height):
    names = read_split(out, split)
    # Fingerprint source metadata and split; invalidate after any source edit.
    signature = []
    for name in names:
        for folder in ['images', 'masks_png']:
            p = Path(data)/folder/name
            s = p.stat(); signature.append((str(p.resolve()), s.st_size, s.st_mtime_ns))
    key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:16]
    cache = Path(out)/'cache'/f'{split}_{width}x{height}_{key}.npz'
    if cache.exists():
        with np.load(cache) as z:
            return names, z['x'], z['y']
    xs, ys = [], []
    for name in names:
        x = normalize(gray(Path(data)/'images'/name))
        xs.append(np.round(cv2.resize(x, (width,height), interpolation=cv2.INTER_AREA)*255).astype(np.uint8))
        y = gray(Path(data)/'masks_png'/name)>0
        ys.append(cv2.resize(y.astype(np.uint8), (width,height), interpolation=cv2.INTER_NEAREST))
    x, y = np.stack(xs), np.stack(ys)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, y=y)
    return names, x, y


def confusion(pred, target):
    pred, target = pred.astype(bool), target.astype(bool)
    return np.array([np.sum(pred & target), np.sum(pred & ~target), np.sum(~pred & target)], dtype=np.int64)


def scores(counts):
    tp, fp, fn = map(int, counts)
    return dict(iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                precision=tp/(tp+fp) if tp+fp else (1. if tp+fn==0 else 0.),
                recall=tp/(tp+fn) if tp+fn else 1.)


def summarize(counts):
    positive = [c for c in counts if c[0]+c[2]>0]
    return dict(micro=scores(np.sum(counts, axis=0)),
                macro={k:float(np.mean([scores(c)[k] for c in counts])) for k in scores(counts[0])},
                macro_nonempty={k:float(np.mean([scores(c)[k] for c in positive])) if positive else None for k in scores(counts[0])},
                empty_target_images=len(counts)-len(positive),
                empty_target_false_positive_images=int(sum(c[1]>0 for c in counts if c[0]+c[2]==0)))


def classical(image, config):
    x = cv2.GaussianBlur(image, (3,3), 0)
    local = cv2.GaussianBlur(x, (0,0), 9)
    mask = ((local.astype(np.float32)-x) > config['delta']).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA] < config['min_area']:
            continue
        pts = np.column_stack(np.where(labels==i))[:,::-1].astype(np.float32)
        (_, _), (w,h), _ = cv2.minAreaRect(pts)
        if max(w,h)/max(min(w,h),1) >= config['min_aspect']:
            keep[labels==i] = 1
    return keep


def fit_classical(args):
    _, x, y = cached_pairs(args.data,args.out,'val',args.width,args.height)
    best = None
    for delta, area, aspect in itertools.product([4,8,12], [12,40], [1.,2.]):
        config = dict(delta=delta, min_area=area, min_aspect=aspect)
        metrics = summarize([confusion(classical(a,config),b) for a,b in zip(x,y)])
        score = metrics['macro_nonempty']['dice']
        if best is None or score>best['val']['macro_nonempty']['dice']:
            best = dict(config=config, val=metrics, width=args.width, height=args.height,
                        selection='maximum macro Dice over nonempty validation targets', candidates=12)
    save_json(Path(args.out)/'classical.json',best)
    print(json.dumps(best,indent=2))


class Block(nn.Sequential):
    def __init__(self, a,b):
        super().__init__(nn.Conv2d(a,b,3,padding=1), nn.GroupNorm(4,b), nn.ReLU(),
                         nn.Conv2d(b,b,3,padding=1), nn.GroupNorm(4,b), nn.ReLU())


class SmallUNet(nn.Module):
    def __init__(self, base=12):
        super().__init__()
        self.enc = nn.ModuleList([Block(1,base),Block(base,base*2),Block(base*2,base*4)])
        self.bottom = Block(base*4,base*8)
        self.dec = nn.ModuleList([Block(base*12,base*4),Block(base*6,base*2),Block(base*3,base)])
        self.head = nn.Conv2d(base,1,1)

    def forward(self,x):
        skips = []
        for layer in self.enc:
            x = layer(x); skips.append(x); x = nn.functional.max_pool2d(x,2)
        x = self.bottom(x)
        for layer,skip in zip(self.dec,reversed(skips)):
            x = nn.functional.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x = layer(torch.cat([x,skip],1))
        return self.head(x)


class Pairs(Dataset):
    def __init__(self,x,y,augment=False):
        self.x,self.y,self.augment=x,y,augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self,i):
        x,y = self.x[i].astype(np.float32)/255, self.y[i].astype(np.float32)
        if self.augment:
            if random.random()<.5: x,y=np.fliplr(x),np.fliplr(y)
            if random.random()<.5: x,y=np.flipud(x),np.flipud(y)
            x=np.clip(x*random.uniform(.9,1.1)+random.uniform(-.04,.04),0,1)
        return torch.from_numpy(x.copy()[None]),torch.from_numpy(y.copy()[None])


def get_device(request):
    if request!='auto': return torch.device(request)
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


@torch.no_grad()
def probabilities(model, x, device, batch=4):
    model.eval()
    result=[]
    for i in range(0,len(x),batch):
        tensor=torch.from_numpy(x[i:i+batch,None].astype(np.float32)/255).to(device)
        result.append(model(tensor).sigmoid().cpu().numpy()[:,0])
    return np.concatenate(result)


def threshold_search(probs,y):
    candidates=[]
    for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
        metrics=summarize([confusion(p>=t,b) for p,b in zip(probs,y)])
        candidates.append(dict(threshold=t,metrics=metrics))
    return max(candidates,key=lambda c:c['metrics']['macro_nonempty']['dice'])


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device=get_device(args.device)
    _,x,y=cached_pairs(args.data,args.out,'train',args.width,args.height)
    _,vx,vy=cached_pairs(args.data,args.out,'val',args.width,args.height)
    loader=DataLoader(Pairs(x,y,True),batch_size=args.batch,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(args.seed))
    model=SmallUNet().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=4)
    weight=min(10.,float(np.sqrt((y.size-y.sum())/max(y.sum(),1))))
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight],device=device))
    config={k:v for k,v in vars(args).items() if k!='func'}
    config.update(device=str(device),parameters=sum(p.numel() for p in model.parameters()),
                  positive_weight=weight,loss='0.5 weighted BCE + 0.5 per-image soft Dice',
                  selection='maximum macro Dice over nonempty validation targets; threshold grid 0.1:0.1:0.9',
                  normalization='per-image 1st/99th percentile; clip to [0,1]',
                  augmentation='horizontal/vertical flips, gain 0.9:1.1, offset -0.04:0.04',
                  torch_version=str(torch.__version__))
    save_json(Path(args.out)/'training_config.json',config)
    history=[]; best=-1; stale=0; start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train(); total=0
        for a,b in loader:
            a,b=a.to(device),b.to(device)
            optimizer.zero_grad(set_to_none=True); logits=model(a); p=logits.sigmoid()
            dice=(2*(p*b).sum((1,2,3))+1)/(p.sum((1,2,3))+b.sum((1,2,3))+1)
            loss=.5*bce(logits,b)+.5*(1-dice.mean())
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); optimizer.step(); total+=loss.item()*len(a)
        chosen=threshold_search(probabilities(model,vx,device,args.batch),vy)
        score=chosen['metrics']['macro_nonempty']['dice']; scheduler.step(score)
        row=dict(epoch=epoch,loss=total/len(x),val_dice_macro_nonempty=score,
                 val_dice_macro=chosen['metrics']['macro']['dice'],
                 val_dice_micro=chosen['metrics']['micro']['dice'],threshold=chosen['threshold'],
                 lr=optimizer.param_groups[0]['lr'],elapsed_seconds=time.perf_counter()-start)
        history.append(row); save_json(Path(args.out)/'history.json',history)
        print(json.dumps(row),flush=True)
        if score>best:
            best=score; stale=0
            torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                            config=config,epoch=epoch,threshold=chosen['threshold'],val=chosen['metrics']),
                       Path(args.out)/'unet.pt')
        else: stale+=1
        if stale>=args.patience: break


def load_net(out,device):
    checkpoint=torch.load(Path(out)/'unet.pt',map_location='cpu',weights_only=True)
    net=SmallUNet().to(device); net.load_state_dict(checkpoint['state_dict']); net.eval()
    return net,checkpoint


def evaluate(args):
    torch.set_num_threads(4); device=get_device(args.device)
    net,ckpt=load_net(args.out,device)
    width,height=ckpt['config']['width'],ckpt['config']['height']
    names,x,y=cached_pairs(args.data,args.out,'test',width,height)
    classic=json.loads((Path(args.out)/'classical.json').read_text())
    if (classic['width'],classic['height']) != (width,height):
        raise ValueError('Train and fit the classical baseline at the same resolution before comparing.')
    # Synchronize MPS inside probabilities via CPU transfer; first call warms GPU.
    probabilities(net,x[:1],device,1)
    start=time.perf_counter(); probs=probabilities(net,x,device,args.batch); elapsed=time.perf_counter()-start
    preds={'unet':probs>=ckpt['threshold']}
    start=time.perf_counter(); preds['classical']=np.stack([classical(a,classic['config']) for a in x]); classical_time=time.perf_counter()-start
    result=dict(split='test',images=len(names),checkpoint_epoch=ckpt['epoch'],threshold=ckpt['threshold'],
                resolution=[width,height],selection='Validation only; test not used for tuning',
                inference_seconds_excluding_io={'unet':elapsed,'classical':classical_time})
    rows=[]
    # Compare at native resolution, exactly the same upsampling for both masks.
    for method,pred in preds.items():
        reduced=[]; native=[]
        for name,a,b in zip(names,pred,y):
            reduced.append(confusion(a,b))
            target=gray(Path(args.data)/'masks_png'/name)>0
            restored=cv2.resize(a.astype(np.uint8),(target.shape[1],target.shape[0]),interpolation=cv2.INTER_NEAREST)
            c=confusion(restored,target);native.append(c)
            rows.append(dict(file=name,method=method,**scores(c)))
        result[method]=dict(native_resolution=summarize(native),model_resolution=summarize(reduced))
    save_json(Path(args.out)/'test_metrics.json',result)
    with (Path(args.out)/'test_per_image.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # A few fixed examples, chosen by name before looking at scores.
    from PIL import Image,ImageDraw
    n=min(6,len(names)); panel=Image.new('RGB',(320*5,270*n),'white'); draw=ImageDraw.Draw(panel)
    for i in range(n):
        cells=[x[i],y[i]*255,(probs[i]*255).astype(np.uint8),preds['unet'][i].astype(np.uint8)*255,preds['classical'][i]*255]
        for j,(title,a) in enumerate(zip(['Image','Target','Probability','U-Net','Classical'],cells)):
            draw.text((j*320,i*270),title,fill='black')
            im=Image.fromarray(a).convert('RGB');im.thumbnail((320,240));panel.paste(im,(j*320,i*270+18))
        draw.text((0,i*270+258),names[i],fill='black')
    panel.save(Path(args.out)/'examples.jpg')
    print(json.dumps(result,indent=2))


def predict(args):
    torch.set_num_threads(4); device=get_device(args.device)
    out=Path(args.destination); out.mkdir(parents=True,exist_ok=True)
    if out.resolve()==Path(args.input).resolve(): raise ValueError('Output must differ from input')
    source=Path(args.input)
    files=[source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'})
    if args.limit: files=files[:args.limit]
    if not files: raise ValueError('No input images')
    if args.method=='unet':
        net,config=load_net(args.out,device); width,height=config['config']['width'],config['config']['height']
    else:
        config=json.loads((Path(args.out)/'classical.json').read_text());width,height=config['width'],config['height']
    rows=[]
    for i,path in enumerate(files):
        target=out/(path.name+'.mask.png')
        if target.exists(): raise FileExistsError(f'Refusing to overwrite {target}')
        start=time.perf_counter(); raw=gray(path); h,w=raw.shape
        x=np.round(cv2.resize(normalize(raw),(width,height),interpolation=cv2.INTER_AREA)*255).astype(np.uint8)
        if args.method=='unet':
            p=probabilities(net,x[None],device,1)[0]
            mask=p>=config['threshold']
        else: mask=classical(x,config['config'])
        mask=cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
        if not cv2.imwrite(str(target),mask*255): raise IOError(f'Cannot write {target}')
        rows.append(dict(file=path.name,seconds=time.perf_counter()-start,foreground_fraction=float(mask.mean())))
        if (i+1)%100==0: print(f'{i+1}/{len(files)}',flush=True)
    save_json(out/'inference.json',dict(method=args.method,images=len(rows),device=str(device),
              total_seconds=sum(r['seconds'] for r in rows),per_image=rows,
              note='No ground truth: timing and predicted coverage only; not accuracy. Masks stored as 0/255.'))
    print(f'Saved {len(rows)} masks to {out}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default=str(ROOT/'vt'));p.add_argument('--out',default=str(ROOT/'outputs'))
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.set_defaults(func=prepare);prep.add_argument('--seed',type=int,default=42)
    for command,func in [('classical',fit_classical),('train',train),('evaluate',evaluate),('predict',predict)]:
        q=sub.add_parser(command);q.set_defaults(func=func)
        q.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
        q.add_argument('--batch',type=int,default=4)
        if command in ['train','classical']:
            q.add_argument('--width',type=int,default=512);q.add_argument('--height',type=int,default=384)
        if command=='train':
            q.add_argument('--seed',type=int,default=42);q.add_argument('--epochs',type=int,default=40)
            q.add_argument('--patience',type=int,default=10);q.add_argument('--lr',type=float,default=.001)
        if command=='predict':
            q.add_argument('--input',required=True);q.add_argument('--destination',required=True)
            q.add_argument('--method',choices=['unet','classical'],default='unet');q.add_argument('--limit',type=int)
    args=p.parse_args();args.func(args)


if __name__=='__main__':
    main()
