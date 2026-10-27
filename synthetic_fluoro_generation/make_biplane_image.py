#!/usr/bin/env python3
"""Idealised static cone-beam CT and vessel projections on one shared geometry."""
import argparse, json, math
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw


def load_volumes(ct_path, labels_path, ids):
    ct, labels = nib.load(ct_path), nib.load(labels_path)
    if len(ct.shape)!=3 or ct.shape!=labels.shape:
        raise ValueError('CT and labels must be 3D with identical dimensions')
    if not np.allclose(ct.affine, labels.affine, atol=1e-3, rtol=0):
        raise ValueError('CT and labels have different physical grids; do not overwrite geometry to fix this')
    if any(n<2 for n in ct.shape):
        raise ValueError('All volume dimensions must be at least 2')
    hu=ct.get_fdata(dtype=np.float32)
    lab=labels.get_fdata(dtype=np.float32)
    if not np.isfinite(hu).all() or not np.isfinite(lab).all():
        raise ValueError('Non-finite voxels')
    if not np.allclose(lab,np.rint(lab)):
        raise ValueError('Labels must be integer IDs')
    present=np.unique(lab).astype(int)
    print('Label IDs present:',present.tolist(),flush=True)
    if ids:
        absent=set(ids)-set(present.tolist())
        if absent: raise ValueError(f'Requested IDs absent: {sorted(absent)}')
        vessel=np.isin(lab,ids).astype(np.float32)
    else:
        vessel=(lab>0).astype(np.float32)
    if not vessel.any(): raise ValueError('Selected vessel volume is empty')
    # Simple monoenergetic approximation, not DeepDRR material/spectrum physics.
    mu=.02*np.maximum(0,1+np.clip(hu,-1000,4000)/1000)
    affine=np.diag([-1.,-1.,1.,1.])@ct.affine # nibabel RAS world -> LPS world
    data=np.stack([mu,vessel]).transpose(0,3,2,1).copy()
    return data,affine,ct.shape,present.tolist()


def project(data,affine,shape,angles,shift,sod,sdd,field,size,step,batch,device):
    center=(affine@np.r_[((np.array(shape)-1)/2),1])[:3]
    target=center+np.array(shift)
    # Explicit intrinsic ZXY renderer angles; NOT a calibrated machine-angle mapping.
    rot=Rotation.from_euler('ZXY',angles,degrees=True).as_matrix()
    direction=rot@np.array([0.,1.,0.]) # AP: source anterior (LPS -Y), beam posterior
    up=rot@np.array([0.,0.,1.])
    right=np.cross(direction,up)
    source=target-sod*direction
    detector=source+sdd*direction
    pixel=field/size
    horizontal=(np.arange(size)-(size-1)/2)*pixel
    vertical=((size-1)/2-np.arange(size))*pixel
    xx,yy=np.meshgrid(horizontal,vertical)
    points=detector+xx[...,None]*right+yy[...,None]*up
    delta=points.reshape(-1,3)-source
    lengths=np.linalg.norm(delta,axis=1)
    rays=delta/lengths[:,None]
    inv=np.linalg.inv(affine)
    srcvox=(inv@np.r_[source,1])[:3]
    dv=rays@inv[:3,:3].T
    # Slab intersection with voxel support, including half-voxel boundary.
    lo=np.full(len(rays),-np.inf);hi=np.full(len(rays),np.inf)
    valid=np.ones(len(rays),bool)
    for axis in range(3):
        parallel=np.abs(dv[:,axis])<1e-10
        if srcvox[axis]<-.5 or srcvox[axis]>shape[axis]-.5:
            valid &= ~parallel
        denom=np.where(parallel,1,dv[:,axis])
        a=(-.5-srcvox[axis])/denom
        b=(shape[axis]-.5-srcvox[axis])/denom
        lo=np.maximum(lo,np.where(parallel,-np.inf,np.minimum(a,b)))
        hi=np.minimum(hi,np.where(parallel,np.inf,np.maximum(a,b)))
    lo=np.maximum(lo,0);hi=np.minimum(hi,lengths)
    valid &= hi>lo
    indices=np.flatnonzero(valid)
    output=np.zeros((2,len(rays)),np.float32)
    volume=torch.as_tensor(data,device=device).unsqueeze(0)
    src=torch.tensor(srcvox,dtype=torch.float32,device=device)
    norm=torch.tensor(np.array(shape)-1,dtype=torch.float32,device=device)
    with torch.inference_mode():
        for start in range(0,len(indices),batch):
            ix=indices[start:start+batch]
            span=hi[ix]-lo[ix]
            n=max(1,math.ceil(float(span.max())/step))
            fraction=(torch.arange(n,device=device,dtype=torch.float32)+.5)/n
            t=torch.tensor(lo[ix],device=device,dtype=torch.float32)[:,None]+torch.tensor(span,device=device,dtype=torch.float32)[:,None]*fraction
            directions=torch.tensor(dv[ix],device=device,dtype=torch.float32)
            xyz=src+directions[:,None,:]*t[:,:,None]
            grid=(2*xyz/norm-1).reshape(1,1,len(ix),n,3)
            samples=F.grid_sample(volume,grid,mode='bilinear',padding_mode='zeros',align_corners=True)
            integral=samples[0,:,0].sum(-1)*torch.tensor(span/n,device=device,dtype=torch.float32)
            output[:,ix]=integral.cpu().numpy()
            if start%(batch*20)==0: print(f'  Rays {min(start+batch,len(indices))}/{len(indices)}',flush=True)
    return output.reshape(2,size,size),dict(angles_ZXY_deg=angles,target_LPS_mm=target.tolist(),source_LPS_mm=source.tolist(),beam_LPS=direction.tolist(),sod_mm=sod,sdd_mm=sdd,detector_field_mm=field,pixel_mm=pixel)


def displays(integrals,blur,noise,seed):
    ct,vessels=integrals
    transmission=np.exp(-ct)
    low,high=np.percentile(transmission,[1,99])
    ctview=np.clip((transmission-low)/max(high-low,1e-6),0,1)
    vessel=gaussian_filter(vessels,blur) if blur>0 else vessels.copy()
    positive=vessel[vessel>0]
    scale=np.percentile(positive,99) if positive.size else 1
    strength=np.clip(vessel/max(scale,1e-6),0,1)
    rng=np.random.default_rng(seed)
    # Opposite display polarities of the SAME static vessel projection.
    bright=.48+.48*strength
    dark=.88-.75*strength
    result=[]
    for arr in (bright,ctview,dark):
        if noise>0: arr=arr+rng.normal(0,noise,arr.shape)
        result.append(np.uint8(np.clip(arr,0,1)*255))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ct',type=Path,required=True);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--label-ids',type=int,nargs='+',help='Selected arterial IDs; omitted means all positive labels')
    p.add_argument('--size',type=int,default=512)
    p.add_argument('--step-mm',type=float,default=.7)
    p.add_argument('--batch-rays',type=int,default=1024)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--a-angles',type=float,nargs=3,default=[0,0,0],metavar=('Z','X','Y'))
    p.add_argument('--b-angles',type=float,nargs=3,default=[90,0,0],metavar=('Z','X','Y'))
    p.add_argument('--target-shift-mm',type=float,nargs=3,default=[0,0,0])
    p.add_argument('--sod-mm',type=float,default=850)
    p.add_argument('--a-sdd-mm',type=float,default=1000);p.add_argument('--b-sdd-mm',type=float,default=1170)
    p.add_argument('--a-field-mm',type=float,default=320);p.add_argument('--b-field-mm',type=float,default=400)
    p.add_argument('--blur-px',type=float,default=.7);p.add_argument('--noise',type=float,default=.008)
    args=p.parse_args()
    if args.size<16 or args.step_mm<=0 or args.batch_rays<1 or args.sod_mm<=0 or min(args.a_field_mm,args.b_field_mm)<=0:
        p.error('Invalid size, sampling or distance')
    if min(args.a_sdd_mm,args.b_sdd_mm)<=args.sod_mm: p.error('SDD must exceed source-to-target distance')
    if args.blur_px<0 or args.noise<0: p.error('Blur and noise must be nonnegative')
    if args.label_ids and min(args.label_ids)<=0: p.error('Choose positive vessel label IDs')
    device='cpu'
    if args.device!='cpu':
        try:
            if not torch.cuda.is_available(): raise RuntimeError('CUDA not available')
            F.grid_sample(torch.zeros((1,1,2,2,2),device='cuda'),torch.zeros((1,1,1,1,3),device='cuda'),align_corners=True)
            torch.cuda.synchronize();device='cuda'
        except Exception as e:
            if args.device=='cuda': raise
            print('Using CPU:',e,flush=True)
    print('Rendering device:',device,flush=True)
    data,affine,shape,present=load_volumes(args.ct,args.labels,args.label_ids)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    canvas=Image.new('RGB',(args.size*3,(args.size+36)*2),'black')
    metadata={'ct':str(args.ct),'labels':str(args.labels),'selected_ids':args.label_ids or 'all positive','present_ids':present,'device':device,'step_mm':args.step_mm,'views':{},'warning':'Idealised static projections; not calibrated DSA, roadmap, or DeepDRR output. Angles are intrinsic ZXY in LPS, not verified machine RAO/LAO values.'}
    for row,(name,angles,sdd,field) in enumerate([('A',args.a_angles,args.a_sdd_mm,args.a_field_mm),('B',args.b_angles,args.b_sdd_mm,args.b_field_mm)]):
        print('Rendering plane',name,flush=True)
        integral,geom=project(data,affine,shape,angles,args.target_shift_mm,args.sod_mm,sdd,field,args.size,args.step_mm,args.batch_rays,device)
        np.savez_compressed(args.output_dir/f'plane_{name}_raw.npz',ct_line_integral=integral[0],vessel_length_mm=integral[1])
        metadata['views'][name]=geom
        for col,(title,array) in enumerate(zip(['Bright vessel projection','CT DRR','Dark vessel projection'],displays(integral,args.blur_px,args.noise,row))):
            image=Image.fromarray(array)
            image.save(args.output_dir/f'plane_{name}_{["vessels_bright","ct_drr","vessels_dark"][col]}.png')
            y=row*(args.size+36)
            canvas.paste(image.convert('RGB'),(col*args.size,y+36))
            ImageDraw.Draw(canvas).text((col*args.size+8,y+10),f'{name}: {title}',fill='white')
    canvas.save(args.output_dir/'biplane_six_panel.png')
    (args.output_dir/'geometry.json').write_text(json.dumps(metadata,indent=2))
    print('Saved',args.output_dir/'biplane_six_panel.png',flush=True)

if __name__=='__main__':main()
