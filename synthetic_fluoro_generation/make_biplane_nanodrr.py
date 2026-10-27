#!/usr/bin/env python3
"""Six static panels using the nanoDRR API of the user's existing GUI."""
import argparse, json, tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from nanodrr.data import Subject
from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.drr import render


def window(raw,low=2,high=99):
    if not np.isfinite(raw).all(): raise ValueError('Non-finite projection')
    lo,hi=np.percentile(raw,[low,high])
    if hi<=lo: lo,hi=float(raw.min()),float(raw.max())
    if hi<=lo: raise ValueError('CT projection is constant; inspect raw output')
    return np.clip((raw-lo)/(hi-lo),0,1)


def geometry(subject,args,rot,trans,sdd,field,device):
    pixel=field/args.size
    k=make_k_inv(sdd,pixel,pixel,0.,0.,args.size,args.size,device=device)
    # Same inputs/API call as 8_1_fluoroscopy_sim_nanodrr(3).py.
    r=torch.tensor([rot],dtype=torch.float32,device=device)
    t=torch.tensor([trans],dtype=torch.float32,device=device)
    rt=make_rt_inv(r,t,orientation='AP',isocenter=subject.isocenter)
    return k,rt,torch.tensor([sdd],dtype=torch.float32,device=device)


def panels(ct,vessel,args):
    base=window(ct,args.window_low,args.window_high)
    if not args.no_invert_ct: base=1-base
    v=gaussian_filter(vessel,args.blur_px) if args.blur_px else vessel.copy()
    positive=v[v>0]
    if positive.size==0: raise ValueError('No vessels in projection; check pose and labels')
    strength=np.clip(v/max(float(np.percentile(positive,99)),1e-8),0,1)
    return [np.uint8(np.clip(a,0,1)*255) for a in (.48+.48*strength,base,.88-.75*strength)]


PALETTE=np.array([[230,25,75],[60,180,75],[0,130,200],[245,130,48],[145,30,180],[70,240,240],[240,50,230],[210,245,60],[250,190,212],[0,128,128],[220,190,255],[170,110,40],[255,250,200],[128,0,0],[170,255,195],[128,128,0],[255,215,180],[0,0,128],[128,128,128],[255,255,255]],dtype=np.uint8)


def colours_for_ids(ids):
    import colorsys
    # Colours depend on original ID, so changing the selection does not recolour others.
    return np.array([PALETTE[i-1] if i<=len(PALETTE) else np.array(colorsys.hsv_to_rgb((i*.61803398875)%1,.75,.95))*255 for i in ids],dtype=np.uint8)


def impose_labels(gray, channels, colours, alpha):
    rgb=np.repeat(gray[...,None],3,axis=2).astype(np.float32)
    strongest=channels.argmax(axis=0)
    mask=channels.max(axis=0)>0
    # Multiple 3D labels may project onto the same pixel: strongest channel wins.
    rgb[mask]=(1-alpha)*rgb[mask]+alpha*colours[strongest[mask]]
    return np.uint8(np.clip(rgb,0,255))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ct',type=Path,required=True);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--size',type=int,default=512)
    p.add_argument('--label-ids',type=int,nargs='+')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--a-rot',type=float,nargs=3,default=[0,0,0])
    p.add_argument('--b-rot',type=float,nargs=3,default=[90,0,0])
    p.add_argument('--a-trans',type=float,nargs=3,default=[0,850,0])
    p.add_argument('--b-trans',type=float,nargs=3,default=[0,850,0])
    p.add_argument('--a-sdd-mm',type=float,default=1000);p.add_argument('--b-sdd-mm',type=float,default=1170)
    p.add_argument('--a-field-mm',type=float,default=320);p.add_argument('--b-field-mm',type=float,default=400)
    p.add_argument('--window-low',type=float,default=2);p.add_argument('--window-high',type=float,default=99)
    p.add_argument('--no-invert-ct',action='store_true')
    p.add_argument('--blur-px',type=float,default=.7)
    p.add_argument('--overlay-alpha',type=float,default=.4)
    p.add_argument('--a-flip-horizontal',action='store_true')
    p.add_argument('--b-flip-horizontal',action='store_true')
    args=p.parse_args()
    if not 0<=args.overlay_alpha<=1: p.error('Overlay alpha must be between 0 and 1')
    if args.size<16 or args.blur_px<0 or min(args.a_sdd_mm,args.b_sdd_mm,args.a_field_mm,args.b_field_mm)<=0:
        p.error('Invalid dimensions/distances')
    if not 0<=args.window_low<args.window_high<=100: p.error('Invalid window percentiles')
    device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    print('Device:',device,flush=True)
    ct=nib.load(args.ct);lab=nib.load(args.labels)
    if len(ct.shape)!=3 or ct.shape!=lab.shape or not np.allclose(ct.affine,lab.affine,atol=1e-3,rtol=0):
        raise ValueError('CT and labels must share the same 3D dimensions and affine')
    values=lab.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all() or not np.allclose(values,np.rint(values),atol=1e-4,rtol=0):
        raise ValueError('Expected finite integer labels')
    ids=sorted(int(v) for v in np.unique(values) if v>0)
    selected=args.label_ids or ids
    if not selected or min(selected)<=0 or set(selected)-set(ids): raise ValueError('Selected labels missing/invalid')
    selected=sorted(set(selected))
    remapped=np.zeros(values.shape,dtype=np.uint16)
    for channel,label_id in enumerate(selected,start=1): remapped[values==label_id]=channel
    colours=colours_for_ids(selected)
    print('Selected label IDs:',selected,flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    metadata={'engine':'nanodrr','ct':str(args.ct),'labels':str(args.labels),'selected_ids':selected,'pose_convention':'Exact make_rt_inv input convention of supplied nanoDRR GUI; rotation values passed directly, AP orientation; translation is camera/source pose, not target shift. Machine-angle mapping unverified.','views':{}}
    with tempfile.TemporaryDirectory(prefix='nanodrr-vessels-') as tmp:
        binary_path=Path(tmp)/'selected_multiclass_labels.nii.gz'
        image=nib.Nifti1Image(remapped,lab.affine)
        image.set_data_dtype(np.uint16);nib.save(image,binary_path)
        # Consecutive temporary IDs ensure channel order; original IDs are retained in the legend.
        subject=Subject.from_filepath(str(args.ct),str(binary_path)).to(device)
        canvas=Image.new('RGB',(args.size*3,(args.size+36)*2),'black')
        labelled_canvas=canvas.copy()
        for row,(name,rot,trans,sdd,field) in enumerate([('A',args.a_rot,args.a_trans,args.a_sdd_mm,args.a_field_mm),('B',args.b_rot,args.b_trans,args.b_sdd_mm,args.b_field_mm)]):
            print('Rendering plane',name,flush=True)
            k,rt,sd=geometry(subject,args,rot,trans,sdd,field,device)
            with torch.no_grad(): projection=render(subject,k,rt,sd,args.size,args.size)
            raw=projection.detach().cpu().numpy()
            if raw.ndim!=4 or raw.shape[0]!=1 or raw.shape[1]!=len(selected)+1:
                raise ValueError(f'Expected nanoDRR CT + {len(selected)} foreground channels, got {raw.shape}. Installed API/channel layout differs from reference GUI.')
            flipped=args.a_flip_horizontal if name=='A' else args.b_flip_horizontal
            if flipped: raw=np.flip(raw,axis=-1).copy()
            ct_raw=raw[0,0]
            label_channels=raw[0,1:]
            vessel_raw=label_channels.sum(axis=0)
            print('CT raw range:',float(ct_raw.min()),float(ct_raw.max()),flush=True)
            np.savez_compressed(args.output_dir/f'plane_{name}_raw.npz',ct_projection=ct_raw,vessel_projection=vessel_raw,label_projections=label_channels,label_ids=np.array(selected),label_colours_rgb=colours)
            metadata['views'][name]={'display_horizontal_flip':flipped,'rot_input':rot,'trans_input_mm':trans,'sdd_mm':sdd,'field_mm':field,'pixel_mm':field/args.size,'k_inv':k.detach().cpu().tolist(),'rt_inv':rt.detach().cpu().tolist()}
            display_panels=panels(ct_raw,vessel_raw,args)
            overlay=Image.fromarray(impose_labels(display_panels[1],label_channels,colours,args.overlay_alpha))
            overlay.save(args.output_dir/f'plane_{name}_enhanced_with_labels.png')
            for col,(title,array) in enumerate(zip(['Bright vessel projection','Enhanced nanoDRR CT','Dark vessel projection'],display_panels)):
                tile=Image.fromarray(array)
                tile.save(args.output_dir/f'plane_{name}_{["vessels_bright","ct_drr","vessels_dark"][col]}.png')
                y=row*(args.size+36)
                canvas.paste(tile.convert('RGB'),(col*args.size,y+36))
                labelled_canvas.paste(overlay if col==1 else tile.convert('RGB'),(col*args.size,y+36))
                ImageDraw.Draw(labelled_canvas).text((col*args.size+8,y+10),f'{name}: {title}' + (' + labels' if col==1 else ''),fill='white')
                ImageDraw.Draw(canvas).text((col*args.size+8,y+10),f'{name}: {title}',fill='white')
        canvas.save(args.output_dir/'biplane_without_labels.png')
        labelled_canvas.save(args.output_dir/'biplane_with_labels.png')
        metadata['overlay']={'alpha':args.overlay_alpha,'overlap_rule':'Strongest raw label channel wins','colours_by_original_id':{str(i):c.tolist() for i,c in zip(selected,colours)}}
        legend=Image.new('RGB',(360,32*len(selected)+30),'white')
        draw=ImageDraw.Draw(legend)
        draw.text((10,8),'Original label IDs and overlay colours',fill='black')
        for index,(label_id,colour) in enumerate(zip(selected,colours)):
            y=30+32*index
            draw.rectangle((10,y,40,y+22),fill=tuple(int(v) for v in colour))
            draw.text((52,y+5),f'Label {label_id}',fill='black')
        legend.save(args.output_dir/'label_colour_legend.png')
        (args.output_dir/'geometry.json').write_text(json.dumps(metadata,indent=2))
    print('Saved',args.output_dir/'biplane_without_labels.png',flush=True)
    print('Saved',args.output_dir/'biplane_with_labels.png',flush=True)

if __name__=='__main__':main()
