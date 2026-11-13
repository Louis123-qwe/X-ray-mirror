#!/usr/bin/env python3
"""Builds public/models/torso.glb + labels_torso.json from the torso BodyParts3D .obj files.
Usage: python3 tools/build_torso.py --lists DIR_WITH_LIST_FILES --obj DIR [DIR ...] --out public/models
Same head space as anatomy.glb (mm, nasion at origin, +Y up, +Z forward), so head and torso line up."""
import argparse,csv,glob,json,os,re,struct,collections
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--lists',required=True);ap.add_argument('--obj',nargs='+',required=True);ap.add_argument('--out',default='public/models');a=ap.parse_args()
N=[0,-185,1530]
els=collections.defaultdict(set);nm={}
for r in list(csv.reader(open(f'{a.lists}/bodyparts3d_partof_element_parts.csv')))[1:]+[l.rstrip('\n').split('\t') for l in open(f'{a.lists}/isa_element_parts.txt')][1:]:
    els[r[0]].add(r[2]);nm[r[0]]=r[1]
inv=collections.defaultdict(list)
for c,s in els.items():
    for f in s:inv[f].append(c)
leaf={f:nm[min(cs,key=lambda c:len(els[c]))] for f,cs in inv.items()}
obj={}
for d in a.obj:
    for p in glob.glob(f'{d}/**/FJ*.obj',recursive=True):obj[os.path.basename(p)[:-4]]=p
have=lambda pat:sorted(f for f,n in leaf.items() if f in obj and re.search(pat,n,re.I))
def parse(fn):
    V=[];F=[]
    for l in open(fn):
        t=l.split()
        if not t:continue
        if t[0]=='v':V.append([float(x) for x in t[1:4]])
        elif t[0]=='f':
            ix=[int(x.split('/')[0])-1 for x in t[1:]]
            for k in range(1,len(ix)-1):F.append([ix[0],ix[k],ix[k+1]])
    return np.array(V).reshape(-1,3),np.array(F,dtype=np.int64).reshape(-1,3)
def cluster(V,F,cs):
    c=np.round(V/cs).astype(np.int64);u,inv_=np.unique(c,axis=0,return_inverse=True);inv_=inv_.ravel()
    nv=np.zeros((len(u),3));cnt=np.zeros(len(u));np.add.at(nv,inv_,V);np.add.at(cnt,inv_,1);nv/=cnt[:,None]
    nf=inv_[F];k=(nf[:,0]!=nf[:,1])&(nf[:,1]!=nf[:,2])&(nf[:,0]!=nf[:,2]);return nv,nf[k]
def head_space(V):return np.stack([V[:,0]-N[0],V[:,2]-N[2],-(V[:,1]-N[1])],1)
G={}
def add(node,pat,cs):G[node]=(have(pat),cs)
add('ribs',r'^(left|right) \w+ (rib|costal cartilage)$|^(manubrium|body of sternum|xiphoid process)$',2.0)
add('spine_t',r'^(\w+) (thoracic|lumbar) vertebra$|^intervertebral disk of \w+ (thoracic|lumbar) vertebra$|^sacrum$',1.8)
add('shoulder',r'^(left|right) (clavicle|scapula)$',2.0)
add('pelvis',r'^(left|right) pelvic girdle$',2.5)
add('heart',r'^(wall of ventricle|wall of (left|right) atrium)$',1.6)
add('vessels',r'^(pulmonary trunk|ascending aorta|arch of aorta|descending aorta|descending thoracic aorta|abdominal aorta|inferior vena cava|superior vena cava)$',1.6)
add('airways',r'^trachea$|main bronchus$|bronchial tree$',1.0)
add('liver',r'^(hepatovenous segment .*|caudate lobe of liver)$',3.0)
add('gi',r'^(esophagus|stomach|spleen|pancreas|gallbladder)$',2.5)
add('kidneys',r'^(left|right) (kidney|adrenal gland)$',2.5)
add('tmuscle',r'^(diaphragm|(left|right) psoas major)$',3.0)
bin_=b'';bv=[];ac=[];meshes=[];nodes=[];cent={};ext={}
for node,(ids,cs) in G.items():
    if not ids:print('skip (no files):',node);continue
    Vs=[];Fs=[];off=0
    for i in ids:
        V,F=parse(obj[i])
        if len(F)==0 or len(V)==0:print('empty:',i,leaf[i]);continue
        V,F=cluster(V,F,cs)
        if len(F)==0:print('collapsed:',i,leaf[i]);continue
        Vh=head_space(V);cent[i]=Vh.mean(0);Vs.append(Vh);Fs.append(F+off);off+=len(V)
    V=np.vstack(Vs).astype(np.float32);F=np.vstack(Fs).astype(np.uint32)
    ext[node]=(V.min(0),V.max(0))
    bv.append(dict(buffer=0,byteOffset=len(bin_),byteLength=V.nbytes,target=34962));bin_+=V.tobytes()
    ac.append(dict(bufferView=len(bv)-1,componentType=5126,count=len(V),type='VEC3',min=V.min(0).tolist(),max=V.max(0).tolist()))
    bv.append(dict(buffer=0,byteOffset=len(bin_),byteLength=F.nbytes,target=34963));bin_+=F.tobytes()
    ac.append(dict(bufferView=len(bv)-1,componentType=5125,count=F.size,type='SCALAR'))
    meshes.append(dict(primitives=[dict(attributes=dict(POSITION=len(ac)-2),indices=len(ac)-1)]));nodes.append(dict(name=node,mesh=len(meshes)-1))
    print(f'{node:9s} {len(ids):3d} files {len(V):6d} verts {len(F)//3:6d} tris  y {V[:,1].min():.0f}..{V[:,1].max():.0f}')
j=dict(asset=dict(version='2.0'),scene=0,scenes=[dict(nodes=list(range(len(nodes))))],nodes=nodes,meshes=meshes,accessors=ac,bufferViews=bv,buffers=[dict(byteLength=len(bin_))])
jb=json.dumps(j,separators=(',',':')).encode();jb+=b' '*((-len(jb))%4)
os.makedirs(a.out,exist_ok=True)
open(f'{a.out}/torso.glb','wb').write(struct.pack('<III',0x46546C67,2,28+len(jb)+len(bin_))+struct.pack('<II',len(jb),0x4E4F534A)+jb+struct.pack('<II',len(bin_),0x004E4942)+bin_)
# ---- anchors used by the app to place the torso on the body-pose landmarks ----
def verts(pat):
    return np.vstack([head_space(parse(obj[f])[0]) for f in have(pat)])
lh=verts(r'^left humerus$');rh=verts(r'^right humerus$')
sl=lh[lh[:,1].argmax()];sr=rh[rh[:,1].argmax()]          # top of each humerus = shoulder joint
sl=np.array([sl[0],sl[1]-12,sl[2]]);sr=np.array([sr[0],sr[1]-12,sr[2]])
pv=verts(r'pelvic girdle$');top,bot=pv[:,1].max(),pv[:,1].min();wid=pv[:,0].max()-pv[:,0].min();cx=pv[:,0].mean();zc=pv[:,2].mean()
hl=[cx+0.30*wid,top-0.58*(top-bot),zc];hr=[cx-0.30*wid,hl[1],zc]
if sl[0]<sr[0]:sl,sr=sr,sl                               # sl = +x side
A=dict(shoulderPlus=[round(float(x),1) for x in sl],shoulderMinus=[round(float(x),1) for x in sr],hipPlus=[round(x,1) for x in hl],hipMinus=[round(x,1) for x in hr])
print('anchors',A)
# ---- labels ----
L=[('Sternum','tbones',r'body of sternum'),('Clavicle','tbones',r'clavicle$'),('Scapula','tbones',r'scapula$'),('1st rib','tbones',r'first rib$'),('7th rib','tbones',r'seventh rib$'),('12th rib','tbones',r'twelfth rib$'),
('Costal cartilage','tbones',r'fifth costal cartilage'),('Manubrium','tbones',r'^manubrium$'),('Xiphoid process','tbones',r'xiphoid'),('T1 vertebra','tbones',r'^first thoracic vertebra$'),('T7 vertebra','tbones',r'^seventh thoracic vertebra$'),
('T12 vertebra','tbones',r'^twelfth thoracic vertebra$'),('L1 vertebra','tbones',r'^first lumbar vertebra$'),('L3 vertebra','tbones',r'^third lumbar vertebra$'),('L5 vertebra','tbones',r'^fifth lumbar vertebra$'),
('Sacrum','tbones',r'^sacrum$'),('Pelvis (hip bone)','tbones',r'pelvic girdle'),('Heart','organs',r'^wall of ventricle$'),('Left atrium','organs',r'wall of left atrium'),('Right atrium','organs',r'wall of right atrium'),
('Aortic arch','organs',r'arch of aorta'),('Ascending aorta','organs',r'ascending aorta'),('Pulmonary trunk','organs',r'pulmonary trunk'),('Superior vena cava','organs',r'superior vena cava'),
('Trachea','organs',r'^trachea$'),('Bronchial tree','organs',r'left main bronchus|right main bronchus'),('Oesophagus','organs',r'^esophagus$'),('Liver','organs',r'hepatovenous segment (v|vi|vii|viii)$'),
('Stomach','organs',r'^stomach$'),('Pancreas','organs',r'^pancreas$'),('Gallbladder','organs',r'^gallbladder$'),('Kidney','organs',r'kidney$'),('Adrenal gland','organs',r'adrenal gland$'),
('Diaphragm','organs',r'^diaphragm$'),('Psoas major','organs',r'psoas major'),('Descending aorta','organs',r'descending thoracic aorta'),('Inferior vena cava','organs',r'inferior vena cava')]
out=[]
for t,layer,pat in L:
    ids=[f for f in cent if re.search(pat,leaf[f],re.I)]
    if ids:out.append(dict(t=t,l=layer,f='T',a=[[round(float(x),1) for x in cent[f]] for f in ids[:2]]))
    else:print('no anchor for label:',t)
json.dump(dict(anchors=A,labels=out),open(f'{a.out}/torso.json','w'),separators=(',',':'))
print(len(out),'labels;',os.path.getsize(f'{a.out}/torso.glb')//1024,'KB glb')
