#!/usr/bin/env python3
"""3D hand bones -> public/models/hands.glb + hands.json
python3 tools/build_hands.py --lists DIR --obj DIR --out public/models
One node per bone; hands.json holds each bone's pivot, axis and length so the app can pose it from MediaPipe hand landmarks."""
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
def parse(fn):
    V=[];F=[]
    for l in open(fn):
        t=l.split()
        if not t:continue
        if t[0]=='v' and len(t)>=4:V.append([float(x) for x in t[1:4]])
        elif t[0]=='f':
            ix=[int(x.split('/')[0])-1 for x in t[1:]]
            for k in range(1,len(ix)-1):F.append([ix[0],ix[k],ix[k+1]])
    return np.array(V).reshape(-1,3),np.array(F,dtype=np.int64).reshape(-1,3)
def cluster(V,F,cs):
    c=np.round(V/cs).astype(np.int64);u,inv_=np.unique(c,axis=0,return_inverse=True);inv_=inv_.ravel()
    nv=np.zeros((len(u),3));cnt=np.zeros(len(u));np.add.at(nv,inv_,V);np.add.at(cnt,inv_,1);nv/=cnt[:,None]
    nf=inv_[F];k=(nf[:,0]!=nf[:,1])&(nf[:,1]!=nf[:,2])&(nf[:,0]!=nf[:,2]);return nv,nf[k]
hs=lambda V:np.stack([V[:,0]-N[0],V[:,2]-N[2],-(V[:,1]-N[1])],1)
FING={'thumb':'thumb','index':'index finger','middle':'middle finger','ring':'ring finger','little':'little finger'}
ORD=['first','second','third','fourth','fifth']
raw={}   # (side, key) -> (V,F)
for f,n in leaf.items():
    if f not in obj:continue
    key=None
    m=re.match(r'^(left|right) (capitate|hamate|lunate|pisiform|scaphoid|trapezium|trapezoid|triquetral)$',n)
    if m:key='carp:'+m.group(2)
    m=re.match(r'^(left|right) (first|second|third|fourth|fifth) metacarpal bone$',n)
    if m:key=f'met{ORD.index(m.group(2))+1}'
    m=re.match(r'^(distal|middle|proximal) phalanx of (left|right) (thumb|index finger|middle finger|ring finger|little finger)$',n)
    if m:key={'distal':'dist','middle':'mid','proximal':'prox'}[m.group(1)]+'_'+[k for k,v in FING.items() if v==m.group(3)][0]
    if not key:continue
    V,F=parse(obj[f]);V,F=cluster(V,F,0.7)
    if len(F)==0:continue
    V=hs(V);sd='P' if V[:,0].mean()>0 else 'M'
    raw[(sd,key)]=(V,F)
print(len(raw),'bones')
nodes=[];meshes=[];ac=[];bv=[];bin_=b''
def addmesh(name,V,F):
    global bin_
    V=V.astype(np.float32);F=F.astype(np.uint32)
    bv.append(dict(buffer=0,byteOffset=len(bin_),byteLength=V.nbytes,target=34962));bin_+=V.tobytes()
    ac.append(dict(bufferView=len(bv)-1,componentType=5126,count=len(V),type='VEC3',min=V.min(0).tolist(),max=V.max(0).tolist()))
    bv.append(dict(buffer=0,byteOffset=len(bin_),byteLength=F.nbytes,target=34963));bin_+=F.tobytes()
    ac.append(dict(bufferView=len(bv)-1,componentType=5125,count=F.size,type='SCALAR'))
    meshes.append(dict(primitives=[dict(attributes=dict(POSITION=len(ac)-2),indices=len(ac)-1)]));nodes.append(dict(name=name,mesh=len(meshes)-1))
r3=lambda v:[round(float(x),2) for x in v]
INFO={}
for sd in 'PM':
    car=[v for (s,k),v in raw.items() if s==sd and k.startswith('carp:')]
    if not car:print('no carpals for',sd);continue
    CV=np.vstack([c[0] for c in car]);CF=[];off=0
    for V,F in car:CF.append(F+off);off+=len(V)
    wc=CV.mean(0);addmesh(f'carp_{sd}',CV,np.vstack(CF))
    B={}
    for (s,k),(V,F) in raw.items():
        if s!=sd or k.startswith('carp:'):continue
        c=V.mean(0);u,sv,vt=np.linalg.svd(V-c,full_matrices=False);ax=vt[0];t=(V-c)@ax
        L=t.max()-t.min();lo=V[t<t.min()+0.12*L].mean(0);hi=V[t>t.max()-0.12*L].mean(0)
        prox,dist=(lo,hi) if np.linalg.norm(lo-wc)<np.linalg.norm(hi-wc) else (hi,lo)
        ad=(dist-prox)/np.linalg.norm(dist-prox)
        is_met=k.startswith('met')
        B[k]=dict(pivot=r3(dist if is_met else prox),axis=r3(ad),len=round(float(np.linalg.norm(dist-prox)),2),anchor='dist' if is_met else 'prox')
        addmesh(f'{k}_{sd}',V,F)
    m3=np.array(B['met3']['pivot']);ax=(m3-wc);ln=float(np.linalg.norm(ax))
    B['carp']=dict(pivot=r3(wc),axis=r3(ax/ln),len=round(ln,2),anchor='prox')
    INFO[sd]=B
    print(sd,len(B),'bones; palm length',round(ln,1),'mm')
j=dict(asset=dict(version='2.0'),scene=0,scenes=[dict(nodes=list(range(len(nodes))))],nodes=nodes,meshes=meshes,accessors=ac,bufferViews=bv,buffers=[dict(byteLength=len(bin_))])
jb=json.dumps(j,separators=(',',':')).encode();jb+=b' '*((-len(jb))%4)
os.makedirs(a.out,exist_ok=True)
open(f'{a.out}/hands.glb','wb').write(struct.pack('<III',0x46546C67,2,28+len(jb)+len(bin_))+struct.pack('<II',len(jb),0x4E4F534A)+jb+struct.pack('<II',len(bin_),0x004E4942)+bin_)
json.dump(INFO,open(f'{a.out}/hands.json','w'),separators=(',',':'))
print(os.path.getsize(f'{a.out}/hands.glb')//1024,'KB glb')
