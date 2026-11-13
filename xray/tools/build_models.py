#!/usr/bin/env python3
"""Builds public/models/anatomy.glb + labels.json from extracted BodyParts3D .obj files.
Usage: python3 tools/build_models.py --lists DIR_WITH_LIST_FILES --obj DIR [DIR ...] --out public/models
Needs: python3 + numpy. List files: bodyparts3d_partof_element_parts.csv, isa_element_parts.txt
Head space: millimetres, nasion at origin, +Y up, +Z forward. Jaw parts are relative to the jaw hinge."""
import argparse,csv,glob,json,os,re,struct,collections
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--lists',required=True);ap.add_argument('--obj',nargs='+',required=True);ap.add_argument('--out',default='public/models');a=ap.parse_args()
N=[0,-185,1530];HG=[0,-95,1500]   # nasion / jaw hinge in BodyParts3D coordinates
# ---- names ----
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
have=lambda pat,fm=False:sorted(f for f,n in leaf.items() if f in obj and (re.fullmatch(pat,n,re.I) if fm else re.search(pat,n,re.I)))
# ---- geometry helpers ----
def parse(fn):
    V=[];F=[]
    for l in open(fn):
        t=l.split()
        if not t:continue
        if t[0]=='v':V.append([float(x) for x in t[1:4]])
        elif t[0]=='f':
            ix=[int(x.split('/')[0])-1 for x in t[1:]]
            for k in range(1,len(ix)-1):F.append([ix[0],ix[k],ix[k+1]])
    return np.array(V),np.array(F)
def cluster(V,F,cs):
    c=np.round(V/cs).astype(np.int64);u,inv_=np.unique(c,axis=0,return_inverse=True);inv_=inv_.ravel()
    nv=np.zeros((len(u),3));cnt=np.zeros(len(u));np.add.at(nv,inv_,V);np.add.at(cnt,inv_,1);nv/=cnt[:,None]
    nf=inv_[F];k=(nf[:,0]!=nf[:,1])&(nf[:,1]!=nf[:,2])&(nf[:,0]!=nf[:,2]);return nv,nf[k]
def head_space(V,o):return np.stack([V[:,0]-o[0],V[:,2]-o[2],-(V[:,1]-o[1])],1)
# ---- groups: node -> (frame, ids, cluster size) ----
G={}
def add(node,ids,frame='N',cs=None):G[node]=(frame,sorted(set(ids)),cs)
brain=sorted(els['FMA50801']&set(obj))
bk=lambda pat:[f for f in brain if re.search(pat,leaf[f],re.I)]
cb=bk(r'^cerebellum$');bs=bk(r'^(pons|medulla oblongata)$')
dp=bk(r'ventricle|aqueduct|habenula|hypothalam|colliculus|internal capsule|hippocampus|pineal|subcortex|cardinal segment')
add('cerebrum',[f for f in brain if f not in cb+bs+dp],cs=1.8);add('deep',dp,cs=1.8);add('cerebellum',cb,cs=1.8);add('brainstem',bs,cs=1.8)
add('skull',have(r'^(frontal bone|ethmoid|occipital part of head|sphenoid bone|vomer|(left|right) (inferior nasal concha|lacrimal bone|maxilla|nasal bone|palatine bone|parietal bone|temporal bone|zygomatic bone))$',True))
add('jaw_bone',have(r'^mandible$',True),'HG')
add('teeth_upper',have(r'upper .*tooth$|^upper jaw$'));add('teeth_lower',have(r'lower .*tooth$|^gingiva of lower jaw$'),'HG')
iris=have(r'^(left|right) iris$',True)
add('iris',iris,cs=.6);add('eyes',[f for f in have(r'^(left|right) (choroid|cornea|sclera|lens|vitreous body|suspensory ligament of (left|right) lens)$|^anterior chamber of (left|right) eyeball$',True)],cs=.9)
add('eye_muscles',have(r'^(left|right) (superior rectus|inferior rectus|medial rectus|lateral rectus|superior oblique|inferior oblique|levator palpebrae superioris)$',True))
add('palate',have(r'(levator|tensor) veli palatini|^uvular muscle$'),cs=.8);add('tongue',have(r'^tongue$',True),'HG',.8)
add('glands',have(r'lacrimal gland$|submandibular gland$'),cs=1.0)
add('neck',have(r'^(atlas|axis|(third|fourth|fifth|sixth|seventh) cervical vertebra|hyoid bone|intervertebral disk of (axis|(third|fourth|fifth|sixth|seventh) cervical vertebra))$',True),cs=1.0)
# ---- build ----
bin_=b'';bv=[];ac=[];meshes=[];nodes=[];cent={}
for node,(fr,ids,cs) in G.items():
    if not ids:print('skip (no files):',node);continue
    o=HG if fr=='HG' else N;Vs=[];Fs=[];off=0
    for i in ids:
        V,F=parse(obj[i])
        if cs:V,F=cluster(V,F,cs)
        Vh=head_space(V,o);cent[i]=Vh.mean(0);Vs.append(Vh);Fs.append(F+off);off+=len(V)
    V=np.vstack(Vs).astype(np.float32);F=np.vstack(Fs).astype(np.uint32)
    bv.append(dict(buffer=0,byteOffset=len(bin_),byteLength=V.nbytes,target=34962));bin_+=V.tobytes()
    ac.append(dict(bufferView=len(bv)-1,componentType=5126,count=len(V),type='VEC3',min=V.min(0).tolist(),max=V.max(0).tolist()))
    bv.append(dict(buffer=0,byteOffset=len(bin_),byteLength=F.nbytes,target=34963));bin_+=F.tobytes()
    ac.append(dict(bufferView=len(bv)-1,componentType=5125,count=F.size,type='SCALAR'))
    meshes.append(dict(primitives=[dict(attributes=dict(POSITION=len(ac)-2),indices=len(ac)-1)]));nodes.append(dict(name=node,mesh=len(meshes)-1))
    print(f'{node:12s} {len(ids):3d} files {len(V):6d} verts  head-space y {V[:,1].min():.0f}..{V[:,1].max():.0f}')
j=dict(asset=dict(version='2.0'),scene=0,scenes=[dict(nodes=list(range(len(nodes))))],nodes=nodes,meshes=meshes,accessors=ac,bufferViews=bv,buffers=[dict(byteLength=len(bin_))])
jb=json.dumps(j,separators=(',',':')).encode();jb+=b' '*((-len(jb))%4)
os.makedirs(a.out,exist_ok=True)
open(f'{a.out}/anatomy.glb','wb').write(struct.pack('<III',0x46546C67,2,28+len(jb)+len(bin_))+struct.pack('<II',len(jb),0x4E4F534A)+jb+struct.pack('<II',len(bin_),0x004E4942)+bin_)
# ---- labels (priority order = display order) ----
L=[('Frontal bone','skull',r'^frontal bone$'),('Mandible (lower jaw)','skull',r'^mandible$'),('Nasal bone','skull',r'nasal bone$'),('Zygomatic bone','skull',r'zygomatic bone$'),
('Cerebellum','brain',r'^cerebellum$'),('Precentral gyrus','brain',r'precentral gyrus'),('Occipital lobe','brain',r'occipital lobe'),('Cornea','eyes',r'cornea$'),('Iris','eyes',r'iris$'),
('Superior rectus','muscles',r'superior rectus'),('Tongue','muscles',r'^tongue$'),('Central incisor','teeth',r'upper central secondary incisor'),('Maxilla','skull',r'maxilla$'),
('Temporal bone','skull',r'temporal bone$'),('Parietal bone','skull',r'parietal bone$'),('Atlas (C1)','neck',r'^atlas$'),('Axis (C2)','neck',r'^axis$'),('Hyoid bone','neck',r'^hyoid bone$'),
('Pons','brain',r'^pons$'),('Medulla oblongata','brain',r'^medulla oblongata$'),('Superior frontal gyrus','brain',r'superior frontal gyrus'),('Postcentral gyrus','brain',r'postcentral gyrus'),
('Inferior temporal gyrus','brain',r'inferior temporal gyrus'),('Insula','brain',r'insula$'),('Lens','eyes',r'lens$'),('Sclera','eyes',r'sclera$'),('Lateral rectus','muscles',r'lateral rectus'),
('Levator palpebrae superioris','muscles',r'levator palpebrae'),('Submandibular gland','muscles',r'submandibular gland'),('Lacrimal gland','muscles',r'lacrimal gland'),('Uvular muscle','muscles',r'uvular muscle'),
('Canine tooth','teeth',r'upper secondary canine'),('Molar tooth','teeth',r'upper second secondary molar'),('Occipital bone','skull',r'occipital part of head'),('Sphenoid bone','skull',r'sphenoid bone'),
('Ethmoid','skull',r'^ethmoid$'),('Vomer','skull',r'^vomer$'),('Lacrimal bone','skull',r'lacrimal bone$'),('Inferior nasal concha','skull',r'inferior nasal concha'),
('Hippocampus','brain',r'hippocampus'),('Hypothalamus','brain',r'^hypothalamus$'),('Lateral ventricle','brain',r'lateral ventricle'),('Cingulate gyrus','brain',r'cingulate gyrus'),
('Supramarginal gyrus','brain',r'supramarginal'),('Angular gyrus','brain',r'angular gyrus'),('Pineal body','brain',r'pineal'),('Third cervical vertebra (C3)','neck',r'^third cervical vertebra$'),
('C4 vertebra','neck',r'^fourth cervical vertebra$'),('C5 vertebra','neck',r'^fifth cervical vertebra$'),('C6 vertebra','neck',r'^sixth cervical vertebra$'),('C7 vertebra','neck',r'^seventh cervical vertebra$')]
out=[]
for t,layer,pat in L:
    ids=[f for f in cent if re.search(pat,leaf[f],re.I)]
    if ids:out.append(dict(t=t,l=layer,f='HG' if any(G[n][0]=='HG' and f in G[n][1] for n in G for f in ids) else 'N',a=[[round(float(x),1) for x in cent[f]] for f in ids]))
    else:print('no anchor for label:',t)
json.dump(out,open(f'{a.out}/labels.json','w'),separators=(',',':'))
print(len(out),'labels;',os.path.getsize(f'{a.out}/anatomy.glb')//1024,'KB glb')
