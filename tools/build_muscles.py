#!/usr/bin/env python3
"""Muscles + arm bones. Two modes:
  python3 tools/build_muscles.py --lists DIR --ids muscle_ids.txt            -> writes the list of .obj ids to collect
  python3 tools/build_muscles.py --lists DIR --obj DIR [DIR ...] --out public/models   -> writes muscles.glb + muscles.json
Same head space as anatomy.glb / torso.glb (mm, nasion origin, +Y up, +Z forward)."""
import argparse,csv,glob,json,os,re,struct,collections
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--lists',required=True);ap.add_argument('--obj',nargs='*',default=[]);ap.add_argument('--out',default='public/models');ap.add_argument('--ids');a=ap.parse_args()
N=[0,-185,1530]
els=collections.defaultdict(set);nm={}
for r in list(csv.reader(open(f'{a.lists}/bodyparts3d_partof_element_parts.csv')))[1:]+[l.rstrip('\n').split('\t') for l in open(f'{a.lists}/isa_element_parts.txt')][1:]:
    els[r[0]].add(r[2]);nm[r[0]]=r[1]
inv=collections.defaultdict(list)
for c,s in els.items():
    for f in s:inv[f].append(c)
leaf={f:nm[min(cs,key=lambda c:len(els[c]))] for f,cs in inv.items()}
BAD=re.compile(r'artery|vein|nerve|ganglion|tributary|foot|leg\b|thigh|femor|gluteus|sartorius|gracilis|adductor|pectineus|soleus|gastrocnemius|tibialis|fibularis|popliteus|plantar|hallucis|hand$|of (left|right) hand|retinaculum|membrane|ligament|tendon|digitorum longus|digitorum brevis|obturator|gemellus|piriformis|quadratus femoris|coccygeus|levator ani|puborectalis|perineum|lumbrical|interossei',re.I)
# node -> (regex, frame, simplify cell mm). Frame: N head, T torso, UP/UM upper arm plus/minus side, FP/FM forearm
DEF=[('neckm',r'sternocleidomastoid|scalenus|platysma|digastric|mylohyoid|geniohyoid|stylohyoid|omohyoid|sternohyoid|sternothyroid|thyrohyoid|longus colli|longus capitis|rectus capitis|obliquus capitis|splenius|longissimus (capitis|cervicis)|semispinalis (capitis|cervicis)|interspinales cervicis|cervical rotator|cervical intertransversarii|iliocostalis cervicis','N',1.6),
('chest',r'pectoralis|serratus anterior|subclavius|transversus thoracis|external oblique','T',3.2),
('back',r'trapezius|rhomboid|levator scapulae|serratus posterior|supraspinatus|infraspinatus|subscapularis|teres (major|minor)|iliocostalis (thoracis|lumborum)|longissimus thoracis|spinalis|(thoracic|lumbar) rotator|interspinalis thoracis|interspinales lumborum|levatores costarum|lumbar intertransversarius','T',3.0),
('inter',r'intercostal muscle','T',4.0),
('uarm_m',r'biceps brachii|triceps brachii|brachialis|coracobrachialis|deltoid','U',1.6),
('farm_m',r'brachioradialis|anconeus|pronator|supinator|flexor carpi|flexor digitorum (superficialis|profundus)|flexor pollicis longus|extensor carpi|extensor digitorum$|extensor digiti minimi$|extensor indicis|extensor pollicis|abductor pollicis longus|palmaris longus','F',1.4),
('uarm_b',r'^(left|right) humerus$','U',1.2),('farm_b',r'^(left|right) (radius|ulna)$','F',1.0)]
def assign():
    out=collections.defaultdict(list);used=set()
    for node,pat,fr,cs in DEF:
        for f,n in sorted(leaf.items()):
            if f in used or not re.fullmatch(r'FJ\d+M?',f):continue
            if node.endswith('_b') or not BAD.search(n):
                if re.search(pat,n,re.I):out[node].append(f);used.add(f)
    return out
G=assign()
if a.ids:
    ids=sorted({f for v in G.values() for f in v});open(a.ids,'w').write('\n'.join(ids)+'\n')
    for k,v in G.items():print(k,len(v))
    print('total',len(ids));raise SystemExit
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
def hs(V):return np.stack([V[:,0]-N[0],V[:,2]-N[2],-(V[:,1]-N[1])],1)
# load + simplify every part once
P={}
for node,pat,fr,cs in DEF:
    for f in G[node]:
        if f not in obj:continue
        V,F=parse(obj[f])
        if len(F)==0:continue
        V,F=cluster(V,F,cs)
        if len(F)==0:continue
        P[f]=(node,fr,hs(V),F)
print(len(P),'parts loaded of',sum(len(v) for v in G.values()))
# ---- arm joints (plus/minus x side) from the bones ----
def side(V):return 'P' if V[:,0].mean()>0 else 'M'
bones={}
for f,(node,fr,V,F) in P.items():
    if node in('uarm_b','farm_b'):bones.setdefault((node,side(V)),[]).append(V)
J={}
for sd in 'PM':
    hum=np.vstack(bones[('uarm_b',sd)]);fa=np.vstack(bones[('farm_b',sd)])
    sh=hum[hum[:,1].argmax()].copy();sh[1]-=12
    el=hum[hum[:,1]<hum[:,1].min()+25].mean(0)             # lower end of humerus
    wr=fa[fa[:,1]<fa[:,1].min()+20].mean(0)                # lower end of forearm bones
    J[sd]=dict(shoulder=sh,elbow=el,wrist=wr)
    print(sd,'shoulder',sh.round(0),'elbow',el.round(0),'wrist',wr.round(0))
# ---- groups ----
G2=collections.defaultdict(lambda:([],[]))
for f,(node,fr,V,F) in P.items():
    key={'N':'neckm','T':node}.get(fr)
    if fr=='U' or fr=='F':key=('uarm' if fr=='U' else 'farm')+('_b' if node.endswith('_b') else '_m')+'_'+side(V)
    elif fr=='N':key='neckm'
    G2[key][0].append(V);G2[key][1].append(F)
bin_=b'';bv=[];ac=[];meshes=[];nodes=[]
for key,(Vs,Fs) in sorted(G2.items()):
    off=0;FF=[]
    for V,F in zip(Vs,Fs):FF.append(F+off);off+=len(V)
    V=np.vstack(Vs).astype(np.float32);F=np.vstack(FF).astype(np.uint32)
    bv.append(dict(buffer=0,byteOffset=len(bin_),byteLength=V.nbytes,target=34962));bin_+=V.tobytes()
    ac.append(dict(bufferView=len(bv)-1,componentType=5126,count=len(V),type='VEC3',min=V.min(0).tolist(),max=V.max(0).tolist()))
    bv.append(dict(buffer=0,byteOffset=len(bin_),byteLength=F.nbytes,target=34963));bin_+=F.tobytes()
    ac.append(dict(bufferView=len(bv)-1,componentType=5125,count=F.size,type='SCALAR'))
    meshes.append(dict(primitives=[dict(attributes=dict(POSITION=len(ac)-2),indices=len(ac)-1)]));nodes.append(dict(name=key,mesh=len(meshes)-1))
    print(f'{key:10s} {len(V):6d} verts {len(F)//3:6d} tris')
j=dict(asset=dict(version='2.0'),scene=0,scenes=[dict(nodes=list(range(len(nodes))))],nodes=nodes,meshes=meshes,accessors=ac,bufferViews=bv,buffers=[dict(byteLength=len(bin_))])
jb=json.dumps(j,separators=(',',':')).encode();jb+=b' '*((-len(jb))%4)
os.makedirs(a.out,exist_ok=True)
open(f'{a.out}/muscles.glb','wb').write(struct.pack('<III',0x46546C67,2,28+len(jb)+len(bin_))+struct.pack('<II',len(jb),0x4E4F534A)+jb+struct.pack('<II',len(bin_),0x004E4942)+bin_)
# ---- labels ----
LB=[('Pectoralis major','chest','tmusc',r'pectoralis major'),('Serratus anterior','chest','tmusc',r'serratus anterior'),('External oblique','chest','tmusc',r'external oblique'),('Pectoralis minor','chest','tmusc',r'pectoralis minor'),
('Trapezius','back','tmusc',r'transverse part of \w+ trapezius'),('Rhomboid major','back','tmusc',r'rhomboid major'),('Infraspinatus','back','tmusc',r'infraspinatus'),('Teres major','back','tmusc',r'teres major'),
('Supraspinatus','back','tmusc',r'supraspinatus'),('Erector spinae','back','tmusc',r'longissimus thoracis'),('Intercostal muscle','inter','tmusc',r'external intercostal'),
('Sternocleidomastoid','neckm','tmusc',r'sternocleidomastoid'),('Platysma','neckm','tmusc',r'platysma'),('Scalenus anterior','neckm','tmusc',r'scalenus anterior'),('Digastric','neckm','tmusc',r'digastric'),('Splenius capitis','neckm','tmusc',r'splenius capitis'),
('Deltoid','uarm_m','amusc',r'acromial part of \w+ deltoid'),('Biceps brachii','uarm_m','amusc',r'long head of \w+ biceps brachii'),('Triceps brachii','uarm_m','amusc',r'long head of \w+ triceps brachii'),('Brachialis','uarm_m','amusc',r'brachialis'),
('Brachioradialis','farm_m','amusc',r'brachioradialis'),('Flexor carpi radialis','farm_m','amusc',r'flexor carpi radialis'),('Extensor digitorum','farm_m','amusc',r'extensor digitorum$'),('Pronator teres','farm_m','amusc',r'humeral head of \w+ pronator teres'),
('Humerus','uarm_b','abones',r'humerus$'),('Radius','farm_b','abones',r'radius$'),('Ulna','farm_b','abones',r'ulna$')]
out=[]
for t,node,layer,pat in LB:
    c=[(f,P[f]) for f in P if P[f][0]==node and re.search(pat,leaf[f],re.I)]
    if not c:print('no anchor for label:',t);continue
    for f,(nd,fr,V,F) in c[:2]:
        fk={'N':'N','T':'T','U':'U'+side(V),'F':'F'+side(V)}[fr]
        out.append(dict(t=t,l=layer,f=fk,a=[[round(float(x),1) for x in V.mean(0)]]))
json.dump(dict(joints={sd:{k:[round(float(x),1) for x in v] for k,v in d.items()} for sd,d in J.items()},labels=out),open(f'{a.out}/muscles.json','w'),separators=(',',':'))
print(len(out),'labels;',os.path.getsize(f'{a.out}/muscles.glb')//1024,'KB glb')
