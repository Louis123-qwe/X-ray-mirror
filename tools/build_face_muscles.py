#!/usr/bin/env python3
"""Approximate facial muscles (BodyParts3D has none): simple shapes fitted to the skull in anatomy.glb.
python3 tools/build_face_muscles.py --glb public/models/anatomy.glb --out public/models   -> face_muscles.glb + face_muscles.json"""
import argparse,json,os,struct
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--glb',default='public/models/anatomy.glb');ap.add_argument('--out',default='public/models');ap.add_argument('--plot');a=ap.parse_args()
def load(fn):
    b=open(fn,'rb').read();jl=struct.unpack('<I',b[12:16])[0];j=json.loads(b[20:20+jl]);bs=b[20+jl+8:];o={}
    for n in j['nodes']:
        ac=j['accessors'][j['meshes'][n['mesh']]['primitives'][0]['attributes']['POSITION']];bv=j['bufferViews'][ac['bufferView']]
        o[n['name']]=np.frombuffer(bs[bv['byteOffset']:bv['byteOffset']+bv['byteLength']],dtype=np.float32).reshape(-1,3).copy()
    return o
H=load(a.glb);sk=H['skull']
def zsurf(x,y,r=7):
    m=(np.abs(sk[:,0]-x)<r)&(np.abs(sk[:,1]-y)<r)
    return sk[m,2].max() if m.any() else -60
def xsurf(sg,y,z,r=7):
    m=(np.abs(sk[:,1]-y)<r)&(np.abs(sk[:,2]-z)<r)&(sk[:,0]*sg>0)
    return (np.abs(sk[m,0]).max() if m.any() else 60)*sg
V=[];F=[]
def add(vs,fs):
    off=sum(len(v) for v in V);V.append(np.array(vs,dtype=float));F.append(np.array(fs)+off)
def sheet(f,nu=10,nv=8):
    pts=[f(i/nu,j/nv) for i in range(nu+1) for j in range(nv+1)];fs=[]
    for i in range(nu):
        for j in range(nv):
            p=i*(nv+1)+j;fs+=[[p,p+1,p+nv+1],[p+1,p+nv+2,p+nv+1]]
    add(pts,fs)
def tube(path,rad,closed=False,seg=8):
    path=np.array(path,float);n=len(path);rings=[];fs=[]
    for i in range(n):
        t=path[(i+1)%n]-path[i-1] if closed else path[min(i+1,n-1)]-path[max(i-1,0)]
        t/=np.linalg.norm(t);up=np.array([0,0,1.]) if abs(t[2])<.9 else np.array([1.,0,0]);u=np.cross(t,up);u/=np.linalg.norm(u);w=np.cross(t,u)
        r=rad[i] if hasattr(rad,'__len__') else rad
        rings+= [path[i]+r*(np.cos(k*2*np.pi/seg)*u+np.sin(k*2*np.pi/seg)*w) for k in range(seg)]
    m=n if closed else n-1
    for i in range(m):
        for k in range(seg):
            p=i*seg+k;q=((i+1)%n)*seg+k;p2=i*seg+(k+1)%seg;q2=((i+1)%n)*seg+(k+1)%seg;fs+=[[p,q,p2],[p2,q,q2]]
    add(rings,fs)
def ellipse(c,rx,ry,n=28,tilt=0):
    t=np.linspace(0,2*np.pi,n,endpoint=False);return [[c[0]+rx*np.cos(k),c[1]+ry*np.sin(k),c[2]+tilt*np.cos(k)] for k in t]
def ellipsoid(c,r,nu=10,nv=8):
    sheet(lambda u,v:[c[0]+r[0]*np.sin(np.pi*v)*np.cos(2*np.pi*u),c[1]+r[1]*np.cos(np.pi*v),c[2]+r[2]*np.sin(np.pi*v)*np.sin(2*np.pi*u)],nu,nv)
anch={}
for sg in (1,-1):
    # frontalis: forehead belly over the skull surface
    sheet(lambda u,v:(lambda x,y:[x,y,zsurf(x,y)+3])(sg*(4+u*36),24+v*40),10,8)
    # temporalis: fan over the temple from the arch up to the temporal line
    def temp(u,v):
        y=-8+u*62;lo=max(-52-0.85*(y+8),-128);hi=min(-52+0.4*(y+8),-36);z=lo+v*(hi-lo)
        return [xsurf(sg,y,z)+sg*2.5,y,z]
    sheet(temp,10,8)
    # masseter: from the zygomatic arch down to the angle of the jaw
    def mass(u,v):
        top=np.array([sg*62,-18,-74])*(1-u)+np.array([sg*60,-18,-48])*u
        bot=np.array([sg*52,-90,-84])*(1-u)+np.array([sg*40,-96,-52])*u
        p=top*(1-v)+bot*v;p[0]+=sg*7*np.sin(np.pi*v)*np.sin(np.pi*u);return p
    sheet(mass,8,8)
    tube(ellipse([sg*33,-12,-17],25,20),3.2,closed=True)                       # orbicularis oculi
    tube([[sg*46,-24,-24],[sg*40,-44,-8],[sg*30,-62,0],[sg*24,-70,3]],[3.2,3.4,3,2.6])  # zygomaticus major
tube(ellipse([0,-72,3],21,8),4.2,closed=True)                                   # orbicularis oris
ellipsoid([0,-88,-8],[9,6,5])                                                   # mentalis
V=np.vstack(V);F=np.vstack(F)
if a.plot:
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    off=np.array([0,-30,-90]);fig,ax=plt.subplots(1,2,figsize=(12,7))
    for k,xi in enumerate([0,2]):
        for n in ['skull','jaw_bone','teeth_upper','teeth_lower']:
            v=H[n][::4]+(off if n in('jaw_bone','teeth_lower') else 0);ax[k].scatter(v[:,xi],v[:,1],s=.3,c='#9bd')
        ax[k].scatter(V[::2,xi],V[::2,1],s=.6,c='#f55');ax[k].set_aspect('equal');ax[k].grid(True);ax[k].set_ylim(-120,110);ax[k].set_xlim((-90,90) if xi==0 else (-210,40))
    plt.tight_layout();plt.savefig(a.plot,dpi=70)
V32=V.astype(np.float32);F32=F.astype(np.uint32)
bin_=V32.tobytes()+F32.tobytes()
j=dict(asset=dict(version='2.0'),scene=0,scenes=[dict(nodes=[0])],nodes=[dict(name='face_m',mesh=0)],meshes=[dict(primitives=[dict(attributes=dict(POSITION=0),indices=1)])],
 accessors=[dict(bufferView=0,componentType=5126,count=len(V32),type='VEC3',min=V32.min(0).tolist(),max=V32.max(0).tolist()),dict(bufferView=1,componentType=5125,count=F32.size,type='SCALAR')],
 bufferViews=[dict(buffer=0,byteOffset=0,byteLength=V32.nbytes,target=34962),dict(buffer=0,byteOffset=V32.nbytes,byteLength=F32.nbytes,target=34963)],buffers=[dict(byteLength=len(bin_))])
jb=json.dumps(j,separators=(',',':')).encode();jb+=b' '*((-len(jb))%4)
os.makedirs(a.out,exist_ok=True)
open(f'{a.out}/face_muscles.glb','wb').write(struct.pack('<III',0x46546C67,2,28+len(jb)+len(bin_))+struct.pack('<II',len(jb),0x4E4F534A)+jb+struct.pack('<II',len(bin_),0x004E4942)+bin_)
L=[('Frontalis (approx.)',[[22,45,zsurf(22,45)+3]]),('Temporalis (approx.)',[[xsurf(1,25,-70)+3,25,-70]]),('Masseter (approx.)',[[62,-52,-62]]),('Orbicularis oculi (approx.)',[[33,-34,-17]]),
   ('Zygomaticus major (approx.)',[[40,-44,-8]]),('Orbicularis oris (approx.)',[[-21,-72,3]]),('Mentalis (approx.)',[[0,-88,-8]])]
json.dump([dict(t=t,l='muscles',f='N',a=[[round(float(c),1) for c in q] for q in p]) for t,p in L],open(f'{a.out}/face_muscles.json','w'),separators=(',',':'))
print(len(V),'verts',len(F),'tris',os.path.getsize(f'{a.out}/face_muscles.glb')//1024,'KB')
