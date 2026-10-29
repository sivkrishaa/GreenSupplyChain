import numpy as np, json, itertools
from scipy.stats import spearmanr
rng=np.random.default_rng(2026)
C=[f"C{i}" for i in range(1,13)]
W=np.array([(.40,.50),(.55,.75),(.75,.90),(.20,.35),(.50,.60),(.50,.60),(.65,.85),(.75,.90),(.63,.78),(.73,.88),(.80,.98),(.69,.93)])
R=np.array([
[(7,9.5),(6.5,8.5),(5.5,7.5),(6,9)],[(5.3,6.8),(5.8,8.3),(5.8,8.3),(5.3,6.8)],[(3.5,4.5),(2.3,3.8),(2.5,4),(2.8,4)],
[(1.8,3.5),(2.3,3.8),(1.5,3.3),(2.5,4)],[(3.3,4.3),(2.8,4),(3.3,4.5),(3.3,4.3)],[(4.8,6.3),(4.8,6.3),(5.3,7.3),(5.5,8)],
[(5.5,7.5),(6.8,9.3),(5.5,7.5),(5.8,8.3)],[(5,6),(5.8,8.3),(5.5,7.5),(5.5,7.5)],[(5.5,7.5),(6.5,8.5),(5.8,8.3),(5.5,7.5)],
[(5.5,7.5),(6.5,8.5),(5.3,6.8),(5.5,7.5)],[(5.3,6.8),(6.5,8.5),(5.8,8.3),(5.5,7.5)],[(5.3,6.8),(7.3,8.8),(5.5,7.5),(5.3,6.8)]])  # [crit, supp, 2]
SC=[(0,.1),(.1,.3),(.3,.4),(.4,.5),(.5,.6),(.6,.9),(.9,1.0)]
def P(a,b):  # P{a<=b}
    L=(a[1]-a[0])+(b[1]-b[0])
    if L==0: return 1.0 if a[0]<=b[0] else 0.0
    return max(0,L-max(0,a[1]-b[0]))/L
def weighted(Wm,Rm=R):
    N=Rm/10.0
    V=np.empty_like(N); V[:,:,0]=N[:,:,0]*Wm[:,None,0]; V[:,:,1]=N[:,:,1]*Wm[:,None,1]; return V
def grey(Wm,Rm=R):
    V=weighted(Wm,Rm); m=V.shape[0]
    smax=np.stack([V[:,:,0].max(1),V[:,:,1].max(1)],1); smin=np.stack([V[:,:,0].min(1),V[:,:,1].min(1)],1)
    P1=np.array([np.mean([P(V[j,s],smax[j]) for j in range(m)]) for s in range(4)])
    P2=np.array([np.mean([P(smin[j],V[j,s]) for j in range(m)]) for s in range(4)])
    return V,smax,smin,P1,P2,P1/P2
def rank(x,asc=True):
    o=np.argsort(x if asc else -x); r=np.empty(4,int); r[o]=np.arange(1,5); return r
out={}
V,smax,smin,P1,P2,CC=grey(W)
out['V']=np.round(V,3).tolist(); out['smax']=np.round(smax,3).tolist(); out['smin']=np.round(smin,3).tolist()
out['P1']=P1.round(3).tolist(); out['P2']=P2.round(3).tolist(); out['CC']=CC.round(3).tolist(); base=rank(CC); out['rank']=base.tolist()
# rounding robustness: CC interval if P reported to 2dp
out['CCrange']=[[round((round(P1[s],2)-.005)/(round(P2[s],2)+.005),3),round((round(P1[s],2)+.005)/(round(P2[s],2)-.005),3)] for s in range(4)]
# ---- method comparison
wm=W.mean(1); wn=wm/wm.sum(); X=R.mean(2)  # crit x supp
meth={}
meth['Proposed grey possibility ratio (CC = P1/P2)']=base
meth['Li et al. (2007) possibility degree (P1 only)']=rank(P1)
# crisp TOPSIS
Nv=X/np.sqrt((X**2).sum(1,keepdims=True)); Vv=Nv*wn[:,None]
dp=np.sqrt(((Vv-Vv.max(1,keepdims=True))**2).sum(0)); dm=np.sqrt(((Vv-Vv.min(1,keepdims=True))**2).sum(0)); cct=dm/(dp+dm)
meth['Crisp TOPSIS (interval midpoints)']=rank(cct,False)
# grey distance TOPSIS
gp=np.sqrt(((V[:,:,0]-smax[:,None,0])**2+(V[:,:,1]-smax[:,None,1])**2).sum(0)/2)
gm=np.sqrt(((V[:,:,0]-smin[:,None,0])**2+(V[:,:,1]-smin[:,None,1])**2).sum(0)/2)
ccg=gm/(gp+gm); meth['Grey TOPSIS (interval Euclidean distance)']=rank(ccg,False)
# GRA
Nm=(X-X.min(1,keepdims=True))/(X.max(1,keepdims=True)-X.min(1,keepdims=True)); D=1-Nm
gc=(D.min()+0.5*D.max())/(D+0.5*D.max()); grg=(gc*wn[:,None]).sum(0)
meth['Grey relational analysis (ζ = 0.5)']=rank(grg,False)
# VIKOR
f_best=X.max(1,keepdims=True); f_w=X.min(1,keepdims=True); d=wn[:,None]*(f_best-X)/(f_best-f_w)
S=d.sum(0); Rv=d.max(0); Q=.5*(S-S.min())/(S.max()-S.min())+.5*(Rv-Rv.min())/(Rv.max()-Rv.min())
meth['VIKOR (v = 0.5, midpoints)']=rank(Q)
out['scores']={'crispTOPSIS':cct.round(3).tolist(),'greyDist':ccg.round(3).tolist(),'GRA':grg.round(3).tolist(),'VIKOR_Q':Q.round(3).tolist()}
out['methods']={k:[v.tolist(),round(spearmanr(v,base)[0],3)] for k,v in meth.items()}
# cost-only
out['cost_only_rank']=rank(X[0],False).tolist()
# dimension mean weights
dims={'Traditional':range(0,6),'Environmental':range(6,9),'Social':range(9,12)}
out['dimw']={k:round(float(wm[list(v)].mean()),3) for k,v in dims.items()}
# ---- SA1 OAT
oat=[];top=np.zeros(4)
for j in range(12):
  for f in [0.6,0.8,1.2,1.4]:
    Wm=W.copy(); Wm[j]=np.minimum(Wm[j]*f,1.0)
    cc=grey(Wm)[5]; r=rank(cc); top[np.argmin(cc)]+=1
    oat.append(dict(c=C[j],f=f,cc=cc.round(3).tolist(),r=r.tolist(),rho=round(spearmanr(r,base)[0],3)))
out['oat']=oat; out['oat_top']=top.tolist(); out['oat_rev']=int(sum(1 for o in oat if o['r']!=base.tolist()))
out['oat_minrho']=min(o['rho'] for o in oat)
out['oat_revlist']=[(o['c'],o['f'],o['r']) for o in oat if o['r']!=base.tolist()]
# ---- SA2 scenarios
AH,M,L=(.9,1.0),(.4,.5),(.3,.4)
scen={}
scen['Base case (expert weights)']=W
scen['Equal weights (all Medium)']=np.array([M]*12)
for name,idx in dims.items():
  Wm=np.array([L]*12); Wm[list(idx)]=AH; scen[f'{name}-led (AH on {name.lower()}, L elsewhere)']=Wm
Wm=np.array([L]*12); Wm[0]=AH; scen['Cost-led (AH on C1, L elsewhere)']=Wm
Wm=W.copy(); Wm[[6,7,8,9,10,11]]=np.array([M]*6); scen['Sustainability de-emphasised (C7–C12 set to Medium)']=Wm
out['scen']={k:dict(cc=grey(v)[5].round(3).tolist(),r=rank(grey(v)[5]).tolist()) for k,v in scen.items()}
# identical-interval (original design) recomputed
out['uniform']={f'[{a}, {b}]':dict(cc=grey(np.array([(a,b)]*12))[5].round(3).tolist(),r=rank(grey(np.array([(a,b)]*12))[5]).tolist()) for a,b in SC}
# ---- SA3 Monte Carlo weights
N=10000; RA=np.zeros((4,4)); ccs=[]
for _ in range(N):
  Wm=np.array([np.mean([SC[i] for i in rng.integers(0,7,4)],0) for _ in range(12)])
  cc=grey(Wm)[5]; r=rank(cc); RA[np.arange(4),r-1]+=1; ccs.append(cc)
out['mc_w']=(RA/N*100).round(1).tolist()
ccs=np.array(ccs); out['mc_w_cc']=[[round(float(np.percentile(ccs[:,s],q)),3) for q in (2.5,50,97.5)] for s in range(4)]
# ---- SA4 Monte Carlo ratings ±0.5
RB=np.zeros((4,4))
for _ in range(N):
  Rm=np.clip(R+rng.uniform(-.5,.5,R.shape),0,10); Rm=np.sort(Rm,axis=2)
  r=rank(grey(W,Rm)[5]); RB[np.arange(4),r-1]+=1
out['mc_r']=(RB/N*100).round(1).tolist()
json.dump(out,open('/home/claude/rev/results.json','w'),indent=1)
for k in ['P1','P2','CC','rank','CCrange','methods','cost_only_rank','dimw','oat_top','oat_rev','oat_minrho','oat_revlist','scen','uniform','mc_w','mc_w_cc','mc_r','scores']:
  print(k,out[k])
np.save('/home/claude/rev/ccs.npy',ccs)
