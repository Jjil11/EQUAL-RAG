"""Exact statistical function bodies extracted from the frozen main analyzer."""
import math
import numpy as np

FIELDS=['grounded_quality','answer_f1','mean_tokens','repetition_rate','true_cap_hit_rate']

def clean(obj):
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [clean(v) for v in obj]
    if isinstance(obj,(float,np.floating)):return float(obj) if math.isfinite(obj) else None
    if isinstance(obj,np.integer):return int(obj)
    return obj

def sufficient(vals):
    return np.array([[v['grounded_edges'],v['gold_edges'],v['tokens'],v['answer_f1'],float(v['repeated_units']>0),float(v['true_cap'])] for v in vals],float).reshape(-1,6)

def observed(a):
    if len(a)==0:return np.full(5,np.nan)
    den=a[:,1].sum()
    return np.array([a[:,0].sum()/den if den else np.nan,a[:,3].mean(),a[:,2].mean(),a[:,4].mean(),a[:,5].mean()])

def reduce_draw(v):
    den=v[:,:,1].sum(1)
    return np.column_stack([np.divide(v[:,:,0].sum(1),den,out=np.full(len(v),np.nan),where=den>0),v[:,:,3].mean(1),v[:,:,2].mean(1),v[:,:,4].mean(1),v[:,:,5].mean(1)])

def bootstrap(a,qix,six=None,chunk=256):
    if qix.shape[1]==0:return np.full((len(qix),5),np.nan)
    out=[]
    for start in range(0,len(qix),chunk):
        ix=qix[start:start+chunk]
        v=a[ix] if six is None else a[six[start:start+chunk,:,None],ix[:,None,:],:].mean(1)
        out.append(reduce_draw(v))
    return np.concatenate(out)

def interval(delta,draw,kind):
    out={}
    for i,f in enumerate(FIELDS):
        finite=draw[np.isfinite(draw[:,i]),i]
        out[f]={'delta':float(delta[i]),'ci95':np.quantile(finite,[.025,.975]).tolist() if len(finite) else [None,None],
                'defined_draws':len(finite)}
    out.update(samples=len(draw),question_rng_seed=20261010,seed_rng_seed=20261011,kind=kind,
        zero_Q_denominator_draws=int(np.isnan(draw[:,0]).sum()),no_multiple_comparison_adjustment=True,
        undefined_Q_excluded_only_from_Q=True)
    return clean(out)

def mean_record(v):
    return {'mean':float(np.mean(v)),'SD_sample_ddof1':float(np.std(v,ddof=1))}
