from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'/'analysis'))
from negative_gradient_diagnostic import classify_regions, bag_gradients_from_logits, contiguous_run_mask
from motion_intensity_analysis import intensity_metrics, assign_terciles
from common import bootstrap_mean

m=contiguous_run_mask(np.array([0,1,1,1,0,1,1],dtype=bool),3)
assert m.tolist()==[False,True,True,True,False,False,False]
p=np.array([0.1,0.2,0.8,0.7,0.65,0.1,0.1])
labels=classify_regions(p,0.8,0.01,0.5,3)
assert (labels==2).sum()>=1
logits=np.array([-2,-1,0,1,2],dtype=float)
g,loss=bag_gradients_from_logits(logits,0.1)
assert np.isfinite(g).all() and np.isfinite(loss) and np.abs(g).sum()>0
sig=np.column_stack([np.sin(np.linspace(0,4,100)),np.cos(np.linspace(0,4,100)),np.ones(100)])
met=intensity_metrics(sig,50.0)
assert np.isfinite(met['dynamic_vm_rms'])
groups,q1,q2=assign_terciles({'a':1,'b':2,'c':3,'d':4,'e':5,'f':6})
assert set(groups.values())=={'low','middle','high'}
b=bootstrap_mean([1,2,3,4],1000,1)
assert b['n']==4 and b['ci_low'] <= b['mean'] <= b['ci_high']
print('SYNTHETIC_TESTS_PASSED')
