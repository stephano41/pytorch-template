import torch

from src.loggers.result_trackers import MetricCollection



def test_metriccollection():
    mc = MetricCollection(['f1_macro','accuracy','f1_micro',{'_target_':'src.metrics.roc_auc'}], prefix='train/')
    mc.update(torch.tensor([1,0,1,0,1,0]),torch.tensor([1,0,1,0,1,1]))
    mc.update(torch.tensor([1,0,1,0,1,1]),torch.tensor([1,0,0,0,0,1]))
    mc.result()
    mc.reset()


