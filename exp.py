import nni
import GPUtil
from UCTB.dataset.data_loader import MultiLevelGridTrafficLoader
from UCTB.model.ArbFlow_paradigm import ArbFlowDev
from UCTB.evaluation import metric
import yaml
import argparse
import os
import pickle as pkl

parser = argparse.ArgumentParser(description="Argument Parser")
# data source
parser.add_argument('--dataset', default='Taxi')
parser.add_argument('--city', default='NYC')
# network parameter
parser.add_argument('--num_residual_unit', default=3, type=int)
parser.add_argument('--num_scale', default=1, type=int)
parser.add_argument('--scale', default=4, type=int)
parser.add_argument('--conv_filters', default=48, type=int)
parser.add_argument('--kernel_size', default=3, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--MergeIndex', default=1, type=int)
parser.add_argument('--train', default='True', type=str)
parser.add_argument('--normalize', default='True', type=str)
parser.add_argument('--max_epoch', default=2000, type=int)
parser.add_argument('--early_stop_length', default=300, type=int)
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--result_store',default='True', type=str)
parser.add_argument('--mark',default='new_paradigm_equal_ratio_prior_weight', type=str)
parser.add_argument('--fusion_method',default='concat', type=str)
parser.add_argument('--multi_weight',default='prior', type=str)
# parser.add_argument('--optscale_filename',default='scale_structure', type=str)
parser.add_argument('--option',default='normal', type=str)
parser.add_argument('--result_name',default='ArbFlow', type=str)
parser.add_argument('--target_width',default=128, type=int)
parser.add_argument('-s','--setting',default='scale_setting.yml', type=str)
parser.add_argument('--target_height',default=128, type=int)


args = vars(parser.parse_args())
with open(args['setting'],'r') as f:
    args.update(yaml.safe_load(f))
args['mark'] = args['mark'] + args['option']

if args['normalize']=='True':
    args['normalize']=True
else:
    args['normalize']=False

code_version = 'ArbFlow_{}_{}_{}_{}'.format(args['dataset'], args['city'], args['mark'],args['option'])
scale_list = args['scale_list']
nni_params = nni.get_next_parameter()
nni_sid = nni.get_sequence_id()
if nni_params:
    args.update(nni_params)
    code_version += ('_' + str(nni_sid))

deviceIDs = GPUtil.getAvailable(order='memory', limit=2, maxLoad=1, maxMemory=0.7,
                                includeNan=False, excludeID=[], excludeUUID=[])

if len(deviceIDs) == 0:
    current_device = '-1'
else:
    if nni_params:
        current_device = str(deviceIDs[int(nni_sid) % len(deviceIDs)])
    else:
        current_device = str(deviceIDs[0])

# Config data loader
data_loader = MultiLevelGridTrafficLoader(dataset=args['dataset'], city=args['city'],test_ratio=0.2, closeness_len=6, period_len=7, trend_len=4, MergeIndex=args['MergeIndex'],data_dir=args['data_dir'],normalize=args['normalize'],scale_list = scale_list,is_padding=False)

print(scale_list)

ArbFlow_Obj = ArbFlowDev(closeness_len=data_loader.closeness_len,
                          period_len=data_loader.period_len,
                          trend_len=data_loader.trend_len,
                          external_dim=data_loader.external_dim, lr=args['lr'],scale_list=scale_list,
                          num_residual_unit=args['num_residual_unit'], conv_filters=args['conv_filters'],
                          kernel_size=args['kernel_size'], width=data_loader.width, height=data_loader.height,
                          gpu_device=current_device, code_version=code_version,fusion_method=args['fusion_method'],multi_weight=args['multi_weight'],residual_unit_list=args['blocks_per_scale'])
ArbFlow_Obj.build()

print(args['dataset'], args['city'], code_version)
print('Number of trainable variables', ArbFlow_Obj.trainable_vars)
print('Number of training samples', data_loader.train_sequence_len)

print('debug')
if args['train']=='True':
    ArbFlow_Obj.fit(closeness_feature=data_loader.train_closeness,
                  period_feature=data_loader.train_period,
                  trend_feature=data_loader.train_trend,
                  targets=data_loader.multi_level_truth_train,
                  external_feature=data_loader.train_ef,
                  sequence_length=data_loader.train_sequence_len,
                  batch_size=args['batch_size'], early_stop_length=args['early_stop_length'],
                  validate_ratio=0.125,max_epoch=args['max_epoch'])
ArbFlow_Obj.load(code_version)
# Predict
import time
t_1 = time.time()
prediction = ArbFlow_Obj.predict(closeness_feature=data_loader.test_closeness,
                                   period_feature=data_loader.test_period,
                                   trend_feature=data_loader.test_trend,
                                   targets=data_loader.multi_level_truth_test,
                                   external_feature=data_loader.test_ef,
                                   sequence_length=data_loader.test_sequence_len,
                                   output_names=tuple('prediction_{}'.format(i)for i in range(len(scale_list)+1)))
t_2 = time.time()
print('Duration',t_2-t_1)
train_prediction = ArbFlow_Obj.predict(closeness_feature=data_loader.train_closeness,
                                   period_feature=data_loader.train_period,
                                   trend_feature=data_loader.train_trend,
                                   targets=data_loader.multi_level_truth_train,
                                   external_feature=data_loader.train_ef,
                                   sequence_length=data_loader.train_sequence_len,
                                   output_names=tuple('prediction_{}'.format(i)for i in range(len(scale_list)+1)))

# Compute metric
from UCTB.utils.utils_ArbFlow import save_predict_result

result = save_predict_result(data_loader,args['mark'],data_loader.test_y.shape[0],'multi_scale',prediction,train_prediction,scale_list=scale_list)

# Evaluate
val_loss = ArbFlow_Obj.load_event_scalar('val_loss')

best_val_loss = min([e[-1] for e in val_loss])
import numpy as np
import os
if not os.path.isdir(os.path.join('output',code_version)):
    os.makedirs(os.path.join('output',code_version))
import pickle as pkl
if args['result_store']=='True':
    with open('result/{}.pkl'.format(args['result_name']+'_'+args['dataset']+'_'+args['city']),'wb') as fp:
        pkl.dump(result,fp)
# Evaluate
val_loss = ArbFlow_Obj.load_event_scalar('val_loss')

best_val_loss = min([e[-1] for e in val_loss])
best_val_loss = best_val_loss



print('Converged using %.2f hour' % ((val_loss[-1][0] - val_loss[0][0]) / 3600))
if nni_params:
    nni.report_final_result({
        'default': best_val_loss,
    })