import pickle as pkl
import os
import argparse
import yaml
from UCTB.utils.utils_ArbFlow import *
import pdb
'''
{
    'model_name':'ArbFlow',
    'result':{
        '1':{
            'train':{'pred':np.ndarray,'truth':np.ndarray},
            'test':{'pred':np.ndarray,'truth':np.ndarray}
        },
        '2':{
            'train':{'pred':np.ndarray,'truth':np.ndarray},
            'test':{'pred':np.ndarray,'truth':np.ndarray}
        },
        '4':{
            'train':{'pred':np.ndarray,'truth':np.ndarray},
            'test':{'pred':np.ndarray,'truth':np.ndarray}
        }...

    }
}
'''

parser = argparse.ArgumentParser()
parser.add_argument('--origin_lat',type=float,default=40.70004)
parser.add_argument('--origin_lng',type=float,default= -74.02409)
parser.add_argument('--resolution',type=float,default=0.15)
parser.add_argument('--result_filename',type=str,default='result/ST_ResNet_enhanced_Taxi_NYC')
parser.add_argument('--output_filename',type=str,default='test')
parser.add_argument('--output_dir',type=str,default='.')
parser.add_argument('--data_dir',type=str,default='.')
parser.add_argument('--setting',type=str,default='scale_setting')
parser.add_argument('--regular',type=str,default='regular_file')
parser.add_argument('--scalelist_filename',type=str,default='scale_structure')
parser.add_argument('--M_num',type=int,default=128)
parser.add_argument('--N_num',type=int,default=128)

args = parser.parse_args() 
args = vars(args)

with open(args['regular']+'.yml', 'r') as f:
    args.update(yaml.safe_load(f))

with open(args['result_filename']+'.pkl','rb') as fp:
    data = pkl.load(fp)

with open(args['setting']+'.yml','r') as f:
    args.update(yaml.safe_load(f))
    
scale_list = args['scale_list']
print(data.keys())



# 和同样的 ArbQuery 真实值比， 和同样的标准Query真实值比
values = []
for data_path in args['task_4']:
    print(data_path)
    with open(data_path,'rb') as fp:
        cover_region = pkl.load(fp)

if data['type'] == 'multi_scale':

    arbquery = ArbQuery(args['M_num'],args['N_num'])

    if 'test' not in data['result']['1'].keys():
    	for level in data['result'].keys():
            data['result'][level]['test'] = {
    		}
            data['result'][level]['test']['truth']  = data['result'][level]['truth']
            data['result'][level]['test']['pred']  = data['result'][level]['pred']
    structure_list = [1]
    
    qid = QueryIndexDict(scale_list)
    qid.construct_tree(data,keyword='train')

    for scale in scale_list:
        structure_list.append(structure_list[-1]*scale)


    ScaleList = [(args['M_num'],args['N_num'])]



    for i in range(len(structure_list)-1,-1,-1):
        ScaleList.append((structure_list[i],structure_list[i]))

    print(ScaleList)




    scale_dict = {'scale':ScaleList[0],'next':None}
    head = scale_dict
    for i in range(1,len(ScaleList)):
        head['next'] = {'scale':ScaleList[i],'next':None}
        head = head['next']

    print(scale_dict)
    import UCTB.evaluation.metric as metric
    multi_scale_prediction_combination = {}
    new_keys = []
    all_arbquery_pred = []
    all_arbquery_truth = []
    for key_,region in zip(cover_region.keys(),cover_region.values()):
        if len(region) == 0:
            continue
        t = arbquery.BigFirst(scale_dict=scale_dict,grid_obj=region,sign=1)
        # t_ = arbquery.Arbquery(scale_dict=scale_dict,grid_obj=region,sign=1)
        result = factor(t,structure_list)
        t_len = data['result']['1']['test']['truth'].shape[0]
        truth = np.zeros([t_len])
        pred = np.zeros([t_len])
        
        all_pred = np.zeros(t_len)
        all_truth = np.zeros(t_len)
        for base_level,object in zip(result.keys(),result.values()):
            queue = []
            if len(object)==0:
                continue
            if isinstance(object,list):
                queue.extend(object)
            else:
                for key,value in zip(object.keys(),object.values()):
                    if len(value) == 1:
                        queue.append(value[0])
                    else:
                        level,x,y = key
                        comb = [(item[1]-1,item[2]-1) for item in value]
                        op = value[0][-1]
                        code = comb2code(comb,2,x-1,y-1)
                        all_pred += op*calculate_value_from_path(data,qid.comb_dict[(level,x-1,y-1,code)]['path'],keyword='pred')
                        all_truth += op*calculate_value_from_path(data,qid.comb_dict[(level,x-1,y-1,code)]['path'],keyword='truth')
            while len(queue)!=0:
                scale,x_,y_,op = queue.pop(0)
                flag = qid.index_tree[(scale,x_-1,y_-1)]['op']
                if flag == 'output':
                    all_pred += op*calculate_value_from_path(data,qid.index_tree[(scale,x_-1,y_-1)]['path']) 
                    all_truth += op*calculate_value_from_path(data,qid.index_tree[(scale,x_-1,y_-1)]['path'],keyword='truth') 
                elif flag == 'split':
                    all_pred += op*calculate_value_from_path(data,qid.index_tree[(scale,x_-1,y_-1)]['path']) 
                    all_truth += op*calculate_value_from_path(data,qid.index_tree[(scale,x_-1,y_-1)]['path'],keyword='truth') 
                    
                else:
                    all_pred += op*calculate_value_from_path(data,qid.index_tree[(scale,x_-1,y_-1)]['path']) 
                    all_truth += op*calculate_value_from_path(data,qid.index_tree[(scale,x_-1,y_-1)]['path'],keyword='truth') 
                    
        all_arbquery_pred.append(all_pred)
        all_arbquery_truth.append(all_truth)
    all_arbquery_pred = np.array(all_arbquery_pred)
    all_arbquery_truth = np.array(all_arbquery_truth)

    print('ArbQuery test rmse',metric.rmse(all_arbquery_pred,all_arbquery_truth,threshold=4))
    print('ArbQuery test mae',metric.mae(all_arbquery_pred,all_arbquery_truth,threshold=4))
    print('ArbQuery test mape',metric.mape(all_arbquery_pred,all_arbquery_truth,threshold=150))
if data['type'] == 'single_scale':
    truth = []
    pred = []
    if 'test' not in data['result'].keys():
        data['result']['test'] = {}
        data['result']['test']['truth']  = data['result']['truth']
        data['result']['test']['pred']  = data['result']['pred']
    data['result']['test']['pred'][np.nonzero(data['result']['test']['pred']<0)] = 0
    for key,value in tqdm(zip(cover_region.keys(),cover_region.values())):
        if len(value) == 0:
            continue
        truth_single_region = np.zeros(data['result']['test']['truth'].shape[0])
        pred_single_region = np.zeros(data['result']['test']['truth'].shape[0])

        for x,y in value:
            truth_tmp = data['result']['test']['truth'][:,x-1,y-1,0]
            pred_tmp = data['result']['test']['pred'][:,x-1,y-1,0]
            truth_single_region += truth_tmp
            pred_single_region += pred_tmp

        truth.append(truth_single_region)
        pred.append(pred_single_region)

    truth = np.array(truth)
    pred = np.array(pred)
    print('{} test rmse'.format(args['result_filename']),metric.rmse(pred,truth,threshold=0))
    print('{} test mae'.format(args['result_filename']),metric.mae(pred,truth,threshold=0))
    print('{} test mape'.format(args['result_filename']),metric.mape(pred,truth,threshold=10))

