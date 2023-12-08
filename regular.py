import argparse
import os
from UCTB.utils.utils_ArbFlow import Regular
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--origin_lat',type=float,default=31.1489)
parser.add_argument('--origin_lng',type=float,default=121.377)
parser.add_argument('--resolution',type=float,default=0.15)
parser.add_argument('--shpfile_dir',type=str,default='shp/processed')
parser.add_argument('--shpfilename',type=str,default='sh_processed_queries_hexagon')
parser.add_argument('--M_num',type=int,default=128)
parser.add_argument('--N_num',type=int,default=128)
parser.add_argument('--output_filename',type=str,default='regular_hexagon')
parser.add_argument('--output_dir',type=str,default='.')
parser.add_argument('--id_column',type=str,default='fid')


args = parser.parse_args()
args = vars(args)


r = Regular(M=args['M_num'],N=args['N_num'],origin=(args['origin_lat'],args['origin_lng']),resolution=args['resolution'],area_ratio=0.5)
regular_result = r.irregular_request(os.path.join(args['shpfile_dir'],args['shpfilename']+'.shp'),args['id_column'])
with open(os.path.join(args['output_dir'],args['output_filename']+'.pkl'),'wb') as fp:
    pkl.dump(regular_result,fp)