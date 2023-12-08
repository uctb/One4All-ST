import geopandas as gpd
import math
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
from ..evaluation import metric
import pickle as pkl

F = -1*np.ones([1000],dtype=np.int64)
def get_all_feasible_strucutre(max_scale,steps):
    paths = list()
    if max_scale<1:
        return None
    if max_scale==1:
        paths.append([1])
        return paths
    for step in steps:
        if max_scale%step != 0:
            continue
        tmp_result = get_all_feasible_strucutre(max_scale//step,steps)
        if tmp_result is None:
            continue
        for path in tmp_result:
            path.append(path[-1]*step)
        paths.extend(tmp_result)
    return paths

def get_scale_comb(paths):
    tmp_paths = list(map(lambda x:set(x),paths))
    scale_comb = set()
    for tmp in tmp_paths:
        scale_comb = scale_comb.union(tmp)
    scale_comb = list(scale_comb)
    scale_comb.sort()
    return scale_comb

def solver(n,coin,steps):
    if F[n] > 0:
        return F[n]
    if n==1:
        F[n] = coin[0]
        return F[n]
    # elif n==0:
    #     F[n] = -100000
    #     return F[n]
    else:
        tmp = []
        for step in steps:
            if n % step != 0:
                continue
            tmp.append(solver(n//step,coin,steps))
        F[n] = np.max(np.array(tmp))+coin[n-1]
        return F[n]

def output_structure(steps,n):
    pointer = n
    
    path = []
    while pointer != 1:
        tmp_arr = np.zeros([len(steps)])
        for ind,step in enumerate(steps):
            if pointer%step!=0:
                continue
            tmp_arr[ind] = F[pointer//step]
        path.append(steps[np.argmax(tmp_arr)])
        pointer = pointer//steps[np.argmax(tmp_arr)]
    path.reverse()
    return path
radius = 6371  # (km)

CONSTANTS_RADIUS_OF_EARTH = 6371000


class Regular():
    def __init__(self,M,N,origin,resolution,area_ratio) -> None:
        self.origin = origin
        self.M = M
        self.N = N
        self.resolution = resolution
        self.area_ratio = area_ratio
        pass
    def GPStoXY(self,lat, lon, ref_lat, ref_lon):
        # input GPS and Reference GPS in degrees
        # output XY in meters (m) X:North Y:East
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)

        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        ref_sin_lat = math.sin(ref_lat_rad)
        ref_cos_lat = math.cos(ref_lat_rad)

        cos_d_lon = math.cos(lon_rad - ref_lon_rad)

        arg = np.clip(ref_sin_lat * sin_lat + ref_cos_lat * cos_lat * cos_d_lon, -1.0, 1.0)
        c = math.acos(arg)

        k = 1.0
        if abs(c) > 0:
            k = (c / math.sin(c))

        x = float(k * (ref_cos_lat * sin_lat - ref_sin_lat * cos_lat * cos_d_lon) * CONSTANTS_RADIUS_OF_EARTH)
        y = float(k * cos_lat * math.sin(lon_rad - ref_lon_rad) * CONSTANTS_RADIUS_OF_EARTH)

        return round(x, 6), round(y, 6)

    def regular(self,gbf_obj, num_to_geo):
        # # 创建一个基本地图对象
        # m = folium.Map(location=[40, -74], zoom_start=13)
        vertices = list(gbf_obj.exterior.coords)
        vertices_ = [(i[1], i[0]) for i in vertices]

        polygon_points = vertices_
        # 在地图上绘制多边形

        geo_list = []
        min_x = self.M
        min_y = self.N
        max_x = 0
        max_y = 0
        for k in vertices_:
            geo_list.append(self.GPStoXY(self.origin[0],self.origin[1], k[0], k[1]))
            tmp_x,tmp_y = self.getxy((k[0],k[1]))
            if tmp_x < min_x:
                min_x = tmp_x
            if tmp_x > max_x:
                max_x = tmp_x
            if tmp_y < min_y:
                min_y = tmp_y
            if tmp_y > max_y:
                max_y = tmp_y
        geo_ = Polygon(geo_list)

        square_arealist = []
        medarea = []
        # 放入所有底图的小方格坐标
        count = 0
        for i in range(max(min_x-1,0),min(max_x+1,self.N)):
            for j in range(max(min_y-1,0),min(max_y+1,self.M)):
                medarea.append(num_to_geo[(i, j)])
                medarea.append(num_to_geo[(i + 1, j)])
                medarea.append(num_to_geo[(i, j + 1)])
                medarea.append(num_to_geo[(i + 1, j + 1)])
                square_arealist.append(medarea.copy())
                count = count + 1
                medarea.clear()
        coveredlist = []
        for k in square_arealist:
            ip = []
            ip.append(self.GPStoXY(self.origin[0],self.origin[1], k[0][0], k[0][1]))
            ip.append(self.GPStoXY(self.origin[0],self.origin[1], k[1][0], k[1][1]))
            ip.append(self.GPStoXY(self.origin[0],self.origin[1], k[3][0], k[3][1]))
            ip.append(self.GPStoXY(self.origin[0],self.origin[1], k[2][0], k[2][1]))

            # ip为多边形顶点
            square_polygon = Polygon(ip)
            intersect = geo_.intersection(square_polygon)
            # 计算交集部分面积
            intersection_area = intersect.area
            # 计算正方形面积
            square_area = square_polygon.area

            # 计算面数据覆盖正方形的面积比例
            overlap_ratio = intersection_area / square_area
            if (overlap_ratio > self.area_ratio):
                coveredlist.append(k)
        
        return coveredlist
        #给原点坐标，给出序号，尺度固定为150m,计算（x,y）处的经纬度坐标。x代表x *150远的地方。y代表y *150远的地方。
    def getlatlng(self, x, y):
        '''
        origin:(lat,lng)
        dest:(lat,lng)
        resolution in(km)
        '''
        resolution = self.resolution


        lat_d, lng_d = self.origin

        lat_d = lat_d / 180 * math.pi
        lng_d = lng_d / 180 * math.pi

        lat_o = y * resolution / radius + lat_d
        lng_o = lng_d + x * resolution / radius / math.cos(lat_d)
        lat_o = lat_o * 180 / math.pi
        lng_o = lng_o * 180 / math.pi
        return (lat_o, lng_o)
    def getxy(self,dest):
        '''
        origin:(lat,lng)
        dest:(lat,lng)
        resolution in(km)
        '''
        resolution = self.resolution

        lat_d, lng_d = dest
        lat_o, lng_o = self.origin
        lat_d = lat_d / 180*math.pi
        lng_d = lng_d / 180*math.pi
        lat_o = lat_o / 180*math.pi
        lng_o = lng_o / 180*math.pi

        y = int((lat_d-lat_o)*radius//resolution)
        x = int((lng_d-lng_o)*math.cos(lat_o)*radius//resolution)
        
        return (x,y)
    #地理坐标列表转序号列表
    def geolist_to_gridnum(self,geolist, Geo_to_num):
        numlist = []
        for i in geolist:
            numlist.append(Geo_to_num[i[3]])
        return numlist
    def irregular_request(self, shapefile_path,id_column):
        
        num_to_geo = {}
        for i in range(self.M + 1):
            for j in range(self.N + 1):
                num_to_geo[(i, j)] = self.getlatlng(i, j)
        Geo_to_num = {val: key for key, val in num_to_geo.items()}

        data = gpd.read_file('{}'.format(shapefile_path))
        cover_region = {}
        print(data.columns)
        for index, row in tqdm(data.iterrows()):
            if row['geometry'].type=='MultiPolygon':
                for p in row['geometry']:
                    cover_list = self.regular(p, num_to_geo)
                    regular_grid = self.geolist_to_gridnum(cover_list, Geo_to_num)
                    cover_region[index+10000] = regular_grid
            else:
                cover_list = self.regular(row['geometry'], num_to_geo)
                regular_grid = self.geolist_to_gridnum(cover_list, Geo_to_num)
                cover_region[index] = regular_grid
        return cover_region
        
class ArbQuery:  # 对于类的定义我们要求首字母大写

    def __init__(self, M_, N_):  # 初始化类的属性
        self.M_ = M_
        self.N_ = N_
        self.layer = 1
        self.record_process = []
        #M_,N_分别代表底图长宽，这里是128 和128

    #判断，当前一系列坐标序号，是否都在self.M_ / M尺度下的同一个区域中，是则返回在哪一个区域中。
    def is_together(self,set_grid, M):
        # set_grid是序号列表
        first = True
        for i in set_grid:
            if first:
                first = False
                origin = (int((i[0]-1)/M+1),int((i[1]-1)/M+1))
            else:
                if (int((i[0]-1)/M+1),int((i[1]-1)/M+1)) != origin:
                    return 0,-1
        return 1, origin

    #寻找当前大小的尺度在Scalelist哪一个位置。
    def find_i(self,x, Scalelist):
        cal = 0
        for i in Scalelist:
            if (x == i[0]):
                return cal
            cal = cal + 1
        return -1

    #区域序号的减集。
    def subtract_region(self,region1, region2):
        set_A = set(region1)
        set_B = set(region2)
        result = set_A - set_B
        return list(result)


    # 序号转坐标点
    def num_to_point(self,region):
        new_region = []
        medi = []
        for i in region:
            medi.append((i[0], i[1]))
            medi.append((i[0] - 1, i[1]))
            medi.append((i[0] - 1, i[1] - 1))
            medi.append((i[0], i[1] - 1))
            new_region.append(medi.copy())
            medi.clear()
        return new_region

    #该M尺度下的一个序号，对应的底图上最小方格的序号列表。
    def Scale_to_point(self,orgin, M):
        x = orgin[0]
        y = orgin[1]
        new_region = []
        for m in range(M * (x - 1) + 1, M * x + 1):
            for n in range(M * (y - 1) + 1, M * y + 1):
                new_region.append((m, n))
        return new_region
    
   
        
        
    #当前区域与Scale尺度下的交集，返回一系列交集
    def divide(self,region, Scale):

        Q_dict = {}

        for i in region:
            # 如果对应大尺度的对象不在字典内
            if (int((i[0]-1)/Scale)+1,int((i[1]-1)/Scale)+1) not in Q_dict.keys():
                Q_dict[(int((i[0]-1)/Scale)+1,int((i[1]-1)/Scale)+1)] = [i]
            else:
                Q_dict[(int((i[0]-1)/Scale)+1,int((i[1]-1)/Scale)+1)].append(i)

        return Q_dict
    
    #由序号计算是第几个站点
    def XY_tonum(self,i, j,m,n):
        return (n-j)*m+i-1
    
    # 使用scale尺度下第num个模型预测该区域当前流量。
    def F(self,scale,num):
        testpredict=self.modellist[scale][num].predict(time_sequences=self.test_closeness[scale][:, num, :, 0],forecast_step=1)
        return testpredict
        
        
    #Arbquery50%准则 拼接方法
    def Arbquery(self,scale_dict, grid_obj, sign):
        result = []
        if len(grid_obj)==0:
            return result
        # 这里的grid_obj是序号列表哦，（1,1）开始。
        if scale_dict is None:
            return []
        i = scale_dict['scale']
        # 是否在i这个尺度里。
        is_together_, _y = self.is_together(grid_obj, i[0])
        # 判断连环语句，找到下确界
        if (is_together_ != 1):
            Qlist = self.divide(grid_obj, scale_dict['next']['scale'][0])
            for s in Qlist:
                result.extend(self.Arbquery(scale_dict['next'], s, sign))
            return result
        if (is_together_ == 1):
            if (len(grid_obj) == i[0] ** 2):
                # F[i][a][b]为i尺度下，（a,b）处的流量值
                result.append({'scale':i[0],'coordinate':_y,'op':sign})
                return result
                
            if (len(grid_obj) / (i[0] ** 2) > 0.5):
                result.append({'scale':i[0],'coordinate':_y,'op':sign})
                result.extend(self.Arbquery(scale_dict,self.subtract_region(self.Scale_to_point(_y, i[0]),grid_obj),-sign))
                return result
            else:
                # if (self.find_i(i[0], Scalelist) >= len(Scalelist) - 1):
                #     return len(grid_obj) / (i[0] ** 2) * self.F(self.find_i(i[0], Scalelist),self.XY_tonum(_y[0],_y[1],i[0],i[1]))
                Qlist = self.divide(grid_obj, scale_dict['next']['scale'][0])
                for s in Qlist:
                    result.extend(self.Arbquery(scale_dict['next'], s, sign))
                return result
    def BigFirst(self,scale_dict, grid_obj, sign):
        result = []
        # step = {}
        # step['layer'] = self.layer


        if len(grid_obj)==0:
            return result
        # 这里的grid_obj是序号列表，（1,1）开始。
        if scale_dict is None:
    
            return []
        i = scale_dict['scale']
        # 是否在i这个尺度里。
        is_together_, _y = self.is_together(grid_obj, i[0])
        # 判断连环语句，找到下确界
        if (is_together_ != 1):
            Q_dict = self.divide(grid_obj, scale_dict['next']['scale'][0])
            for s in Q_dict.values():
                # self.layer += 1
                result.extend(self.BigFirst(scale_dict['next'], s, sign))
                # self.layer -= 1

            return result
        if (is_together_ == 1):
            if (len(grid_obj) == i[0] ** 2):
                # F[i][a][b]为i尺度下，（a,b）处的流量值
                # step['op'] = 'final'
                result.append({'scale':i[0],'coordinate':_y,'op':sign})
                # step['before'] = grid_obj
                # step['after'] = [_y]
                # step['result'] = {'scale':i[0],'coordinate':_y,'op':sign}
                # self.record_process.append(step)
                return result
                

            else:
                # step['op'] = 'divide'
                Q_dict = self.divide(grid_obj, scale_dict['next']['scale'][0])
                # step['before'] = grid_obj
                # step['after'] = Q_dict.values()
                # step['result'] = [{'scale':scale_dict['next']['scale'][0],'coordinate':coord,'op':sign} for coord in Q_dict.keys()]
                # if len(Q_dict) >= 1:
                #     self.record_process.append(step)
                for s in Q_dict.values():
                    self.layer += 1
                    result.extend(self.BigFirst(scale_dict['next'], s, sign))
                    self.layer -= 1
                return result

def save_predict_result(data_loader,mark,time_length,output_type,test_prediction,train_prediction,scale_list=None,new_width=144,new_height=144,padding = False):
    if output_type == 'multi_scale':
        ind = 0
        scale = 1
        result = {}
        result['model_name'] = mark
        result['type'] = output_type
        result['result'] = {}
        print(len(scale_list))
        print(len(data_loader.multi_level_truth_test))
        print(len(data_loader.normalizers))
        for normalizer,test_truth,train_truth in zip(data_loader.normalizers,data_loader.multi_level_truth_test,data_loader.multi_level_truth_train):
            
            test_pred = normalizer.inverse_transform(test_prediction['prediction_{}'.format(ind)])
            test_pred[np.nonzero(test_pred<0)] = 0
            train_pred = normalizer.inverse_transform(train_prediction['prediction_{}'.format(ind)])
            train_pred[np.nonzero(test_pred<0)] = 0
            test_truth = normalizer.inverse_transform(test_truth)
            train_truth = normalizer.inverse_transform(train_truth)
            
            train_pred = train_pred.astype(np.float32)
            test_pred = test_pred.astype(np.float32)
            train_truth = train_truth.astype(np.float32)
            test_truth = test_truth.astype(np.float32)
            
            test_rmse = metric.rmse(prediction=test_pred,
                                target=test_truth, threshold=4)
            test_mae = metric.mae(prediction=test_pred,
                                target=test_truth, threshold=4)
            test_mape = metric.mape(prediction=test_pred,
                                target=test_truth, threshold=4)
            result['result'][str(scale)]={}
            result['result'][str(scale)]['train'] = {
                'pred':train_pred,
                'truth':train_truth
            }
            result['result'][str(scale)]['test'] = {
                'pred':test_pred,
                'truth':test_truth
            }
            print('Scale ',scale)
            print('Test RMSE ',test_rmse)
            print('Test MAE ',test_mae)
            print('Test MAPE ',test_mape)
            if ind >= len(scale_list):
                break
            scale *= scale_list[ind]
            ind += 1
        return result
    elif output_type == 'single_scale':
        result = {}
        result['model_name'] = mark
        result['time_length'] = time_length
        result['type'] = output_type
        normalizer = data_loader.normalizer
        test_truth = data_loader.test_y
        test_pred = normalizer.inverse_transform(test_prediction['prediction'])
        test_pred[np.nonzero(test_pred<0)] = 0
        test_truth = normalizer.inverse_transform(test_truth)
        train_truth = data_loader.train_y
        train_pred = normalizer.inverse_transform(train_prediction['prediction'])
        train_pred[np.nonzero(train_pred<0)] = 0
        train_truth = normalizer.inverse_transform(train_truth)
        if padding:
            # prediction padding
            data = pred
            if len(data.shape) == 3:
                t_len,height,width = data.shape
                new_data = np.zeros([t_len,new_height,new_width])
                new_data[:,:height,:width] = data

            else:
                t_len,height,width,_ = data.shape
                new_data = np.zeros([t_len,new_height,new_width,1])
                new_data[:,:height,:width,:] = data

            pred = new_data
            # truth padding
            data = truth
            if len(data.shape) == 3:
                t_len,height,width = data.shape
                new_data = np.zeros([t_len,new_height,new_width])
                new_data[:,:height,:width] = data

            else:
                t_len,height,width,_ = data.shape
                new_data = np.zeros([t_len,new_height,new_width,1])
                new_data[:,:height,:width,:] = data
            truth = new_data

        result['result'] = {}
        result['result']['train'] = {}
        result['result']['train']['pred'] = train_pred
        result['result']['train']['truth'] = train_truth
        result['result']['test'] = {}
        result['result']['test']['pred'] = test_pred
        result['result']['test']['truth'] = test_truth
        
        test_rmse = metric.rmse(prediction=test_pred,
                                target=test_truth, threshold=4)
        test_mae = metric.mae(prediction=test_pred,
                            target=test_truth, threshold=4)
        test_mape = metric.mape(prediction=test_pred,
                            target=test_truth, threshold=4)
        
        print('Test RMSE ',test_rmse)
        print('Test MAE ',test_mae)
        print('Test MAPE ',test_mape)
        return result

def coordinate_convert_to_target_level(coordinate,current_level,origin,target_level):
    x,y = coordinate
    ox,oy = origin
    
    if target_level>current_level:
    
        scale = target_level//current_level
        target_x = (x-ox)//scale+ox
        target_y = (y-oy)//scale+oy

        return (target_x,target_y)
    
    else:
    
        scale = current_level//target_level
        target_x = (x-ox)*scale+ox
        target_y = (y-oy)*scale+oy
        
        return [(target_x+t,target_y+k) for t in range(scale) for k in range(scale)]


def factor(result,scale_candidate_list):
        factor_result = dict(zip(scale_candidate_list[:-1],[{} for i in range(len(scale_candidate_list)-1)]))
        factor_result[scale_candidate_list[-1]] = []

        for item in result:
            level = item['scale']
            x,y = item['coordinate']
            op = item['op']
            ind = scale_candidate_list.index(level)
            if ind != len(scale_candidate_list)-1:
                upper_level = scale_candidate_list[ind+1]
                upper_x,upper_y = coordinate_convert_to_target_level((x,y),level,(1,1),upper_level)
                if (upper_level,upper_x,upper_y) in factor_result[level].keys():
                    factor_result[level][(upper_level,upper_x,upper_y)].append((level,x,y,op))
                else:
                    factor_result[level][(upper_level,upper_x,upper_y)]=[(level,x,y,op)]
            else:
                factor_result[level].append((level,x,y,op))
        return factor_result
class QueryIndexDict():
	def __init__(self,scale_list) -> None:
		self.scale_list = scale_list
	def construct_tree(self,metrics,keyword):
		index_tree = {}
		level = 1
		base_pred = np.sum(metrics['result'][str(level)][keyword]['pred'],axis=-1)
		
		height = base_pred.shape[1]
		width = base_pred.shape[2]

		# record_dict = {}

		for i in range(height):
			for j in range(width):
				index_tree[(level,i,j)] = {}
				index_tree[(level,i,j)]['value'] = base_pred[:,i,j]
				index_tree[(level,i,j)]['op'] = 'output'
				index_tree[(level,i,j)]['path'] = [(level,i,j,1)]
		
		for scale in self.scale_list:
			base_truth = np.sum(metrics['result'][str(level)][keyword]['truth'],axis=-1)
			level *= scale
			pred = np.sum(metrics['result'][str(level)][keyword]['pred'],axis=-1)
			truth = np.sum(metrics['result'][str(level)][keyword]['truth'],axis=-1)
			t_len = pred.shape[0]
			height = pred.shape[1]
			width = pred.shape[2]
			new_base_pred = np.empty([t_len,height,width])
			for i in range(height):
				for j in range(width):
					merge_pred = np.sum(np.sum(base_pred[:,scale*i:scale*(i+1),scale*j:scale*(j+1)],axis=1),axis=1)
					merge_truth = np.sum(np.sum(base_truth[:,scale*i:scale*(i+1),scale*j:scale*(j+1)],axis=1),axis=1)
					error_merge = metric.mae(merge_pred,merge_truth)
					error = metric.mae(pred[:,i,j],merge_truth)
					if error_merge < error:
						new_base_pred[:,i,j] = merge_pred
						index_tree[(level,i,j)] = {}
						index_tree[(level,i,j)]['value'] = merge_pred
						path = []
						for t in range(scale):
							for k in range(scale):
								path.extend(index_tree[(level//scale,i*scale+t,j*scale+k)]['path'])
						index_tree[(level,i,j)]['path'] = path
						index_tree[(level,i,j)]['op'] = 'split'
					else:
						new_base_pred[:,i,j] = pred[:,i,j]
						index_tree[(level,i,j)] = {}
						index_tree[(level,i,j)]['value'] = pred[:,i,j]
						index_tree[(level,i,j)]['path'] = [(level,i,j,1)]
						index_tree[(level,i,j)]['op'] = 'output'

			base_pred = new_base_pred
		level = 1

		comb_dict ={}
			
		for scale in self.scale_list:
			level *= scale
			pred = np.sum(metrics['result'][str(level)][keyword]['pred'],axis=-1)
			t_len = pred.shape[0]
			
			height = pred.shape[1]
			width = pred.shape[2]
			for i in range(height):
				for j in range(width):
					for t in range(1,15):
						truth = np.zeros([t_len])
						comb = code2comb(scale,i,j,t)
						for x,y in comb:
							truth += metrics['result'][str(level//scale)][keyword]['truth'][:,x,y,0]
						pred,path = calculate_comb(index_tree,scale,level,i,j,t,t_len)
						pred_subtract,path_subtract = calculate_comb(index_tree,scale,level,i,j,-t,t_len)
						comb_dict[(level,i,j,t)] = {}
						if metric.mae(pred_subtract,truth) < metric.mae(pred,truth):
							comb_dict[(level,i,j,t)]['op'] = 'comb_subtract'
							comb_dict[(level,i,j,t)]['path'] = path_subtract
							comb_dict[(level,i,j,t)]['value'] = pred_subtract
							# print('comb_subtract')
						else:
							comb_dict[(level,i,j,t)]['op'] = 'comb'
							comb_dict[(level,i,j,t)]['value'] = pred
							comb_dict[(level,i,j,t)]['path'] = path
							# print('comb')
		self.index_tree = index_tree
		self.comb_dict = comb_dict
			
		pass

from copy import deepcopy

def calculate_subtract(index_tree,scale,level,i,j):
	upper_i = i//scale
	upper_j = j//scale
	upper_op = (scale*level,upper_i,upper_j)
	op_set = set([(level,upper_i*scale+t,upper_j*scale+k) for t in range(scale) for k in range(scale)])
	op_set = op_set.difference(set([(level,i,j)]))
	result = deepcopy(index_tree[upper_op]['path'])
	for op in op_set:
		new_op = []
		for item in index_tree[op]['path']:
			tmp_level,tmp_i,tmp_j,tmp_flag = item
			new_op.append((tmp_level,tmp_i,tmp_j,-tmp_flag))
		result.extend(new_op)
	return (index_tree[(scale*level,upper_i,upper_j)]['value'] - sum([index_tree[(level,upper_i*scale+t,upper_j*scale+k)]['value'] for t in range(scale) for k in range(scale)]) + index_tree[(level,i,j)]['value'],result)

def comb2code(comb,scale,i,j):
	if len(comb) == 2:
		if (i*scale,j*scale) in comb and (i*scale,j*scale+1) in comb:
			return 1
		if (i*scale,j*scale) in comb and (i*scale+1,j*scale) in comb:
			return 2
		if (i*scale,j*scale) in comb and (i*scale+1,j*scale+1) in comb:
			return 3
		if (i*scale,j*scale+1) in comb and (i*scale+1,j*scale+1) in comb:
			return 4
		if (i*scale+1,j*scale) in comb and (i*scale+1,j*scale+1) in comb:
			return 5
		if (i*scale+1,j*scale) in comb and (i*scale,j*scale+1) in comb:
			return 6
	if len(comb) == 3:
		if (i*scale,j*scale) in comb and (i*scale+1,j*scale) in comb and (i*scale,j*scale+1) in comb:
			return 7
		if (i*scale+1,j*scale) in comb and (i*scale+1,j*scale+1) in comb and (i*scale,j*scale+1) in comb:
			return 8
		if (i*scale,j*scale) in comb and (i*scale+1,j*scale+1) in comb and (i*scale,j*scale+1) in comb:
			return 9
		if (i*scale,j*scale) in comb and (i*scale+1,j*scale) in comb and (i*scale+1,j*scale+1) in comb:
			return 10
	if len(comb) == 1:
		if (i*scale+1,j*scale+1) in comb:
			return 11
		if (i*scale,j*scale) in comb:
			return 12
		if (i*scale+1,j*scale) in comb:
			return 13
		if (i*scale,j*scale+1) in comb:
			return 14
def code2comb(scale,i,j,code):
	if code == 1:
		comb = [(i*scale,j*scale),(i*scale,j*scale+1)]
	elif code == 2:
		comb = [(i*scale,j*scale),(i*scale+1,j*scale)]
	elif code == 3:
		comb = [(i*scale,j*scale),(i*scale+1,j*scale+1)]
	elif code == 4:
		comb = [(i*scale,j*scale+1),(i*scale+1,j*scale+1)]
	elif code == 5:
		comb = [(i*scale+1,j*scale),(i*scale+1,j*scale+1)]
	elif code == 6:
		comb = [(i*scale+1,j*scale),(i*scale,j*scale+1)]
	elif code == 7:
		comb = [(i*scale,j*scale),(i*scale+1,j*scale),(i*scale,j*scale+1)]
	elif code == 8:
		comb = [(i*scale+1,j*scale),(i*scale+1,j*scale+1),(i*scale,j*scale+1)]
	elif code == 9:
		comb = [(i*scale,j*scale),(i*scale+1,j*scale+1),(i*scale,j*scale+1)]
	elif code == 10:
		comb = [(i*scale,j*scale),(i*scale+1,j*scale),(i*scale+1,j*scale+1)]
	elif code == 11:
		comb = [(i*scale+1,j*scale+1)]
	elif code == 12:
		comb = [(i*scale,j*scale)]
	elif code == 13:
		comb = [(i*scale+1,j*scale)]
	elif code == 14:
		comb = [(i*scale,j*scale+1)]
	return comb


def calculate_comb(index_tree,scale,level,i,j,type,t_len):
	
	if type > 0:
		comb = code2comb(scale,i,j,type)

		pred = np.zeros([t_len])
		path = []
		for x,y in comb:
			pred += index_tree[(level//scale,x,y)]['value']
			path.extend(index_tree[(level//scale,x,y)]['path'])
		return pred,path
	if type < 0:
		comb = code2comb(scale,i,j,-type)

		full_set = set([(i*scale,j*scale),(i*scale+1,j*scale),(i*scale,j*scale+1),(i*scale+1,j*scale+1)])
		new_set = full_set.difference(comb)
		comb = list(new_set)
		pred = deepcopy(index_tree[(level,i,j)]['value'])
		path = deepcopy(index_tree[(level,i,j)]['path'])
		for x,y in comb:
			pred -= index_tree[(level//scale,x,y)]['value']
			new_op = []
			for item in index_tree[(level//scale,x,y)]['path']:
				tmp_level,tmp_i,tmp_j,tmp_flag = item
				new_op.append((tmp_level,tmp_i,tmp_j,-tmp_flag))
			path.extend(new_op)
		return pred,path
def calculate_value_from_path(data,path,keyword='pred'):
	result = np.zeros(data['result']['1']['test'][keyword].shape[0])
	for item in path:
		level,x,y,op = item
		result+=op*data['result'][str(level)]['test'][keyword][:,x,y,0]
	return result