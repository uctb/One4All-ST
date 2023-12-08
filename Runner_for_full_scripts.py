import os
# 规整化
os.system('python regular.py --shpfilename processed_queries_census --output_dir regular --output_filename regular_census_NYC --id_column OBJECTID')
os.system('python regular.py --shpfilename sh_processed_queries_full --output_dir regular --output_filename regular_road_segment_full_SH')
os.system('python regular.py --shpfilename sh_processed_queries_second --output_dir regular --output_filename regular_road_segment_second_SH')
os.system('python regular.py --shpfilename sh_processed_queries_primary --output_dir regular --output_filename regular_road_segment_primary_SH')
# 训练ArbFlow模型 具体选项见exp.py
os.system('python exp.py')
# 结果测评
os.system('python test_query.py')