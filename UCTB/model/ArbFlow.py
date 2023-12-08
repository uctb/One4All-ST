import tensorflow as tf

from ..model_unit import BaseModel


class ArbFlowDev(BaseModel):
    """ST-ResNet is a deep-learning model with an end-to-end structure
    based on unique properties of spatio-temporal data making use of convolution and residual units.

    References:
        - `Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction (Junbo Zhang et al., 2016)
          <https://arxiv.org/pdf/1610.00081.pdf>`_.
        - `Github repository (lucktroy)
          <https://github.com/lucktroy/DeepST/tree/master/scripts/papers/AAAI17>`_.
    Args:
        width (int): The width of grid data.
        height (int): The height of grid data.
        externai_dim (int): Number of dimensions of external data.
        closeness_len (int): The length of closeness data history. The former consecutive ``closeness_len`` time slots
            of data will be used as closeness history.
        period_len (int): The length of period data history. The data of exact same time slots in former consecutive
            ``period_len`` days will be used as period history.
        trend_len (int): The length of trend data history. The data of exact same time slots in former consecutive
            ``trend_len`` weeks (every seven days) will be used as trend history.
        num_residual_unit (int): Number of residual units. Default: 4
        kernel_size (int): Kernel size in Convolutional neural networks. Default: 3
        lr (float): Learning rate. Default: 1e-5
        code_version (str): Current version of this model code.
        model_dir (str): The directory to store model files. Default:'model_dir'
        conv_filters (int):  the Number of filters in the convolution. Default: 64
        gpu_device (str): To specify the GPU to use. Default: '0'
    """
    def __init__(self,
                 width,
                 height,
                 external_dim,
                 closeness_len,
                 period_len,
                 trend_len,
                 num_residual_unit=4,
                 #!--------new parameters----------------
                 scale=2,
                 num_scale=1,
                 multi_weight=1,
                 fusion_method=1,
                 #!--------------------------------------
                 kernel_size=3,
                 lr=5e-5,
                 model_dir='model_dir',
                 code_version='QuickStart',
                 conv_filters=64,
                 gpu_device='0',is_mask='False'):

        super(ArbFlowDev, self).__init__(code_version=code_version, model_dir=model_dir, gpu_device=gpu_device)
        
        self._width = width
        self._height = height
        self._closeness_len = closeness_len
        self._period_len = period_len
        self._trend_len = trend_len
        self._conv_filters = conv_filters
        self._kernel_size = kernel_size
        self._external_dim = external_dim
        self._scale = scale
        self._num_scale = num_scale
        self._num_residual_unit = num_residual_unit
        self._lr = lr
        self._is_mask = is_mask
        self._multi_weight = multi_weight
        self._fusion_method = fusion_method

    def build(self):
        with self._graph.as_default():

            target_conf = []

            if self._closeness_len is not None and self._closeness_len > 0:
                c_conf = tf.placeholder(tf.float32, [None, self._height, self._width, self._closeness_len], name='c')
                self._input['closeness_feature'] = c_conf.name
                target_conf.append(c_conf)

            if self._period_len is not None and self._period_len > 0:
                p_conf = tf.placeholder(tf.float32, [None, self._height, self._width, self._period_len], name='p')
                self._input['period_feature'] = p_conf.name
                target_conf.append(p_conf)

            if self._trend_len is not None and self._trend_len > 0:
                t_conf = tf.placeholder(tf.float32, [None, self._height, self._width, self._trend_len], name='t')
                self._input['trend_feature'] = t_conf.name
                target_conf.append(t_conf)
            if self._is_mask:
                mask = tf.placeholder(tf.float32, [None, self._height, self._width,1], name='mask')
                self._input['mask'] = mask.name
            target = tf.placeholder(tf.float32, [None, self._height, self._width, 1], name='target')
            self._input['target'] = target.name
            multi_scale_target = [target]
            for s in range(self._num_scale):
                filters = tf.ones([self._scale,self._scale,1,1])
                multi_scale_target.append(tf.nn.conv2d(multi_scale_target[-1],filters,strides=[1,self._scale,self._scale,1],padding='SAME',data_format='NHWC'))

            outputs = []
            for conf in target_conf:

                residual_input = tf.layers.conv2d(conf, filters=self._conv_filters,
                                                  kernel_size=[self._kernel_size, self._kernel_size],
                                                  padding='SAME', activation=tf.nn.relu)

                def residual_unit(x):
                    residual_output = tf.nn.relu(x)
                    residual_output = tf.layers.conv2d(residual_output, filters=self._conv_filters,
                                                       kernel_size=[self._kernel_size, self._kernel_size], padding='SAME')
                    residual_output = tf.nn.relu(residual_output)
                    residual_output = tf.layers.conv2d(residual_output, filters=self._conv_filters,
                                                       kernel_size=[self._kernel_size, self._kernel_size], padding='SAME')
                    return residual_output + x

                for i in range(self._num_residual_unit):
                    residual_input = residual_unit(residual_input)

                outputs.append(tf.layers.conv2d(tf.nn.relu(residual_input), filters=self._conv_filters,
                                                kernel_size=[self._kernel_size, self._kernel_size], padding='SAME'))
            
            if len(outputs) == 1:
                x = outputs[0]
            else:
                fusion_weight = tf.Variable(tf.random_normal([len(outputs), ]))
                for i in range(len(outputs)):
                    outputs[i] = fusion_weight[i] * outputs[i]
                x = tf.reduce_sum(outputs, axis=0)
            #! followings are a way of current ArbFlowDev framework's implementation
            #* 构建多尺度特征图
            feature_pyramid = [x]
            scale_factors = [1]
            #* bottom-up
            for s in range(self._num_scale):
                # patch merge
                upper_feat = tf.layers.conv2d(feature_pyramid[-1], filters=self._conv_filters,
                                                kernel_size=[self._scale, self._scale],
                                                strides=[self._scale,self._scale], 
                                                activation=tf.nn.relu,
                                                padding='SAME')
                
                for i in range(self._num_residual_unit):
                    upper_feat = residual_unit(upper_feat)

                feature_pyramid.append(upper_feat)
                scale_factors.append(scale_factors[-1]*self._scale*self._scale)
            
            #* 尺度间建模
            def inter_modeling(x,scale):
                return tf.keras.layers.UpSampling2D(size=(scale, scale),data_format="channels_last")(x)
            
            #* 融合
            def fusion(feature_map_up_sampled,feature_map,fusion_method):
                if fusion_method=='concat':
                    x = tf.concat([feature_map_up_sampled,feature_map],axis=-1)
                    return tf.layers.conv2d(x, filters=self._conv_filters,
                                                      kernel_size=[1, 1],
                                                      padding='SAME', activation=tf.nn.relu)
                elif fusion_method=='add':
                    return tf.add(feature_map_up_sampled,feature_map)
            
            #* top-down
            # 初始化loss
            losses = []
            for s in range(self._num_scale,0,-1):
                feature_pyramid[s-1] = fusion(inter_modeling(feature_pyramid[s], self._scale), feature_pyramid[s-1],fusion_method=self._fusion_method)
            
                # external dims
                if self._external_dim is not None and self._external_dim > 0:
                    external_input = tf.placeholder(tf.float32, [None, self._external_dim])
                    self._input['external_feature'] = external_input.name
                    external_dense = tf.layers.dense(inputs=external_input, units=10)
                    external_dense = tf.tile(tf.reshape(external_dense, [-1, 1, 1, 10]),
                                             [1, self._height, self._width, 1])
                    x = tf.concat([feature_pyramid[s], external_dense], axis=-1)
                else:
                    x = feature_pyramid[s]
                x = tf.layers.dense(x, units=1, activation=tf.nn.tanh)
                x = tf.multiply(x,scale_factors[s],name='prediction_{}'.format(s))
                if self._is_mask:
                    losses.append(tf.compat.v1.losses.mean_squared_error(multi_scale_target[s],x,weights=mask))
                else:
                    losses.append(tf.compat.v1.losses.mean_squared_error(multi_scale_target[s],x,weights=1))
                self._output['prediction_{}'.format(s)] = x.name
                
            # external dims
            if self._external_dim is not None and self._external_dim > 0:
                external_input = tf.placeholder(tf.float32, [None, self._external_dim])
                self._input['external_feature'] = external_input.name
                external_dense = tf.layers.dense(inputs=external_input, units=10)
                external_dense = tf.tile(tf.reshape(external_dense, [-1, 1, 1, 10]),
                                         [1, self._height, self._width, 1])
                x = tf.concat([feature_pyramid[0], external_dense], axis=-1)
            else:
                x = feature_pyramid[0]
            x = tf.layers.dense(x, units=1, name='prediction_{}'.format(0), activation=tf.nn.tanh)
            self._output['prediction_{}'.format(0)] = x.name
            if self._is_mask:
                losses.append(tf.compat.v1.losses.mean_squared_error(multi_scale_target[0],x,weights=mask))
            else:
                losses.append(tf.compat.v1.losses.mean_squared_error(multi_scale_target[0],x,weights=1))
            def multitask_loss(losess,scale_factors,method='prior'):
                loss_sum = 0
                if method=='adaptive':
                    weight = tf.Variable(tf.random_normal([len(losess), ]))
                    for ind,loss in enumerate(losess):
                        loss_sum += loss*tf.nn.sigmoid(weight[ind])
                elif method=='prior':
                    for loss,scale_factor in zip(losess,scale_factors):
                        loss_sum += loss*scale_factor
                return loss_sum
            loss = multitask_loss(losses,[0,1],method=self._multi_weight)
            train_op = tf.train.AdamOptimizer(self._lr).minimize(loss)

            self._output['loss'] = loss.name

            self._op['train_op'] = train_op.name

        super(ArbFlowDev, self).build()

    def _get_feed_dict(self, closeness_feature=None, period_feature=None, trend_feature=None,
                       target=None, external_feature=None,mask=None):
        '''
        The method to get feet dict for tensorflow model.

        Users may modify this method according to the format of input.

        Args:
            closeness_feature (np.ndarray or ``None``): The closeness history data.
                If type is np.ndarray, its shape is [time_slot_num, height, width, closeness_len].
            period_feature (np.ndarray or ``None``): The period history data.
                If type is np.ndarray, its shape is [time_slot_num, height, width, period_len].
            trend_feature (np.ndarray or ``None``): The trend history data.
                If type is np.ndarray, its shape is [time_slot_num, height, width, trend_len].
            target (np.ndarray or ``None``): The target value data.
                If type is np.ndarray, its shape is [time_slot_num, height, width, 1].
            external_feature (np.ndarray or ``None``): The external feature data.
                If type is np.ndaaray, its shape is [time_slot_num, feature_num].
        '''
        feed_dict = {}
        if target is not None:
            feed_dict['target'] = target
        if self._external_dim is not None and self._external_dim > 0:
            feed_dict['external_feature'] = external_feature
        if self._closeness_len is not None and self._closeness_len > 0:
            feed_dict['closeness_feature'] = closeness_feature
        if self._period_len is not None and self._period_len > 0:
            feed_dict['period_feature'] = period_feature
        if self._trend_len is not None and self._trend_len > 0:
            feed_dict['trend_feature'] = trend_feature
        if self._is_mask:
            feed_dict['mask'] = mask
        return feed_dict

class ArbFlow(BaseModel):
    """ST-ResNet is a deep-learning model with an end-to-end structure
    based on unique properties of spatio-temporal data making use of convolution and residual units.

    References:
        - `Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction (Junbo Zhang et al., 2016)
          <https://arxiv.org/pdf/1610.00081.pdf>`_.
        - `Github repository (lucktroy)
          <https://github.com/lucktroy/DeepST/tree/master/scripts/papers/AAAI17>`_.
    Args:
        width (int): The width of grid data.
        height (int): The height of grid data.
        externai_dim (int): Number of dimensions of external data.
        closeness_len (int): The length of closeness data history. The former consecutive ``closeness_len`` time slots
            of data will be used as closeness history.
        period_len (int): The length of period data history. The data of exact same time slots in former consecutive
            ``period_len`` days will be used as period history.
        trend_len (int): The length of trend data history. The data of exact same time slots in former consecutive
            ``trend_len`` weeks (every seven days) will be used as trend history.
        num_residual_unit (int): Number of residual units. Default: 4
        kernel_size (int): Kernel size in Convolutional neural networks. Default: 3
        lr (float): Learning rate. Default: 1e-5
        code_version (str): Current version of this model code.
        model_dir (str): The directory to store model files. Default:'model_dir'
        conv_filters (int):  the Number of filters in the convolution. Default: 64
        gpu_device (str): To specify the GPU to use. Default: '0'
    """
    def __init__(self,
                 width,
                 height,
                 external_dim,
                 closeness_len,
                 period_len,
                 trend_len,
                 num_residual_unit=4,
                 #!--------new parameters----------------
                 scale=2,
                 num_scale=1,
                 fusion_method='concat',
                 #!--------------------------------------
                 kernel_size=3,
                 lr=5e-5,
                 model_dir='model_dir',
                 code_version='QuickStart',
                 conv_filters=64,
                 gpu_device='0',is_mask='False'):

        super(ArbFlow, self).__init__(code_version=code_version, model_dir=model_dir, gpu_device=gpu_device)
        
        self._width = width
        self._height = height
        self._closeness_len = closeness_len
        self._period_len = period_len
        self._trend_len = trend_len
        self._conv_filters = conv_filters
        self._kernel_size = kernel_size
        self._external_dim = external_dim
        self._scale = scale
        self._num_scale = num_scale
        self._num_residual_unit = num_residual_unit
        self._lr = lr
        self._is_mask = is_mask
        self._fusion_method = fusion_method

    def build(self):
        with self._graph.as_default():

            target_conf = []

            if self._closeness_len is not None and self._closeness_len > 0:
                c_conf = tf.placeholder(tf.float32, [None, self._height, self._width, self._closeness_len], name='c')
                self._input['closeness_feature'] = c_conf.name
                target_conf.append(c_conf)

            if self._period_len is not None and self._period_len > 0:
                p_conf = tf.placeholder(tf.float32, [None, self._height, self._width, self._period_len], name='p')
                self._input['period_feature'] = p_conf.name
                target_conf.append(p_conf)

            if self._trend_len is not None and self._trend_len > 0:
                t_conf = tf.placeholder(tf.float32, [None, self._height, self._width, self._trend_len], name='t')
                self._input['trend_feature'] = t_conf.name
                target_conf.append(t_conf)
            if self._is_mask:
                mask = tf.placeholder(tf.float32, [None, self._height, self._width,1], name='mask')
                self._input['mask'] = mask.name
            target = tf.placeholder(tf.float32, [None, self._height, self._width, 1], name='target')

            self._input['target'] = target.name

            outputs = []
            for conf in target_conf:

                residual_input = tf.layers.conv2d(conf, filters=self._conv_filters,
                                                  kernel_size=[self._kernel_size, self._kernel_size],
                                                  padding='SAME', activation=tf.nn.relu)

                def residual_unit(x):
                    residual_output = tf.nn.relu(x)
                    residual_output = tf.layers.conv2d(residual_output, filters=self._conv_filters,
                                                       kernel_size=[self._kernel_size, self._kernel_size], padding='SAME')
                    residual_output = tf.nn.relu(residual_output)
                    residual_output = tf.layers.conv2d(residual_output, filters=self._conv_filters,
                                                       kernel_size=[self._kernel_size, self._kernel_size], padding='SAME')
                    return residual_output + x

                for i in range(self._num_residual_unit):
                    residual_input = residual_unit(residual_input)

                outputs.append(tf.layers.conv2d(tf.nn.relu(residual_input), filters=self._conv_filters,
                                                kernel_size=[self._kernel_size, self._kernel_size], padding='SAME'))
            
            if len(outputs) == 1:
                x = outputs[0]
            else:
                fusion_weight = tf.Variable(tf.random_normal([len(outputs), ]))
                for i in range(len(outputs)):
                    outputs[i] = fusion_weight[i] * outputs[i]
                x = tf.reduce_sum(outputs, axis=0)
            #! followings are a way of current ArbFlow framework's implementation
            #* 构建多尺度特征图
            feature_pyramid = [x]
            #* bottom-up
            for s in range(self._num_scale):
                # patch merge
                upper_feat = tf.layers.conv2d(feature_pyramid[-1], filters=self._conv_filters,
                                                kernel_size=[self._scale, self._scale],
                                                strides=[self._scale,self._scale], 
                                                activation=tf.nn.relu,
                                                padding='SAME')
                
                for i in range(self._num_residual_unit):
                    upper_feat = residual_unit(upper_feat)

                feature_pyramid.append(upper_feat)
            #* 尺度间建模
            def inter_modeling(x,scale):
                return tf.keras.layers.UpSampling2D(size=(scale, scale),data_format="channels_last")(x)
            
            #* 融合
            def fusion(feature_map_up_sampled,feature_map,fusion_method):
                if fusion_method=='concat':
                    x = tf.concat([feature_map_up_sampled,feature_map],axis=-1)
                    return tf.layers.conv2d(x, filters=self._conv_filters,
                                                      kernel_size=[1, 1],
                                                      padding='SAME', activation=tf.nn.relu)
                elif fusion_method=='add':
                    return tf.add(feature_map_up_sampled,feature_map)

            #* top-down
            for s in range(self._num_scale,0,-1):
                feature_pyramid[s-1] = fusion(inter_modeling(feature_pyramid[s],self._scale),feature_pyramid[s-1],fusion_method=self._fusion_method)
            x = feature_pyramid[0]
            # external dims
            if self._external_dim is not None and self._external_dim > 0:
                external_input = tf.placeholder(tf.float32, [None, self._external_dim])
                self._input['external_feature'] = external_input.name
                external_dense = tf.layers.dense(inputs=external_input, units=10)
                external_dense = tf.tile(tf.reshape(external_dense, [-1, 1, 1, 10]),
                                         [1, self._height, self._width, 1])
                x = tf.concat([x, external_dense], axis=-1)

            x = tf.layers.dense(x, units=1, name='prediction', activation=tf.nn.tanh)
            if self._is_mask:
                loss = tf.compat.v1.losses.mean_squared_error(target,x,weights=mask)
            else:
                loss = tf.compat.v1.losses.mean_squared_error(target,x,weights=1)
            train_op = tf.train.AdamOptimizer(self._lr).minimize(loss)

            self._output['prediction'] = x.name
            self._output['loss'] = loss.name

            self._op['train_op'] = train_op.name

        super(ArbFlow, self).build()

    def _get_feed_dict(self, closeness_feature=None, period_feature=None, trend_feature=None,
                       target=None, external_feature=None,mask=None):
        '''
        The method to get feet dict for tensorflow model.

        Users may modify this method according to the format of input.

        Args:
            closeness_feature (np.ndarray or ``None``): The closeness history data.
                If type is np.ndarray, its shape is [time_slot_num, height, width, closeness_len].
            period_feature (np.ndarray or ``None``): The period history data.
                If type is np.ndarray, its shape is [time_slot_num, height, width, period_len].
            trend_feature (np.ndarray or ``None``): The trend history data.
                If type is np.ndarray, its shape is [time_slot_num, height, width, trend_len].
            target (np.ndarray or ``None``): The target value data.
                If type is np.ndarray, its shape is [time_slot_num, height, width, 1].
            external_feature (np.ndarray or ``None``): The external feature data.
                If type is np.ndaaray, its shape is [time_slot_num, feature_num].
        '''
        feed_dict = {}
        if target is not None:
            feed_dict['target'] = target
        if self._external_dim is not None and self._external_dim > 0:
            feed_dict['external_feature'] = external_feature
        if self._closeness_len is not None and self._closeness_len > 0:
            feed_dict['closeness_feature'] = closeness_feature
        if self._period_len is not None and self._period_len > 0:
            feed_dict['period_feature'] = period_feature
        if self._trend_len is not None and self._trend_len > 0:
            feed_dict['trend_feature'] = trend_feature
        if self._is_mask:
            feed_dict['mask'] = mask
        return feed_dict
