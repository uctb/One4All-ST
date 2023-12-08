def StationLevelContextModeling():
    '''
    Provide a station level context modeling method conforming to the second hiearchy of weather modeling and poi modeling

    Args:
    
        st_contextdata(ndarray): context data(Weather or poi) in spatio-temporal raster form after duplication through temporal dimension or spatial dimension.
        is_varying(bool): Whether we choose a station varying way or a station constant way to model context.
        ...
        ...
    
    :return: context representation.
    :type: numpy.ndarray.
    '''
    #TODO: 去借鉴一下GraphModelLayers.py文件的写法，可以稍稍系统性的学一下tensorflow相关技术。
    pass

# class MultiEmbeddingLayer()