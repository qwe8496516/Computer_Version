imx = surface_normals.shape[1]
    imy = surface_normals.shape[0] #flipped for indexing
    #height_map = np.zeros((imx,imy)
    fx = surface_normals[:,:,0] / surface_normals[:,:,2]
    fy = surface_normals[:,:,1] / surface_normals[:,:,2]
    fy = np.nan_to_num(fy)
    fx = np.nan_to_num(fx)

    row = np.cumsum(fx,axis=1)
    column = np.cumsum(fy,axis=0)
    if integration_method == 'row':
        row_temp = np.vstack([row[0,:]]*imy)
        height_map = column + row_temp     
        #print(np.max(height_map))
    if integration_method == 'column':
        col_temp = np.stack([column[:,0].T]*imx,axis=1)
        height_map = row + col_temp   
        #print(height_map.T)
    if integration_method == 'average':
        row_temp = np.vstack([row[0,:]]*imy)
        col_temp = np.stack([column[:,0].T]*imx,axis=1)
        height_map = (row + column + row_temp + col_temp) / 2
        
    if integration_method == 'random':
        iteration = 10
        height_map = np.zeros((imy,imx))
        for x in range(iteration):
            print(x)
            for i in range(imy):
                print(i)
                for j in range(imx):
                    id1 = 0
                    id2 = 0
                    val = 0
                    path = [0] * i + [1] * j
                    random.shuffle(path)
                    for move in path:
                        if move == 0:
                            id1 += 1
                            if id1 > imy - 1: id1 -= 1
                            val += fy[id1][id2]
                            #print(val,fx[id1][id2])
                        if move == 1:
                            id2 += 1
                            if id2 > imx - 1: id2 -= 1
                            val += fx[id1][id2]
                    height_map[i][j] += val
                    #print(i,j,val)
        height_map = height_map / iteration
        #print(np.max(height_map))
    # print(height_map)
    return height_map