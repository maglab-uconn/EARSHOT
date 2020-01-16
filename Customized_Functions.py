import tensorflow as tf

def Correlation2D(x, y):
    """
    Compute the correlations between each rows of two tensors. Main purpose is checking the
        correlations between the units of two layers
    Args:
        x: 2d tensor (MxN). The number of second dimension should be same to y's second dimension.
        y: 2d tensor (LxN). The number of second dimension should be same to x's second dimension.
    Returns:
        correlation_Tensor: A `Tensor` representing the correlation between the rows. Size is (M x L)
        p_Value_Tensor: A `Tensor` representing the p-value of correlation. Size is (M x L)
    """
    avgsub_X_Tensor = x - tf.reduce_mean(x, axis = 1, keepdims=True)  #[M, N]
    avgsub_Y_Tensor = y - tf.reduce_mean(y, axis = 1, keepdims=True)  #[L, N]

    sumed_Pow_X_Tensor = tf.reduce_sum(tf.pow(avgsub_X_Tensor, 2), axis=1, keepdims= True)      #[M, 1]
    sumed_Pow_Y_Tensor = tf.reduce_sum(tf.pow(avgsub_Y_Tensor, 2), axis=1, keepdims= True)    #[L, 1]

    correlation_Tensor = tf.matmul(avgsub_X_Tensor, tf.transpose(avgsub_Y_Tensor)) / tf.sqrt(tf.matmul(sumed_Pow_X_Tensor, tf.transpose(sumed_Pow_Y_Tensor)))    #[M, L]
    p_Value_Tensor = 1 - tf.erf(tf.abs(correlation_Tensor) * tf.sqrt(tf.cast(tf.shape(x)[1], tf.float32)) / tf.sqrt(2.0))  #[M, L]

    correlation_Tensor = tf.identity(correlation_Tensor, name="correlation")
    p_Value_Tensor = tf.identity(p_Value_Tensor, name="p_value")

    return (correlation_Tensor, p_Value_Tensor)

def Batch_Correlation2D(x, y):
    """
    Compute the correlations between each rows of two tensors. Main purpose is checking the
        correlations between the units of two layers
    Args:
        x: 3d tensor (BATCHxMxN). The number of first and third dimension should be same to y's first and third dimension.
        y: 3d tensor (BATCHxLxN). The number of first and third dimension should be same to x's first and third dimension.
    Returns:
        correlation_Tensor: A `Tensor` representing the correlation between the rows. Size is (BATCH x M x L)
        p_Value_Tensor: A `Tensor` representing the p-value of correlation. Size is (BATCH x M x L)
    """
    t = tf.concat([x,y], axis = 1)
    t_Min = tf.reduce_min(tf.abs(t)) + 1e-8
    t_Max = tf.reduce_max(tf.abs(t))
    x = x / t_Min * t_Max
    y = y / t_Min * t_Max

    avgsub_X_Tensor = x - tf.reduce_mean(x, axis = 2, keepdims=True)  #[Batch, M, N]
    avgsub_Y_Tensor = y - tf.reduce_mean(y, axis = 2, keepdims=True)  #[Batch, L, N]

    sumed_Pow_X_Tensor = tf.reduce_sum(tf.pow(avgsub_X_Tensor, 2), axis=2, keepdims= True)      #[Batch, M, 1]
    sumed_Pow_Y_Tensor = tf.reduce_sum(tf.pow(avgsub_Y_Tensor, 2), axis=2, keepdims= True)    #[Batch, L, 1]

    correlation_Tensor = tf.matmul(avgsub_X_Tensor, tf.transpose(avgsub_Y_Tensor, perm=[0, 2, 1])) / tf.sqrt(tf.matmul(sumed_Pow_X_Tensor, tf.transpose(sumed_Pow_Y_Tensor, perm=[0, 2, 1])))    #[Batch, M, L]
    p_Value_Tensor = 1 - tf.erf(tf.abs(correlation_Tensor) * tf.sqrt(tf.cast(tf.shape(x)[2], tf.float32)) / tf.sqrt(2.0))  #[M, L]

    correlation_Tensor = tf.identity(correlation_Tensor, name="correlation")
    p_Value_Tensor = tf.identity(p_Value_Tensor, name="p_value")

    return (correlation_Tensor, p_Value_Tensor)


def MDS(x, dimension = 2):
    """
    Compute the multidimensional scaling coordinates.
    Equation reference: https://m.blog.naver.com/PostView.nhn?blogId=kmkim1222&logNo=220082090874&proxyReferer=https%3A%2F%2Fwww.google.com%2F
    Args:
        x: 2d tensor (NxN). The distance matrix.
        dimension: int32 or scalar tensor. The compressed dimension.
    Returns:
        mds_Coordinate: A `Tensor` representing the compressed coordinates. Size is (N x Dimension)
    """
    element_Number = tf.shape(x)[1]
    j = tf.eye(element_Number) - tf.cast(1/element_Number, tf.float32) * tf.ones_like(x)
    b = -0.5 * (j @ tf.pow(x, 2) @ j)
    eigen_Value, eigen_Vector = tf.self_adjoint_eig(b)
    selected_Eigen_Value, top_Eigen_Value_Indice = tf.nn.top_k(eigen_Value, k=dimension)
    selected_eigen_Vector = tf.transpose(tf.gather(tf.transpose(eigen_Vector), top_Eigen_Value_Indice))
    mds_Coordinate = selected_eigen_Vector @ tf.sqrt(tf.diag(selected_Eigen_Value))
    mds_Coordinate = tf.identity(mds_Coordinate, name="mds_Coordinate")

    return mds_Coordinate


def Cosine_Similarity2D(x, y):
    """
    Compute the cosine similarity between each row of two tensors.
    Args:
        x: 2d tensor (MxN). The number of second dimension should be same to y's second dimension.
        y: 2d tensor (LxN). The number of second dimension should be same to x's second dimension.
    Returns:
        cosine_Similarity: A `Tensor` representing the cosine similarity between the rows. Size is (M x L)
    """
    print(x)
    print(y)
    tiled_X = tf.tile(tf.expand_dims(x, [1]), multiples = [1, tf.shape(y)[0], 1])   #[M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [0]), multiples = [tf.shape(x)[0], 1, 1])   #[M, L, N]
    cosine_Similarity = tf.reduce_sum(tiled_Y * tiled_X, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Y, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_X, 2), axis = 2)) + 1e-8)  #[M, L]
    cosine_Similarity = tf.identity(cosine_Similarity, name="cosine_Similarity")

    return cosine_Similarity

def Batch_Cosine_Similarity2D(x, y):
    """
    Compute the cosine similarity between each row of two tensors.
    Args:
        x: 3d tensor (BATCHxMxN). The number of first and third dimension should be same to y's first and third dimension.
        y: 3d tensor (BATCHxLxN). The number of first and third dimension should be same to x's first and third dimension.
    Returns:
        cosine_Similarity: A `Tensor` representing the cosine similarity between the rows. Size is (M x L) (BATCH x M x L)
    """
    tiled_X = tf.tile(tf.expand_dims(x, [2]), multiples = [1, 1, tf.shape(y)[1], 1])   #[Batch, M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [1]), multiples = [1, tf.shape(x)[1], 1, 1])   #[Batch, M, L, N]
    cosine_Similarity = tf.reduce_sum(tiled_Y * tiled_X, axis = 3) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Y, 2), axis = 3)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_X, 2), axis = 3)) + 1e-8)  #[Batch, M, L]
    cosine_Similarity = tf.identity(cosine_Similarity, name="cosine_Similarity")

    return cosine_Similarity


def Mean_Squared_Error2D(x, y):
    """
    Compute the cosine mean squared error between each row of two tensors.
    Args:
        x: 2d tensor (MxN). The number of second dimension should be same to y's second dimension.
        y: 2d tensor (LxN). The number of second dimension should be same to x's second dimension.
    Returns:
        mean_Squared_Error: A `Tensor` representing the mean squared error between the rows. Size is (M x L)
    """
    tiled_X = tf.tile(tf.expand_dims(x, [1]), multiples = [1, tf.shape(y)[0], 1])   #[M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [0]), multiples = [tf.shape(x)[0], 1, 1])   #[M, L, N]
    mean_Squared_Error = tf.reduce_mean(tf.pow(tiled_Y - tiled_X, 2), axis=2)  #[M, L]
    mean_Squared_Error = tf.identity(mean_Squared_Error, name="mean_Squared_Error")

    return mean_Squared_Error

def Batch_Mean_Squared_Error2D(x, y):
    """
    Compute the cosine mean squared error between each row of two tensors.
    Args:
        x: 3d tensor (BATCHxMxN). The number of first and third dimension should be same to x's first and third dimension.
        y: 3d tensor (BATCHxLxN). The number of first and third dimension should be same to x's first and third dimension.
    Returns:
        mean_Squared_Error: A `Tensor` representing the mean squared error between the rows. Size is (BATCH x M x L)
    """

    tiled_X = tf.tile(tf.expand_dims(x, [2]), multiples = [1, 1, tf.shape(y)[1], 1])   #[Batch, M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [1]), multiples = [1, tf.shape(x)[1], 1, 1])   #[Batch, M, L, N]
    mean_Squared_Error = tf.reduce_mean(tf.pow(tiled_Y - tiled_X, 2), axis=3)  #[Batch, M, L]
    mean_Squared_Error = tf.identity(mean_Squared_Error, name="mean_Squared_Error")

    return mean_Squared_Error


def Euclidean_Distance2D(x, y):
    """
    Compute the Euclidean distance between each row of two tensors.
    Args:
        x: 2d tensor (MxN). The number of second dimension should be same to y's second dimension.
        y: 2d tensor (LxN). The number of second dimension should be same to x's second dimension.
    Returns:
        euclidean_Distance: A `Tensor` representing the Euclidean distance between the rows. Size is (M x L)
    """
    tiled_X = tf.tile(tf.expand_dims(x, [1]), multiples = [1, tf.shape(y)[0], 1])   #[M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [0]), multiples = [tf.shape(x)[0], 1, 1])   #[M, L, N]
    euclidean_Distance = tf.sqrt(tf.reduce_sum(tf.pow(tiled_Y - tiled_X, 2), axis=2))  #[M, L]
    euclidean_Distance = tf.identity(euclidean_Distance, name="euclidean_Distance")

    return euclidean_Distance

def Batch_Euclidean_Distance2D(x, y):
    """
    Compute the Euclidean distance between each row of two tensors.
    Args:
        x: 3d tensor (BATCHxMxN). The number of first and third dimension should be same to x's first and third dimension.
        y: 3d tensor (BATCHxLxN). The number of first and third dimension should be same to x's first and third dimension.
    Returns:
        euclidean_Distance: A `Tensor` representing the Euclidean distance between the rows. Size is (BATCH x M x L)
    """
    tiled_X = tf.tile(tf.expand_dims(x, [2]), multiples = [1, 1, tf.shape(y)[1], 1])   #[Batch, M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [1]), multiples = [1, tf.shape(x)[1], 1, 1])   #[Batch, M, L, N]
    euclidean_Distance = tf.sqrt(tf.reduce_sum(tf.pow(tiled_Y - tiled_X, 2), axis=3))  #[Batch, M, L]
    euclidean_Distance = tf.identity(euclidean_Distance, name="euclidean_Distance")

    return euclidean_Distance


def Cross_Entropy2D(x, y):
    """
    Compute the cross entropy between each row of two tensors.
    Args:
        x: 2d tensor (MxN). The number of second dimension should be same to y's second dimension.
        y: 2d tensor (LxN). The number of second dimension should be same to x's second dimension.
    Returns:
        cosine_Similarity: A `Tensor` representing the cross entropy between the rows. Size is (M x L)
    """
    tiled_X = tf.tile(tf.expand_dims(x, [1]), multiples = [1, tf.shape(y)[0], 1])   #[M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [0]), multiples = [tf.shape(x)[0], 1, 1])   #[M, L, N]
    cross_Entropy = -tf.reduce_mean(tiled_Y * tf.log(tiled_X + 1e-8) + (1 - tiled_Y) * tf.log(1 - tiled_X + 1e-8), axis = 2)  #[M, L]
    cross_Entropy = tf.identity(cross_Entropy, name="cross_Entropy")

    return cross_Entropy

def Batch_Cross_Entropy2D(x, y):
    """
    Compute the cross entropy between each row of two tensors.
    Args:
        x: 3d tensor (BATCHxMxN). The number of first and third dimension should be same to x's first and third dimension.
        y: 3d tensor (BATCHxLxN). The number of first and third dimension should be same to x's first and third dimension.
    Returns:
        cosine_Similarity: A `Tensor` representing the cross entropy between the rows. Size is (BATCH x M x L)
    """
    tiled_X = tf.tile(tf.expand_dims(x, [2]), multiples = [1, 1, tf.shape(y)[1], 1])   #[Batch, M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [1]), multiples = [1, tf.shape(x)[1], 1, 1])   #[Batch, M, L, N]
    cross_Entropy = -tf.reduce_mean(tiled_Y * tf.log(tiled_X + 1e-8) + (1 - tiled_Y) * tf.log(1 - tiled_X + 1e-8), axis = 3)  #[Batch, M, L]
    cross_Entropy = tf.identity(cross_Entropy, name="cross_Entropy")

    return cross_Entropy


def Z_Score(x, axis = None):
    """
    Calculate the z score of x.
    Args:
        x: nd tensor.
        axis: int or list of int. All values which are on the selected axes are regarded as a single group.
    Returns:
        z: 2d tensor (MxL). The z-score. If the sign of z-score is positive, the x is bigger than y.
        p: 2d tensor (MxL). The p-value based on the two-sided test.
    """
    if type(axis) == int:
        axis = [axis]

    m, v = tf.nn.moments(x, axes=axis)
    for selected_Axis in axis:
        m = tf.expand_dims(m, axis=selected_Axis)
        v = tf.expand_dims(v, axis=selected_Axis)

    multiples = [1] * len(x.get_shape())
    for selected_Axis in axis:
        multiples[selected_Axis] = tf.shape(x)[selected_Axis]

    m = tf.tile(m, multiples=multiples)
    v = tf.tile(v, multiples=multiples)
    z = (x - m) / (tf.sqrt(v) + 1e-8)
    z = tf.identity(z, name="z_Score")

    return z


def Wilcoxon_Signed_Rank_Test2D(x, y):
    """
    Conduct the Wilcoxon signed-rank test between each row of two tensors.
    Formula referred: https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    Args:
        x: 2d tensor (MxN). The number of second dimension should be same to y's second.
        y: 2d tensor (LxN). The number of second dimension should be same to x's second.
    Returns:
        z: 2d tensor (MxL). The z-score. If the sign of z-score is positive, the x is bigger than y.
        p: 2d tensor (MxL). The p-value based on the two-sided test.
    """
    tiled_X = tf.tile(tf.expand_dims(x, [1]), multiples = [1, tf.shape(y)[0], 1])   #[M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [0]), multiples = [tf.shape(x)[0], 1, 1])   #[M, L, N]

    subtract_XY = tiled_X - tiled_Y
    vector_Size = tf.cast(tf.shape(subtract_XY)[2], tf.float32)

    sign_Subtract_XY = tf.sign(subtract_XY)
    abs_Subtract_XY = tf.abs(subtract_XY)

    index_Dimension1 = tf.tile(
        tf.expand_dims(tf.expand_dims(tf.range(tf.shape(subtract_XY)[0]), axis = 1), axis = 2),
        multiples=[1, tf.shape(subtract_XY)[1], tf.shape(subtract_XY)[2]]
        )    #[M, L, N]
    index_Dimension2 = tf.tile(
        tf.expand_dims(tf.expand_dims(tf.range(tf.shape(subtract_XY)[1]), axis = 0), axis = 2),
        multiples=[tf.shape(subtract_XY)[0], 1, tf.shape(subtract_XY)[2]]
        )    #[M, L, N]
    index_Dimension3 = tf.nn.top_k(-abs_Subtract_XY, k=tf.shape(abs_Subtract_XY)[2], sorted=False).indices   #[M, L, N]

    rank_Map = tf.stack([index_Dimension1, index_Dimension2, index_Dimension3], axis=3) #[M, L, N, 3]
    mapped_Sign_X = tf.gather_nd(sign_Subtract_XY, indices= rank_Map)

    tiled_Range = tf.tile(
        tf.expand_dims(tf.expand_dims(tf.cast(tf.range(tf.shape(subtract_XY)[2]), dtype=tf.float32), axis = 0), axis = 1),
        multiples=[tf.shape(subtract_XY)[0], tf.shape(subtract_XY)[1], 1]
        )    #[M, L, N]

    wilcoxon_Value = tf.reduce_sum(mapped_Sign_X * (tiled_Range + 1), axis=2)  #[M, L]

    z_Score = wilcoxon_Value / tf.sqrt(vector_Size * (vector_Size + 1) * (2* vector_Size + 1) / 6)
    p_Value = 1 - tf.erf(tf.abs(z_Score) / tf.sqrt(2.0))

    z_Score = tf.identity(z_Score, name="wilcoxon_Signed_Rank_Test_Z_Score")
    p_Value = tf.identity(p_Value, name="wilcoxon_Signed_Rank_Test_P_Value")

    return z_Score, p_Value


def Wilcoxon_Rank_Sum_Test2D(x, y):
    """
    Conduct the Wilcoxon rank-sum test (Mann–Whitney U test) between each row of two tensors.
    Formula referred: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
                      http://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_nonparametric/BS704_Nonparametric4.html
                      http://3months.tistory.com/128
    Args:
        x: 2d tensor (MxA).
        y: 2d tensor (LxB).
    Returns:
        z: 2d tensor (MxL). The z-score. If the sign of z-score is positive, the x's mean is bigger than y's.
        p: 2d tensor (MxL). The p-value based on the two-sided test.
    """
    x_Size = tf.cast(tf.shape(x)[1], tf.float32)
    y_Size = tf.cast(tf.shape(y)[1], tf.float32)

    tiled_X = tf.tile(tf.expand_dims(x, [1]), multiples = [1, tf.shape(y)[0], 1])   #[M, L, A]
    tiled_Y = tf.tile(tf.expand_dims(y, [0]), multiples = [tf.shape(x)[0], 1, 1])   #[M, L, B]

    concat_XY = tf.concat([tiled_X, tiled_Y], axis=2)   #[M, L, (A+B)]

    rank_Map = tf.cast(tf.nn.top_k(-concat_XY, k=tf.shape(concat_XY)[2], sorted=False).indices, dtype=tf.float32)   #[M, L, (A+B)]

    y_Map = tf.clip_by_value(rank_Map - x_Size + 1, clip_value_min=0, clip_value_max=1) #[M, L, (A+B)]

    tiled_Range = tf.tile(
        tf.expand_dims(tf.expand_dims(tf.cast(tf.range(tf.shape(concat_XY)[2]) + 1, dtype=tf.float32), axis = 0), axis = 1),
        multiples=[tf.shape(concat_XY)[0], tf.shape(concat_XY)[1], 1]
        )    #[M, L, (A+B)]

    sum_Rank_Y = tf.reduce_sum(y_Map * tiled_Range, axis=2)  #[M, L]

    wilcoxon_Value = x_Size * y_Size + (y_Size * (y_Size + 1) / 2) - sum_Rank_Y

    mean_Wilconxon = x_Size * y_Size / 2   #Because, W1 + W2 = n1n2.
    s = tf.sqrt(x_Size * y_Size * (x_Size + y_Size + 1) / 12)
    z_Score = (wilcoxon_Value - mean_Wilconxon) / s
    p_Value = tf.cast(1 - tf.erf(tf.abs(tf.cast(z_Score, tf.float64)) / tf.sqrt(tf.cast(2.0, tf.float64))), tf.float32)    #To know more detail p-value (float32 cannot cover z-score which is over 5.6)

    z_Score = tf.identity(z_Score, name="wilcoxon_Rank_Sum_Test_Z_Score")
    p_Value = tf.identity(p_Value, name="wilcoxon_Rank_Sum_Test_P_Value")

    return z_Score, p_Value

if __name__ == "__main__":
    with tf.Session() as tf_Session:
        import numpy as np

        #x_P = tf.placeholder(tf.float32, shape=[None, 256, None])
        #y_P = tf.placeholder(tf.float32, shape=[None, 300, None])

        #x = np.random.rand(5, 256, 110)
        #y = np.random.rand(5, 300, 110)

        #a = Batch_Mean_Squared_Error2D(x,y)
        #b = Batch_Euclidean_Distance2D(x,y)
        #c = Batch_Cross_Entropy2D(x,y)

        #d,e,f = tf_Session.run([a,b,c], feed_dict={x_P:x, y_P:y})
        #print(d.shape)
        #print(e.shape)
        #print(f.shape)

        #x = np.array([[117.1, 121.3, 127.8, 121.9, 117.4, 124.5, 119.5, 115.1]])
        #y = np.array([[123.5, 125.3, 126.5, 127.9, 122.1, 125.6, 129.8, 117.2]])
        #x = np.array([[305, 16, 122, 68]])
        #y = np.array([[25, 63, 84, 103]])
        x = np.array([[0.093090194, 0.268623055, 0.516203482, 0.67306552, 0.797922431, 0.322991788, 0.92960736, 0.995060907, 0.624005894, 0.385368098, 0.786522125]])
        y = np.array([[0.290919606, 0.805307853, 0.621864781, 0.569390926, 0.626139531, 0.881142382, 0.920040644, 0.264862919, 0.012541362, 0.617295819, 0.171516068]])

        x_P = tf.placeholder(tf.float32, shape=[None, None])
        y_P = tf.placeholder(tf.float32, shape=[None, None])

        t = Wilcoxon_Rank_Sum_Test2D(x_P, y_P)

        e,f = tf_Session.run(t, feed_dict={x_P:x, y_P:y})

        print(e)
        print(f)
