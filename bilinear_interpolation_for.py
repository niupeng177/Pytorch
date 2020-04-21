# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:20:47 2020

@author: 牛朋
"""

def bilinear_interpolate(src, dst_size):
    height_src, width_src, channel_src = src.shape  # (h, w, ch)
    height_dst, width_dst = dst_size  # (h, w)
    
    """
    中心对齐，投影目标图的横轴和纵轴到原图上
    """
    ws_p = np.array([(i + 0.5) / width_dst * width_src - 0.5 for i in range(width_dst)], dtype=np.float32)
    hs_p = np.array([(i + 0.5) / height_dst * height_src - 0.5 for i in range(height_dst)], dtype=np.float32)
    
    """找出每个投影点在原图横轴方向的近邻点坐标对"""
    # w_0的取值范围是 0 ~ (width_src-2)，因为w_1 = w_0 + 1
    ws_0 = np.clip(np.floor(ws_p), 0, width_src-2).astype(np.int)
        
    """找出每个投影点在原图纵轴方向的近邻点坐标对"""
    # h_0的取值范围是 0 ~ (height_src-2)，因为h_1 = h_0 + 1
    hs_0 = np.clip(np.floor(hs_p), 0, height_src-2).astype(np.int)
        
    """
    计算目标图各个点的像素值
    f(h, w) = f(h_0, w_0) * (1 - u) * (1 - v)
            + f(h_0, w_1) * (1 - u) * v
            + f(h_1, w_0) * u * (1 - v)
            + f(h_1, w_1) * u * v
    """
    dst = np.zeros(shape=(height_dst, width_dst, channel_src), dtype=np.float32)
    us = hs_p - hs_0
    vs = ws_p - ws_0
    _1_us = 1 - us
    _1_vs = 1 - vs
    for h in range(height_dst):
        h_0, h_1 = hs_0[h], hs_0[h]+1  # 原图的坐标
        for w in range(width_dst):
            w_0, w_1 = ws_0[w], ws_0[w]+1 # 原图的坐标
            for c in range(channel_src):
                dst[h][w][c] = src[h_0][w_0][c] * _1_us[h] * _1_vs[w] \
                            + src[h_0][w_1][c] * _1_us[h] * vs[w] \
                            + src[h_1][w_0][c] * us[h] * _1_vs[w] \
                            + src[h_1][w_1][c] * us[h] * vs[w]
    return dst
if __name__ == '__main__':
    src = np.array([[0, 1], [2, 4]])
    src = np.expand_dims(src, axis=2)
    dst = bilinear_interpolate(src, dst_size=(4, 4))
    print(dst[:, :, 0])