import numpy as np
import math
def bilinear_interpolate(src, dst_size):
    height_src, width_src, channel_src = src.shape  # (h, w, ch)
    height_dst, width_dst = dst_size  # (h, w)
    """中心对齐，投影目标图的横轴和纵轴到原图上"""
    ws_p = np.array([(i + 0.5) / width_dst * width_src - 0.5 for i in range(width_dst)], dtype=np.float32)
    hs_p = np.array([(i + 0.5) / height_dst * height_src - 0.5 for i in range(height_dst)], dtype=np.float32)
    ws_p = np.repeat(ws_p.reshape(1, width_dst), height_dst, axis=0)
    hs_p = np.repeat(hs_p.reshape(height_dst, 1), width_dst, axis=1)
    """找出每个投影点在原图的近邻点坐标"""
    ws_0 = np.clip(np.floor(ws_p), 0, width_src - 2).astype(np.int)
    hs_0 = np.clip(np.floor(hs_p), 0, height_src - 2).astype(np.int)
    ws_1 = ws_0 + 1
    hs_1 = hs_0 + 1
    """四个临近点的像素值"""
    f_00 = src[hs_0, ws_0, :].T
    f_01 = src[hs_0, ws_1, :].T
    f_10 = src[hs_1, ws_0, :].T
    f_11 = src[hs_1, ws_1, :].T
    """计算权重"""
    w_00 = ((hs_1 - hs_p) * (ws_1 - ws_p)).T
    w_01 = ((hs_1 - hs_p) * (ws_p - ws_0)).T
    w_10 = ((hs_p - hs_0) * (ws_1 - ws_p)).T
    w_11 = ((hs_p - hs_0) * (ws_p - ws_0)).T
    """计算目标像素值"""
    return (f_00 * w_00).T + (f_01 * w_01).T + (f_10 * w_10).T + (f_11 * w_11).T
if __name__ == '__main__':
    src = np.array([[0, 1], [2, 3]])
    src = np.expand_dims(src, axis=2)
    print(src.shape)
    dst = bilinear_interpolate(src, dst_size=(src.shape[0] * 2, src.shape[1] * 2))
    print(dst.shape)
    print(src)
    print(dst)