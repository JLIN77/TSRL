import os
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

def list_files(directory):
    Num = count_files(directory)
    print(directory+'/n'+ "Num={}".format(Num))
    os.makedirs('{}/modes/32'.format(str(directory)), exist_ok=True)
    os.makedirs('{}/modes/32^'.format(str(directory)), exist_ok=True)
    os.makedirs('{}/modes/16'.format(str(directory)), exist_ok=True)
    os.makedirs('{}/modes/16^'.format(str(directory)), exist_ok=True)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename in os.listdir(os.path.join(directory, 'modes/16')):
            Num -= 1
            print("剩余{}".format(str(Num)))
            continue
        if os.path.isfile(file_path):
            Num -= 1
            print("剩余{}".format(str(Num)))
            generate_pseudo_points(file_path, directory)
            
            
def compute_angles(pc_np):
    tan_theta = pc_np[:, 2] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    theta = np.arctan(tan_theta)
    theta = (theta / np.pi) * 180

    sin_phi = pc_np[:, 1] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi_ = np.arcsin(sin_phi)
    phi_ = (phi_ / np.pi) * 180

    cos_phi = pc_np[:, 0] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi = np.arccos(cos_phi)
    phi = (phi / np.pi) * 180

    phi[phi_ < 0] = 360 - phi[phi_ < 0]
    phi[phi == 360] = 0

    return theta, phi

def beam_label(theta, beam):
    estimator=KMeans(n_clusters=beam, n_init=10)
    res=estimator.fit_predict(theta.reshape(-1, 1))
    label=estimator.labels_
    centroids=estimator.cluster_centers_
    return label, centroids[:,0]


def generate_mask(phi, beam, label, idxs, beam_ratio, bin_ratio):
    mask = np.zeros((phi.shape[0])).astype(bool)

    for i in range(0, beam, beam_ratio):
        phi_i = phi[label == idxs[i]]
        idxs_phi = np.argsort(phi_i)
        mask_i = (label == idxs[i])
        mask_temp = np.zeros((phi_i.shape[0])).astype(bool)
        mask_temp[idxs_phi[::bin_ratio]] = True
        mask[mask_i] = mask_temp

    return mask

def generate_pseudo_points(points_path, root_path):
    points = np.fromfile(str(points_path), dtype=np.float32).reshape(-1, 5)
    fts_name = os.path.basename(points_path)
    beam = 32
    pc_np = points[:, :3]
    theta, phi = compute_angles(pc_np)

    label, centroids = beam_label(theta, beam)

    idxs = np.argsort(centroids)
    root_path_cus = root_path
    root_path_cus = Path(root_path_cus)
    
    mask = generate_mask(phi, beam, label, idxs, beam_ratio=2, bin_ratio=1)
    save_downsample = points[mask]
    save_path = root_path_cus / 'modes' / '16' / ('%s' % fts_name)
    save_downsample.tofile(save_path)


    # mask = generate_mask(phi, beam, label, idxs, beam_ratio=4, bin_ratio=1)
    # save_downsample = points[mask]
    # save_path = root_path_cus / 'modes' / '16' / ('%s' % fts_name)
    # save_downsample.tofile(save_path)

    # mask = generate_mask(phi, beam, label, idxs, beam_ratio=4, bin_ratio=2)
    # save_downsample = points[mask]
    # save_path = root_path_cus / 'modes' / '16^' / ('%s' % fts_name)
    # save_downsample.tofile(save_path)
    
def count_files(directory):
    return sum(1 for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename)))

# 示例：遍历当前目录中的文件
list_files('/home/mmdetection3d-main/data/nuscenes/samples/LIDAR_TOP')