import numpy as np
import os 
import cv2 
import matplotlib.pyplot as plt
from matplotlib import patches
import torch
# from sklearn.cluster import KMeans


def generate_H_data_(dataset_name):
    H_data = []
    if dataset_name == 'STUDENT':
        H_data = np.array([[0.02220407,0,-0.48],
                    [0,-0.02477289,13.92551292],
                    [0,0,1]]) #students(001/003)
    elif dataset_name == 'ZARA01':
        H_data = np.array([[0.02174104 ,0,-0.15],
                        [0,-0.02461883,13.77429807],
                        [0,0,1]]) #crowds_zara01
    elif dataset_name == 'ZARA02':
        H_data = np.array([[0.02174104,0,-0.4],
                        [0,-0.02386598,14.98401686],
                        [0,0,1]]) #crowds_zara02/03
    elif dataset_name == 'ETH':
        H_data = np.array([[2.8128700e-02,2.0091900e-03,-4.6693600e+00],
                        [8.0625700e-04,2.5195500e-02,-5.0608800e+00],
                        [3.4555400e-04,9.2512200e-05,4.6255300e-01]]) #biwi_eth
    elif dataset_name == 'HOTEL':
        H_data = np.array([[1.1048200e-02,6.6958900e-04,-3.3295300e+00],
                        [-1.5966000e-03,1.1632400e-02,-5.3951400e+00],
                        [1.1190700e-04,1.3617400e-05,5.4276600e-01]]) #biwi_hotel
    else:
        print('check H_data infor!')
    # print('current H data: ', dataset_name)
    return H_data

def generate_dataset_tag(datasets_path):
        normalized_path = os.path.normpath(datasets_path)

        # 提取文件名（包含扩展名）
        filename = os.path.basename(normalized_path)

        # 分离文件名和扩展名
        dataset_name, extension = os.path.splitext(filename)

        h_data = generate_H_data_(dataset_name)
        if dataset_name == 'ETH' or dataset_name == 'HOTEL':
            xy_tag = 0
        else:
            xy_tag = 1
        return h_data, xy_tag

def trajectory2pixel(traj_data, H_data):
    # traj_data input size [x,2]
    # H_data input size [3,3]
    trajectory_data = traj_data
    N = len(trajectory_data)
    # 创建形状为 (N, 1) 的全 1 列向量
    ones_column = np.ones((N, 1))
    # 拼接数据，结果形状为 (N, 3)
    data = np.hstack((trajectory_data, ones_column))
    inv_H_data = np.linalg.inv(H_data).T
    pixel_traj = np.dot(data, inv_H_data)

    epsilon = 1e-10  # To prevent division by zero
    denom = pixel_traj[:, 2][:, np.newaxis] + epsilon

    pixel_traj_normalized = pixel_traj[:, :2] / denom
    # return the pixeled pos from traj
    return pixel_traj_normalized.astype(int)

def pixel2trajectory(pixel_traj_data, H_data):
    # pixel_traj_data input size [x,2]
    ones_column = np.transpose(np.matrix(np.repeat(1, len(pixel_traj_data))))
    pixel_traj_data = np.hstack((pixel_traj_data, ones_column))
    world_trajectory = np.matmul(H_data, pixel_traj_data.T).T
    world_trajectory = world_trajectory / world_trajectory[:,2]
    world_trajectory = world_trajectory[:, :2]

    return world_trajectory

def pixel2trajectory_torch(pixel_traj_data, H_data):
    """
    将像素轨迹转换为世界坐标轨迹
    - pixel_traj_data: [N, 2] (像素坐标)
    - H_data: [3, 3] (透视变换矩阵)
    
    返回:
    - world_trajectory: [N, 2] (世界坐标轨迹)
    """

    # 确保输入为 tensor
    if not isinstance(pixel_traj_data, torch.Tensor):
        pixel_traj_data = torch.tensor(pixel_traj_data, dtype=torch.float32)
    if not isinstance(H_data, torch.Tensor):
        H_data = torch.tensor(H_data, dtype=torch.float32)

    # 在 pixel_traj_data 添加一列全 1
    ones_column = torch.ones((pixel_traj_data.shape[0], 1), dtype=torch.float32, device=pixel_traj_data.device)
    pixel_traj_data = torch.cat((pixel_traj_data, ones_column), dim=1)  # [N, 3]

    # 计算世界坐标
    world_trajectory = torch.matmul(H_data, pixel_traj_data.T).T  # [N, 3]
    world_trajectory = world_trajectory / world_trajectory[:, 2:3]  # 归一化 (保持形状)
    world_trajectory = world_trajectory[:, :2]  # 取前两列 (x, y)

    return world_trajectory

def collect_scenario_files(root_dir, scenario):
    """
    从 root_dir 下查找所有名称包含 'scenario' 的文件夹，
    并收集其中 .txt 和 .jpg 文件的完整路径。
    """
    all_txt_paths = []
    all_video_paths = []

    # 1. 遍历 root_dir 下所有文件和文件夹
    for item in os.listdir(root_dir):
        # 检查是否是文件夹 & 文件夹名中含 scenario
        if os.path.isdir(os.path.join(root_dir, item)) and scenario in item:
            scenario_dir_path = os.path.join(root_dir, item)
            # 2. 遍历 scenario 文件夹内的所有文件
            for file_name in os.listdir(scenario_dir_path):
                # 判断文件类型
                if file_name.endswith(".txt"):
                    all_txt_paths.append(os.path.join(scenario_dir_path, file_name))
                elif file_name.endswith(".avi") or file_name.endswith(".mov") :
                    all_video_paths.append(os.path.join(scenario_dir_path, file_name))

    return all_txt_paths, all_video_paths

def generate_guide_points(image, num_points=5000, eps=15, min_samples=10, show_result=True):
    """
    读取黑白图片，在白色区域内撒点并进行聚类，生成合理的 Guide Points
    :param image_path: 输入的黑白图片路径
    :param num_points: 撒点数量
    :param eps: DBSCAN 聚类的邻域半径
    :param min_samples: DBSCAN 形成核心点的最小样本数
    :param show_result: 是否可视化
    :return: 过滤后的白色区域点集，最终的 Guide Points
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # HxW

    height, width = image.shape

    # 2. 随机撒点
    points = np.random.randint(0, [width, height], size=(num_points, 2))

    # 3. 过滤掉黑色区域的点
    white_points = np.array([p for p in points if image[p[1], p[0]] > 128])

    # # 4. DBSCAN 聚类
    # if len(white_points) > 0:
    #     clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(white_points)
    #     labels = clustering.labels_
    #     unique_labels = set(labels)

    #     guide_points = []
    #     for label in unique_labels:
    #         if label == -1:  # 过滤噪声点
    #             continue
    #         cluster = white_points[labels == label]
    #         centroid = np.mean(cluster, axis=0)
    #         guide_points.append(centroid)

    #     guide_points = np.array(guide_points)
    # else:
    #     guide_points = np.array([])
    kmeans = KMeans(n_clusters=min_samples, random_state=42, n_init="auto")
    kmeans.fit(white_points)
    guide_points = kmeans.cluster_centers_ 
    # return kmeans.cluster_centers_  # Guide Points

    # # 5. 可视化
    # if show_result:
    #     vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    #     # 画撒点（蓝色）
    #     for (x, y) in white_points:
    #         cv2.circle(vis_image, (int(x), int(y)), 1, (255, 0, 0), -1)

    #     # 画最终 Guide Points（红色）
    #     for (x, y) in guide_points:
    #         cv2.circle(vis_image, (int(x), int(y)), 5, (0, 0, 255), -1)

    #     # 显示结果
    #     cv2.imshow("Guide Points", vis_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return white_points, guide_points.astype(int)

def NNM(gt, guide_point_set):
    # 计算所有 guide point 到所有 ground truth 之间的欧几里得距离
    distances = np.linalg.norm(gt[:, np.newaxis, :] - guide_point_set[np.newaxis, :, :], axis=2)  # (N, M)

    # 计算每个 guide point 到所有 ground truth 的平均距离
    avg_distances = np.mean(distances, axis=0)  # (M,)

    # 找到平均距离最小的 guide point
    best_guide_idx = np.argmin(avg_distances)
    best_guide = guide_point_set[best_guide_idx]

    return best_guide, best_guide_idx

def draw_guide_points(ax, guide_points, tag):
    """ 绘制 Guide Points (红色) """
    if tag == 0:
        ax.scatter(guide_points[:, 1], guide_points[:, 0], c='purple', s=10, label="Inputs")
    else:
        ax.scatter(guide_points[:, 0], guide_points[:, 1], c='purple', s=10, label="Inputs")
    

def draw_best_guide_points(ax, best_guide_point):
    """ 绘制最佳 Guide Points (灰色，大小递增) """
    for i, (x, y) in enumerate(best_guide_point):
        ax.scatter(x, y, s=(10 + 5 * i), color="yellow", label="Best Guide" if i == 0 else None)

def draw_obs_trajectory(ax, each_agent, tag):
    """ 绘制当前 agent 轨迹 """
    for i, (x, y) in enumerate(each_agent[:, [0, 1]]):
        color = "blue" # 观测部分蓝色，预测部分绿色
        label = "Observed" if i == 0 else None
        if tag == 0:
            ax.scatter(y, x, s=20, color=color, label=label)
        else:
            ax.scatter(x, y, s=20, color=color, label=label)

def draw_gt_trajectory(ax, each_agent, tag):
    """ 绘制当前 agent 轨迹 """
    for i, (x, y) in enumerate(each_agent[:, [0, 1]]):
        color =  "green"  # 观测部分蓝色，预测部分绿色
        label = "GT" if i == 0 else None
        if tag == 0:
            ax.scatter(y, x, s=20, color=color, label=label)
        else:
            ax.scatter(x, y, s=20, color=color, label=label)

def draw_surrounding_agents(ax, sur, tag):
    """ 绘制周围行人的轨迹 """
    for each_sur_obs in sur:
        x = int(each_sur_obs[4])
        y = int(each_sur_obs[5])
        if tag == 0:
            ax.scatter(y, x, s=10, color='y', alpha=0.8)
        else:
            ax.scatter(x, y, s=10, color='y', alpha=0.8)

def draw_title_and_border(ax, img, traj_id, first_frame):
    """ 添加标题和边框 """
    title_text = f"Agent ID: {traj_id}, STS: {first_frame}"
    ax.set_title(title_text, fontsize=12, color='blue')
    ax.add_patch(plt.Rectangle((0, 0), img.shape[1], img.shape[0], 
                           edgecolor='black', linewidth=2, fill=False))

def draw_backbone_result(ax, backbone_result, tag):
    """ 绘制当前 agent 轨迹 """
    for i, (x, y) in enumerate(backbone_result[8:, :]):
        color = "grey" 
        label = "Cond" if i == 0 else None
        if tag == 0:
            ax.scatter(y, x, s=20, color=color, label=label)
        else:
            ax.scatter(x, y, s=20, color=color, label=label)

def draw_pcvae_result(ax, pcvae_result, tag):
    """ 绘制当前 agent 轨迹 """
    for i, (x, y) in enumerate(pcvae_result[:, [0, 1]]):
        color = "red"
        label = "BVAE" if i == 0 else None
        if tag == 0:
            ax.scatter(y, x, s=20, color=color, label=label)
        else:
            ax.scatter(x, y, s=20, color=color, label=label)

def draw_seg_result(ax, masks, scores, input_point, input_label, tag):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # fig, ax = plt.subplots(figsize=(10, 10))
        # 2) 绘制
        # ax.imshow(image)
        draw_mask(mask, ax)
        # draw_points(input_point, input_label, tag, ax)
        # ax.set_title(f"Mask {i}, Score: {score:.3f}", fontsize=18)
        # ax.axis('off')

def draw_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def draw_points(coords, labels, tag, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    # if tag == 0:
    #     ax.scatter(pos_points[:, 1], pos_points[:, 0], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    #     ax.scatter(neg_points[:, 1], neg_points[:, 0], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    # else:
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
   
    
def draw_box(ax, box,):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

def result_visualization(img, obs_infor, input_, output, tag):
    """ 统一调用各个绘制函数，在同一张图上绘制所有元素 """
    traj_id = int(obs_infor[0, 0])
    first_frame = int(input_[0, 1])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)  # 显示图像
    obs = input_[0:8,:]
    gt = input_[8:,:]
    # 调用各个绘制函数
    draw_obs_trajectory(ax, obs, tag) #input
    draw_gt_trajectory(ax, gt, tag) #output
    draw_pcvae_result(ax, output, tag) #output
    draw_title_and_border(ax, img, traj_id, first_frame)

    ax.axis("off")
    plt.legend()
    plt.show()

def normalize_data(data):
    """
    对输入数据进行标准化处理。
    假设 data 的形状为 (B, seq_len, feature_dim)，
    则在 B 和 seq_len 维度上计算均值和标准差。
    """
    mean = data.mean(dim=(0, 1), keepdim=True)
    std = data.std(dim=(0, 1), keepdim=True)
    return (data - mean) / (std + 1e-8), mean, std

def denormalize_data(norm_data, mean, std):
    """
    将标准化后的数据反标准化回原始数据尺度。
    """
    return norm_data * std + mean


def transform_coords(coords):
    """
    将坐标系变换为以第一个点为原点，
    且第一、二个点确定的新 x 轴方向。
    
    Args:
        coords: numpy array, shape (N, 2)，原始坐标点序列
    Returns:
        new_coords: numpy array, shape (N, 2)，变换后的坐标
    """
    # 1. 平移：将第一个点设为原点
    origin = coords[0]
    coords_shifted = coords - origin

    # 2. 旋转：计算第二个点的角度
    vec = coords_shifted[1]   # 新的第一、二个点之间的向量
    theta = np.arctan2(vec[1], vec[0])   # 计算角度

    # 构造旋转矩阵 (逆时针旋转 -theta)
    R = np.array([[np.cos(-theta), -np.sin(-theta)],
                  [np.sin(-theta),  np.cos(-theta)]])
    # 对所有点进行旋转
    new_coords = coords_shifted.dot(R.T)
    return new_coords

def project_trajectory_and_compute_offset_batch(trajs, threshold_deg=15):
    """
    对输入的多条轨迹（形状 [N, 20, 2]）进行处理：
      1. 对每条轨迹以第一个点为原点平移；
      2. 利用前两个点确定新坐标系（使得第二个点对应新 x 轴）；
      3. 将整条轨迹投影到该相对坐标系下；
      4. 计算每条轨迹最后一个点的 y 值（偏移量）、与 x 轴的夹角；
      5. 根据夹角与阈值判断方向编码：
            0 -> left, 1 -> forward, 2 -> right

    Args:
        trajs: numpy array, shape (N, 20, 2) —— 多条二维轨迹
        threshold_deg: 判断方向的阈值角度（单位：度），默认为15°
    Returns:
        proj_trajs: 投影后的轨迹，形状 (N, 20, 2)
        final_ys: 每条轨迹最后一点在新坐标系下的 y 值，形状 (N,)
        direction_codes: 每条轨迹的大致方向编码 (0: left, 1: forward, 2: right)，形状 (N,)
        angles: 每条轨迹最后一点与新坐标系 x 轴的夹角（弧度），形状 (N,)
    """
    N, T, _ = trajs.shape
    proj_trajs = np.zeros_like(trajs)   # 存放投影后的轨迹
    final_ys = np.zeros((N,))
    direction_codes = np.zeros((N,3), dtype=int)
    angles = np.zeros((N,))
    threshold = np.deg2rad(threshold_deg)

    for i in range(N):
        traj = trajs[i]  # shape (20, 2)
        # 1. 平移：以第一个点为原点
        origin = traj[0]
        traj_centered = traj - origin

        # 2. 确定新坐标系 x 轴：利用第一个到第二个点的向量
        vec = traj_centered[1]
        norm = np.linalg.norm(vec)
        if norm == 0:
            # 如果第二个点与第一个点重合，默认 x 轴方向为 (1,0)
            x_axis = np.array([1.0, 0.0])
        else:
            x_axis = vec / norm
        theta_align = np.arctan2(x_axis[1], x_axis[0])
        
        # 3. 构造旋转矩阵，将 x_axis 对齐到 (1, 0)
        R = np.array([[np.cos(-theta_align), -np.sin(-theta_align)],
                      [np.sin(-theta_align),  np.cos(-theta_align)]])
        proj = traj_centered.dot(R.T)  # shape (20, 2)
        proj_trajs[i] = proj

        # 4. 计算最后一点相对于新坐标系的偏移及夹角
        final_x, final_y = proj[-1]
        final_ys[i] = final_y
        angle = np.arctan2(final_y, final_x)  # 弧度
        angles[i] = angle

        # 5. 根据夹角判断方向编码
        if np.abs(angle) < threshold:
            direction_codes[i] = np.array([0, 1, 0])   # forward
        elif angle >= threshold:
            direction_codes[i] = np.array([1, 0, 0])   # left
        else:
            direction_codes[i] = np.array([0, 0, 1])   # right

    return proj_trajs, final_ys, direction_codes, angles

def rotate90(data):
    """
    对输入的 Nx20x4 数据进行 90 度逆时针旋转。
    最后一维为 [frame, id, x, y]，仅旋转 (x,y)。
    (x, y)  ->  (-y, x)
    """
    rotated = data.copy()
    x = data[..., 2]
    y = data[..., 3]
    rotated[..., 2] = -y  # 新 x = -y
    rotated[..., 3] =  x  # 新 y =  x
    return rotated

def rotate180(data):
    """
    对输入的 Nx20x4 数据进行 180 度逆时针旋转。
    (x, y)  ->  (-x, -y)
    """
    rotated = data.copy()
    x = data[..., 2]
    y = data[..., 3]
    rotated[..., 2] = -x
    rotated[..., 3] = -y
    return rotated

def rotate270(data):
    """
    对输入的 Nx20x4 数据进行 270 度逆时针旋转。
    (x, y)  ->  (y, -x)
    """
    rotated = data.copy()
    x = data[..., 2]
    y = data[..., 3]
    rotated[..., 2] =  y
    rotated[..., 3] = -x
    return rotated

import cv2

def read_frame_from_avi(video_path: str, frame_index: int):
    """
    从 AVI 视频中读取指定帧

    :param video_path: AVI 文件路径
    :param frame_index: 要读取的帧索引（0-based）
    :return: 读取到的帧（numpy.ndarray），若失败返回 None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"视频总帧数为 {total_frames}")
    # 定位到目标帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()
    cap.release()

    if not success:
        print(f"读取第 {frame_index} 帧失败。")
        return None

    return frame

def load_all_scenarios(data_path):
    """ 加载所有场景数据，并合并为单个训练集 """
    all_data = []
    all_tag = []
    for sec_txt_path in data_path:
        # if len(sec_txt_path) == 1:
        #     sec_txt_path = sec_txt_path[0]
        #     sec_img_path = sec_img_path[0]
        data_loader = ETHLoader(sec_txt_path)  # 加载单个场景
        single_data = data_loader.kinematics_features  # 读取数据
        tag = data_loader.xy_tag
        all_data.append(torch.tensor(single_data, dtype=torch.float32))  
        all_tag.append(tag)  
    # # 合并所有场景数据
    # all_datas = torch.cat(all_datas, dim=0)  # [Total_Samples, obs_len, 2]
    return all_data, all_tag

def get_expanded_box(obs_trajectory, image, noise_range=(1.0, 1.5)):

    pts = np.asarray(obs_trajectory, dtype=np.float32)
    xs, ys = pts[:, 0], pts[:, 1]

    # 计算最小包围框
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    half_w = (x_max - x_min) / 2.0
    half_h = (y_max - y_min) / 2.0
    # 随机扩展倍数
    min_s, max_s = noise_range
    scale = np.random.uniform(min_s, max_s)
    # 扩展后的半宽高
    new_half_w = half_w * scale
    new_half_h = half_h * scale
    # 新坐标
    new_x_min = cx - new_half_w
    new_x_max = cx + new_half_w
    new_y_min = cy - new_half_h
    new_y_max = cy + new_half_h
    # 图像尺寸（高度 H、宽度 W）
    H, W = image.shape[:2]

    # 裁剪到图像边界
    x0 = int(np.round(max(0, new_x_min)))
    y0 = int(np.round(max(0, new_y_min)))
    x1 = int(np.round(min(W - 1, new_x_max)))
    y1 = int(np.round(min(H - 1, new_y_max)))

    # 转成整数并返回
    return np.array([x0, y0, x1, y1])

def visualize_inputs(img, prompt_box, prompt_points, gts, sec, agent_id, save_dir):
    """
    在一个 Figure 里展示 5 个子图：
    img:        [1,3,H,W]
    dense_mask: [1,1,H,W]
    boxs:       [1,4]
    points:     [1,8,2]
    gt:         [1,1,H,W]
    """
    # 转为 numpy
    # print(dense_mask.shape)
    # print(img.shape)
    # print(prompt_box.shape)
    # print(prompt_points.shape)
    # print(gts.shape)
    gts      = gts[0]
    gt_min, gt_max = gts.min(), gts.max()
    # print(f"gts 最小值: {gt_min:.4f}, 最大值: {gt_max:.4f}")
    x0, y0, x1, y1 = prompt_box.astype(int)
    w_box, h_box   = x1 - x0, y1 - y0
    pts_x, pts_y   = prompt_points[:,0], prompt_points[:,1]
    # print(whole_np.shape)
    titles = ["Image Input", "Heat GT ", "Grey GT "]
    panels = [img, gts, gts]
    rows, cols = 1, 3
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    for ax, panel, title in zip(axes, panels, titles):
        if title == "Image Input":
            # 彩色原图
            ax.imshow(panel)
            ax.scatter(
            pts_x[0:8], pts_y[0:8],
            s=20, c="yellow", edgecolors="black", linewidths=1
            )   
            ax.scatter(
                pts_x[8:], pts_y[8:],
                s=20, c="purple", edgecolors="black", linewidths=1
            )
        elif title.startswith("Heat GT"):
            # 灰度风格的 GT
            ax.imshow(panel, cmap="hot",)
            ax.scatter(
            pts_x[0:8], pts_y[0:8],
            s=20, c="yellow", edgecolors="black", linewidths=1
        )
        else:
            lo = np.percentile(panel, 50)
            hi = panel.max()
            ax.imshow(panel, cmap="grey", vmin=lo, vmax=hi)
            ax.scatter(
            pts_x[0:8], pts_y[0:8],
            s=20, c="yellow", edgecolors="black", linewidths=1
        )

        rect = patches.Rectangle(
            (x0, y0), w_box, h_box,
            edgecolor="red", facecolor="none", linewidth=2
        )
        ax.add_patch(rect)

        # # 叠加 points
        
        # ax.scatter(
        #     pts_x[8:], pts_y[8:],
        #     s=20, c="red", edgecolors="black", linewidths=1
        # )
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{sec}_ID_{agent_id}_inputs"+'.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {out_path}")