import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict

def calculate_inter_city_distances(df, method='min'):
    """
    计算城市间距离矩阵。
    
    参数：
    df: DataFrame，包含 'city', 'latitude', 和 'longitude' 列。
    method: str，计算方法，可选 'min'（默认）, 'centroid', 或 'average'。
        - 'min': 使用两个城市中医院位置的最小距离。
        - 'centroid': 使用城市中心（医院坐标的平均值）之间的距离。
        - 'average': 使用两城市所有医院位置对之间距离的平均值。
    
    返回：
    DataFrame，表示城市间的距离矩阵。
    """
    # 获取城市列表和医院坐标分组
    cities = sorted(df['city'].unique())
    city_groups = df.groupby('city')
    
    # 根据城市计算中心点坐标
    city_center_coords = {
        city: city_groups.get_group(city)[['latitude', 'longitude']].mean().values
        for city in cities
    }
    
    # 初始化距离矩阵
    distance_matrix = np.zeros((len(cities), len(cities)))
    
    # 计算距离矩阵
    for i, city1 in enumerate(cities):
        for j, city2 in enumerate(cities):
            if i == j:
                continue
            city1_coords = city_groups.get_group(city1)[['latitude', 'longitude']].values
            city2_coords = city_groups.get_group(city2)[['latitude', 'longitude']].values
            
            if method == 'min':
                # 使用最小距离
                distances = cdist(city1_coords, city2_coords, metric='euclidean')
                distance_matrix[i, j] = np.min(distances)
            elif method == 'centroid':
                # 使用中心点之间的距离
                city1_centroid = city_center_coords[city1]
                city2_centroid = city_center_coords[city2]
                distance_matrix[i, j] = np.linalg.norm(city1_centroid - city2_centroid)
            elif method == 'average':
                # 使用平均距离
                distances = cdist(city1_coords, city2_coords, metric='euclidean')
                distance_matrix[i, j] = np.mean(distances)
            else:
                raise ValueError("Invalid method. Choose from 'min', 'centroid', or 'average'.")
        # 创建距离矩阵的 DataFrame
    distance_df = pd.DataFrame(distance_matrix, index=cities, columns=cities)
    return distance_df



def generate_initial_solution(df, n_territories, weight=0.1, random_state=42):
    """
    使用 K-Means 聚类生成初始解。
    
    参数:
        df (pd.DataFrame): 包含以下列的数据框：
            - 'group': 分组标识
            - 'city': 城市标识
            - 'latitude': 纬度
            - 'longitude': 经度
        n_territories (int): 要划分的区域数。
        weight (float): 城市平均坐标的权重 (默认为 0.1)。
        random_state (int): 随机种子，确保结果可重复 (默认为 42)。
    
    返回:
        pd.DataFrame: 初始分配矩阵，行表示区域，列表示分组。
    """
    # 获取唯一的分组
    groups = df['group'].unique()
    
    # 计算每个城市的平均经纬度
    city_avg = df.groupby('city')[['latitude', 'longitude']].mean().reset_index()
    city_avg_dict = city_avg.set_index('city')[['latitude', 'longitude']].to_dict('index')
    
    # 为每一行添加所属城市的平均经纬度
    city_avg_coords = np.array([
        [city_avg_dict[row['city']]['latitude'], city_avg_dict[row['city']]['longitude']]
        for _, row in df.iterrows()
    ])
    
    # 组合原始坐标和城市平均坐标
    X = np.hstack([
        df[['latitude', 'longitude']].values,  # 原始坐标
        city_avg_coords * weight  # 加权后的城市平均坐标
    ])
    
    # 使用增强后的特征进行 K-Means 聚类
    kmeans = KMeans(n_clusters=n_territories, random_state=random_state)
    kmeans.fit(X)
    territory_centroids = kmeans.cluster_centers_
    
    # 初始化分配矩阵
    initial_solution = pd.DataFrame(0, index=range(n_territories), columns=groups)
    
    # 将每个 group 分配到最近的 territory
    for i in range(n_territories):
        for j, group in enumerate(groups):
            # 获取 group 的位置信息
            group_idx = df['group'] == group
            group_location = np.hstack([
                df.loc[group_idx, ['latitude', 'longitude']].values[0],
                city_avg_coords[group_idx][0] * weight
            ])
            # 计算到所有 territory 中心的距离
            distances = np.linalg.norm(group_location - territory_centroids, axis=1)
            initial_solution.iloc[i, j] = 1 if np.argmin(distances) == i else 0
    
    return initial_solution



def calculate_inter_city_distances(df, method='min'):
    """
    计算城市间距离矩阵。
    
    参数：
    df: DataFrame，包含 'city', 'latitude', 和 'longitude' 列。
    method: str，计算方法，可选 'min'（默认）, 'centroid', 或 'average'。
        - 'min': 使用两个城市中医院位置的最小距离。
        - 'centroid': 使用城市中心（医院坐标的平均值）之间的距离。
        - 'average': 使用两城市所有医院位置对之间距离的平均值。
    
    返回：
    DataFrame，表示城市间的距离矩阵。
    """
    # 获取城市列表和医院坐标分组
    cities = sorted(df['city'].unique())
    city_groups = df.groupby('city')
    
    # 根据城市计算中心点坐标
    city_center_coords = {
        city: city_groups.get_group(city)[['latitude', 'longitude']].mean().values
        for city in cities
    }
    
    # 初始化距离矩阵
    distance_matrix = np.zeros((len(cities), len(cities)))
    
    # 计算距离矩阵
    for i, city1 in enumerate(cities):
        for j, city2 in enumerate(cities):
            if i == j:
                continue
            city1_coords = city_groups.get_group(city1)[['latitude', 'longitude']].values
            city2_coords = city_groups.get_group(city2)[['latitude', 'longitude']].values
            
            if method == 'min':
                # 使用最小距离
                distances = cdist(city1_coords, city2_coords, metric='euclidean')
                distance_matrix[i, j] = np.min(distances)
            elif method == 'centroid':
                # 使用中心点之间的距离
                city1_centroid = city_center_coords[city1]
                city2_centroid = city_center_coords[city2]
                distance_matrix[i, j] = np.linalg.norm(city1_centroid - city2_centroid)
            elif method == 'average':
                # 使用平均距离
                distances = cdist(city1_coords, city2_coords, metric='euclidean')
                distance_matrix[i, j] = np.mean(distances)
            else:
                raise ValueError("Invalid method. Choose from 'min', 'centroid', or 'average'.")
        # 创建距离矩阵的 DataFrame
    distance_df = pd.DataFrame(distance_matrix, index=cities, columns=cities)
    return distance_df
 
def find_nearby_cities(distance_matrix, max_distance=1.0, top_n=3):
    """
    找出每个城市的临近城市，并按距离排序
    参数:
    - distance_matrix: 城市间距离矩阵
    - max_distance: 距离阈值
    - top_n: 选择最近的N个城市
    返回:
    字典，键是城市，值是按距离升序排列的临近城市列表
    """
    nearby_cities = {}
    for city in distance_matrix.index:
        # 获取当前城市到其他城市的距离
        distances = distance_matrix.loc[city]
        # 排除自身，并按距离排序
        city_distances = distances[distances.index != city].sort_values()
        # 筛选方法1：基于距离阈值
        nearby_by_threshold = city_distances[city_distances <= max_distance]
        # 筛选方法2：选择最近的top_n个城市
        nearby_by_top_n = city_distances.head(top_n)
        
        # 合并两种方法的结果（去重）并按距离排序
        combined_nearby = pd.concat([nearby_by_threshold, nearby_by_top_n]).drop_duplicates()
        nearby_with_distances = combined_nearby.sort_values()
        
        # 存储城市名称和对应的距离
        # nearby_cities[city] = list(zip(nearby_with_distances.index, nearby_with_distances.values))
        nearby_cities[city] = list(nearby_with_distances.index)
    return nearby_cities
 
def calculate_city_hospital_distances(df, group_col):
    """
    计算每个分组内（如城市）医院之间的距离统计信息
    
    参数:
    df: 包含医院信息的DataFrame，需要有分组列（如'city'）、'latitude'、'longitude'列
    group_col: 分组列名（如'city'，表示按哪个列进行分组）
    
    返回:
    一个字典，包含每个分组的最大距离和平均距离
    """
    city_distance_stats = {}
    
    # 按指定列分组
    for group, group_df in df.groupby(group_col):
        # 如果分组内医院数量少于2个，跳过
        if len(group_df) < 2:
            city_distance_stats[group] = {
                'max_distance': 0,
                'avg_distance': 0
            }
            continue
        
        # 获取分组内所有医院的坐标
        group_coords = group_df[['latitude', 'longitude']].values
        
        # 计算所有医院两两之间的距离
        distances = cdist(group_coords, group_coords, metric='euclidean')
        
        # 由于距离矩阵是对称的，且对角线为0，我们只取上三角矩阵
        upper_triangle_distances = distances[np.triu_indices_from(distances, k=1)]
        
        # 计算最大距离和平均距离
        max_distance = np.max(upper_triangle_distances)
        avg_distance = np.mean(upper_triangle_distances)
        
        # 存储结果
        city_distance_stats[group] = {
            'max_distance': max_distance,
            'avg_distance': avg_distance
        }
    
    return city_distance_stats


# 函数：判断点是否在多边形内
def is_point_in_polygon(point, polygon):
    """
    判断一个点是否在多边形内。
    使用射线法：统计从点发出的水平射线与多边形边的交点个数。
    
    参数:
        point (tuple): 点的坐标 (x, y)。
        polygon (np.array): 多边形的顶点坐标数组 [[x1, y1], [x2, y2], ...]。

    返回:
        bool: True 如果点在多边形内，否则 False。
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]  # 环形连接
        
        # 判断射线是否穿过边
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    
    return inside

# 函数：计算多边形重叠面积
def compute_overlap_area(hull1, hull2):
    """
    计算两个多边形的重叠面积。

    参数:
        hull1, hull2 (np.array): 每个多边形的顶点坐标数组 [[x1, y1], [x2, y2], ...]。

    返回:
        float: 重叠面积。
    """
    # 获取两组多边形的所有顶点
    points1 = hull1
    points2 = hull2

    # 获取两组点中的候选交集点
    intersection_points = []
    
    # 1. 添加每组点中位于对方多边形内部的点
    intersection_points.extend([p for p in points1 if is_point_in_polygon(p, points2)])
    intersection_points.extend([p for p in points2 if is_point_in_polygon(p, points1)])
    
    # 2. 计算两多边形边的交点
    n1, n2 = len(points1), len(points2)
    for i in range(n1):
        for j in range(n2):
            edge1_start, edge1_end = points1[i], points1[(i + 1) % n1]
            edge2_start, edge2_end = points2[j], points2[(j + 1) % n2]
            inter = line_segment_intersection(edge1_start, edge1_end, edge2_start, edge2_end)
            if inter is not None:
                intersection_points.append(inter)

    # 如果没有交集点，则重叠面积为 0
    if len(intersection_points) < 3:
        return 0.0
    
    # 3. 计算交集点的凸包
    intersection_points = np.unique(intersection_points, axis=0)  # 去重
    hull = ConvexHull(intersection_points)
    return polygon_area(intersection_points[hull.vertices])

# 函数：计算两线段的交点
def line_segment_intersection(p1, p2, q1, q2):
    """
    计算两线段的交点。

    参数:
        p1, p2 (tuple): 第一条线段的两个端点 (x1, y1), (x2, y2)。
        q1, q2 (tuple): 第二条线段的两个端点 (x3, y3), (x4, y4)。

    返回:
        tuple or None: 如果有交点，返回交点坐标 (x, y)，否则返回 None。
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2

    # 计算分母
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # 平行，无交点

    # 计算交点坐标
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    # 检查交点是否在线段范围内
    if (
        min(x1, x2) <= px <= max(x1, x2)
        and min(y1, y2) <= py <= max(y1, y2)
        and min(x3, x4) <= px <= max(x3, x4)
        and min(y3, y4) <= py <= max(y3, y4)
    ):
        return (px, py)
    return None

# 函数：计算多边形面积
def polygon_area(coords):
    """
    计算多边形面积。
    
    参数:
        coords (np.array): 多边形的顶点坐标数组 [[x1, y1], [x2, y2], ...]。

    返回:
        float: 多边形面积。
    """
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def calculate_and_sort_overlaps(city_hulls):
    """
    计算每对分组间的重叠面积，并按重叠面积降序排序。

    参数:
        city_hulls (dict): 每个分组对应的凸包顶点坐标字典，格式为 {group_name: np.array(coords)}。

    返回:
        pd.DataFrame: 包含分组对及其重叠面积的 DataFrame，按 OverlapArea 降序排列。
    """
    overlap_data = []
    group_names = list(city_hulls.keys())
    
    # 计算重叠面积
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            group1, group2 = group_names[i], group_names[j]
            overlap = compute_overlap_area(city_hulls[group1], city_hulls[group2])
            if overlap > 0:
                overlap_data.append({'Group1': group1, 'Group2': group2, 'OverlapArea': overlap})
    
    # 转为 DataFrame 并排序
    overlap_df = pd.DataFrame(overlap_data)
    overlap_df = overlap_df.sort_values(by='OverlapArea', ascending=False).reset_index(drop=True)
    
    return overlap_df

# 函数：计算最小凸包，支持自定义分组列
def compute_convex_hulls(df, col= 'city'):
    city_hulls = {}
    for group_name, group in df.groupby(col):
        coords = group[['longitude', 'latitude']].to_numpy()
        if len(coords) < 3:
            print(f"Group {group_name} has less than 3 points, skipping convex hull.")
            continue
        hull = ConvexHull(coords)
        hull_coords = coords[hull.vertices]  # 获取凸包顶点坐标
        city_hulls[group_name] = hull_coords
    return city_hulls


def analyze_hco_groups(df):
    """
    Analyze HCO groups from a given DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing group, productivity, and other relevant columns
    
    Returns:
    --------
    tuple containing:
    - individual : pandas.DataFrame
        Normalized group representation for each MR Pos
    - group_productivity : list
        Total productivity for each group
    - group_productivity_ly : list
        Total last year productivity for each group
    - group_fte : list
        Total FTE for each group
    - group_potential : list
        Total potential for each group
    - group_TH : list
        Total count for each group
    - group_city_matrix : numpy.ndarray
        Boolean matrix indicating city presence in each group
    """
    # Initialize group-to-HCO and HCO-to-group mappings
    group2hco = defaultdict(set)
    hco2group = {}
    for i, row in df.iterrows():
        group2hco[row['group']].add(row['group'])
        hco2group[row['group']] = row['group']
    
    # Sort groups
    group_sorted = sorted(group2hco.keys())
    
    # Calculate group-level metrics
    group_productivity = [df[df['group']==group]['productivity'].sum() for group in group_sorted]
    group_productivity_ly = [df[df['group']==group]['productivity_ly'].sum() for group in group_sorted]
    group_fte = [df[df['group']==group]['fte'].sum() for group in group_sorted]
    group_potential = [df[df['group']==group]['potential'].sum() for group in group_sorted]
    group_center = [df[df['group']==group][['latitude', 'longitude']].values.mean(axis=0) for group in group_sorted]
    group_TH = [sum(df['group']==group) for group in group_sorted]
    
    # City analysis
    group_city = [set(df[df['group']==group]['city']) for group in group_sorted]
    
    # Create city-to-ID mapping
    all_cities = set.union(*group_city)
    city_to_id = {city: i for i, city in enumerate(all_cities)}
    
    # Create city presence matrix
    group_city_matrix = np.zeros((len(group_sorted), len(all_cities)), dtype=bool)
    for i, cities in enumerate(group_city):
        for city in cities:
            group_city_matrix[i, city_to_id[city]] = True
    
    # Create individual representation
    n_lp = len([(i, list(_df['group'])) for i, _df in df.groupby('MR Pos')])
    individual = pd.DataFrame(np.zeros((n_lp, len(group_sorted))), columns=group_sorted)
    
    i = 0
    for _, _df in df.groupby('MR Pos'):
        for hco in list(_df['group']):
            if hco in group_sorted:
                individual.loc[i, hco] = 1
            elif hco in hco2group:
                individual.loc[i, hco2group[hco]] = 1
        i += 1
    
    # Normalize individual representation
    individual = individual / individual.sum()
    
    return (
        group2hco,
        individual, 
        group_productivity, 
        group_productivity_ly, 
        group_fte, 
        group_potential, 
        group_TH, 
        group_city_matrix
    )

def fitness(individual, group_fte, group_productivity, group_productivity_ly, 
           group_potential, group_TH, group_city_matrix, group2hco,
           avg_productivity_territory, avg_productivity_growth, 
           avg_potential_territory, avg_TH, df,
           detail=False, radius_base=0.5, travel_base=1.0):
    """
    计算适应度分数
    """
    # 1. 工作量得分
    territory_fte = individual.dot(group_fte)
    workload_score = np.mean([x > 0.85 and x < 1.15 for x in territory_fte])
    workload_penalty = - 0.01 * sum([sum([x < 0.85-0.05*i or x > 1.15+0.05*i for x in territory_fte]) for i in range(1,18)])
    workload_score_over80 = 1 if workload_score >= 0.8 else 0
    
    workload_total = np.clip(sum([3*workload_score, 1*workload_penalty, workload_score_over80]), -4, 4)
    
    # 2. 生产力得分和增长得分
    territory_productivity = individual.dot(group_productivity) / avg_productivity_territory
    productivity_score = np.mean([x > 0.5 and x < 1.5 for x in territory_productivity])
    
    territory_productivity_gr = individual.dot(group_productivity) / individual.dot(group_productivity_ly)
    productivity_penalty_doublelow = - 2 * np.mean([pr < 0.7 and pr > 0.5 and gr < avg_productivity_growth 
                                                   for pr, gr in zip(territory_productivity, territory_productivity_gr)])
    productivity_penalty_lowpr = - 3 * np.mean([ pr <= 0.5 for pr in territory_productivity])
    
    productivity_total = np.clip(sum([3.0* productivity_score, productivity_penalty_doublelow, productivity_penalty_lowpr]), -3, 3)
    
    # 3. 潜力得分
    territory_potential = individual.dot(group_potential) / avg_potential_territory
    potential_score = np.mean([x > 0.5 and x < 3 for x in territory_potential])
    potential_total = np.clip(sum([potential_score]), 0, 1)
    
    # 4. 距离得分
    territory_distance = individual.apply(
        lambda row: hcos2distance(
            set([ _ for x in row.index[row > 0].tolist() 
                for _ in group2hco[x]]), df),axis=1)
    avg_territory_distance = territory_distance.mean()
    
    territory_radius = individual.apply(
        lambda row: hcos2distance(
            set([ _ for x in row.index[row > 0].tolist() 
                for _ in group2hco[x]]), df, type='radius'),axis=1)
    avg_territory_radius = territory_radius.mean()
    
    CAP_SCORE = 1.5
    if avg_territory_distance:
        distance_score_radius = np.mean([min(radius_base * i/avg_territory_radius, CAP_SCORE) for i in [0.8, 1.0, 1.2]])
        distance_score_travel = np.mean([min(travel_base * i/avg_territory_distance, CAP_SCORE) for i in [0.8, 1.0, 1.2]])
    else:
        distance_score_radius = CAP_SCORE
        distance_score_travel = CAP_SCORE
    
    distance_penalty_radius = - 0.01 * sum([sum([x > 0.2+0.1*i for x in territory_radius]) for i in range(1,50)]) - 0.05 * sum([sum([x > 1.0+0.1*i for x in territory_radius]) for i in range(1,50)])
    distance_penalty_travel = - 0.01 * sum([sum([x > 0.5+0.25*i for x in territory_distance]) for i in range(1,50)])
    
    distance_total = np.clip(sum([distance_score_radius, distance_score_travel, distance_penalty_radius, distance_penalty_travel]), -2.5, 2.5)
    
    territory_city_matrix = np.dot(individual.values > 0, group_city_matrix)
    num_city = np.sum(territory_city_matrix > 0, axis=1)
    distance_penalty_city = - 0.05 * sum([max(sum(num_city) - len(num_city)-i, 0) for i in [3,2,1,0]])
    distance_penalty_city += - np.sum(num_city >= 4)
    
    territory_TH = individual.dot(group_TH)
    distance_penalty_city += - np.sum(territory_TH > 2*avg_TH)
    
    score_total = sum([workload_total, productivity_total, potential_total, distance_total, distance_penalty_city])
    scores = [workload_score, workload_penalty, workload_score_over80,
              productivity_score, productivity_penalty_doublelow, productivity_penalty_lowpr,
              potential_score,
              distance_score_radius, distance_score_travel, distance_penalty_radius, distance_penalty_travel, distance_penalty_city]
    
    if detail:
        return tuple([score_total] + scores)
    else:
        return score_total

def hcos2distance(hcos, df, fte_weight=True, type='mst'):
    """
    计算HCOs之间的距离
    """
    df_hco = df[df['group'].isin(hcos)]
    coordinates = df_hco[['latitude', 'longitude']].values
    if fte_weight:
        weights = np.clip(df_hco['fte'].values, 0.05, 1)
        coordinates_center = (coordinates.T * weights).T.sum(axis=0) / weights.sum() 
    else:
        coordinates_center = coordinates.mean(axis=0) 
    
    if type == 'mst':
        all_coordinates = np.vstack([coordinates_center, coordinates])
        distances = pdist(all_coordinates)
        distance_matrix = squareform(distances)
        mst = minimum_spanning_tree(distance_matrix)
        return mst.sum()
    else:
        distances = np.sqrt(np.sum((coordinates - coordinates_center) ** 2, axis=1))
        if type == 'radius':
            return distances.max()
        else:
            return distances.sum()

def individual_summary(individual, group_fte, group_productivity, group_productivity_ly, 
                      group_potential, group_TH, group_city_matrix, group2hco,
                      avg_productivity_territory, avg_TH, avg_potential_territory,df, workload_per_fte):
    """
    生成个体摘要信息
    """
    mr_pos_list = df.groupby('MR Pos')['MR Pos'].first().tolist()
    
    territory_assignment = hco_assignment(individual, group2hco, df)
    territory_workload = round(individual.dot(group_fte) * workload_per_fte)
    territory_fte = individual.dot(group_fte)
    territory_TH = individual.dot(group_TH) / avg_TH
    territory_productivity = individual.dot(group_productivity)
    territory_productivity_rate = territory_productivity / avg_productivity_territory
    territory_productivity_gr = individual.dot(group_productivity) / individual.dot(group_productivity_ly)
    territory_potential = individual.dot(group_potential)
    territory_potential_rate = territory_potential / avg_potential_territory
    
    territory_distance = individual.apply(
        lambda row: hcos2distance(
            set([ _ for x in row.index[row > 0].tolist() 
                for _ in group2hco[x]]), df), axis=1) * 100
                
    territory_radius = individual.apply(
        lambda row: hcos2distance(
            set([ _ for x in row.index[row > 0].tolist() 
                for _ in group2hco[x]]), df, type='radius'), axis=1) * 100
                
    territory_city_matrix = np.dot(individual.values > 0, group_city_matrix)
    num_city = np.sum(territory_city_matrix > 0, axis=1)
    
    result = pd.DataFrame({
        'MR Pos': mr_pos_list,
        'assignment': territory_assignment,
        'workload': territory_workload,
        'fte': territory_fte,
        'TH_rate': territory_TH, 
        'productivity': territory_productivity,
        'productivity_rate': territory_productivity_rate,
        'productivity_growth': territory_productivity_gr,
        'potential': territory_potential,
        'potential_rate': territory_potential_rate,
        'distance': territory_distance,
        'radius': territory_radius,
        'city_count': num_city,
    })
    
    return result

def hco_assignment(individual, group2hco, df):
    """
    根据分配结果，输出每个territory分配的hco
    """
    territory_hcos = individual.apply(
        lambda row: ','.join([f"{hco_nm}_{hco_cd}_{round(prop, 3)}" 
                            for hco_nm, hco_cd, prop in sorted([
                                (df[df['group']==hco]['hco_nm'].values[0], 
                                 hco, 
                                 prop*df[df['group']==hco]['fte'].values[0]) 
                                for group, prop in row[row > 0].items()
                                for hco in group2hco[group]
                            ], key=lambda x: x[2], reverse=True)]),
        axis=1)
    return territory_hcos

def fitness_detail(individual, group_fte, group_productivity, group_productivity_ly, 
                  group_potential, group_TH, group_city_matrix, group2hco,
                  avg_productivity_territory, avg_productivity_growth, 
                  avg_potential_territory, avg_TH, df,
                  radius_base=0.5, travel_base=1.0):
    """
    计算并返回详细的适应度评分
    """
    result = fitness(
        individual, group_fte, group_productivity, group_productivity_ly,
        group_potential, group_TH, group_city_matrix, group2hco,
        avg_productivity_territory, avg_productivity_growth,
        avg_potential_territory, avg_TH, df,
        detail=True, radius_base=radius_base, travel_base=travel_base
    )
    return 'score:%f; workload_score: %.3f; workload_penalty: %.3f; workload_score_over80: %.3f; productivity_score: %.3f; productivity_penalty_doublelow: %.3f; productivity_penalty_lowpr: %.3f; potential_score: %.3f; distance_score_radius: %.3f; distance_score_travel: %.3f; distance_penalty_radius: %.3f; distance_penalty_travel: %.3f; distance_penalty_city: %.3f' % result