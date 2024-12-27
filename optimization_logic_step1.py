# optimization_logic.py
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist



def optimize_region_assignment(
    result_df_orig,
    distance_matrix,
    fte_lower_bound=0.85,
    lambda_fte=4.0,
    lambda_dist=1.0,
    time_limit=30
):
    """
    优化区域分配的核心函数
    
    参数:
    - result_df_orig: DataFrame, 包含city和fte列的原始数据
    - distance_matrix: DataFrame, 城市间距离矩阵
    - fte_lower_bound: float, FTE下限
    - lambda_fte: float, FTE权重
    - lambda_dist: float, 距离权重
    - time_limit: int, 优化时间限制（秒）
    
    返回:
    - dict: 包含优化结果的字典
    """
    
    # 提取数据
    cities = result_df_orig['city'].tolist()
    fte = result_df_orig['fte'].tolist()
    distance_matrix_np = distance_matrix.loc[cities, cities].to_numpy()
    
    num_groups = len(cities)
    
    # 创建模型
    model = gp.Model("Clustering")
    
    # 决策变量
    x = model.addVars(len(cities), num_groups, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_groups, vtype=GRB.CONTINUOUS, name="y")
    rounded_fte = model.addVars(num_groups, vtype=GRB.INTEGER, name="rounded_fte")
    z = model.addVars(num_groups, vtype=GRB.BINARY, name="z")
    group_distance = model.addVars(num_groups, vtype=GRB.CONTINUOUS, name="group_distance")
    
    # 目标函数
    model.setObjective(
        lambda_fte * gp.quicksum(y[j] for j in range(num_groups)) +
        lambda_dist * gp.quicksum(group_distance[j] for j in range(num_groups)),
        GRB.MINIMIZE
    )
    
    # 约束1：每个城市必须分配到一个组
    for i in range(len(cities)):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_groups)) == 1)
    
    # 约束2：计算每个组的FTE和
    group_fte = {}
    for j in range(num_groups):
        group_fte[j] = gp.quicksum(fte[i] * x[i, j] for i in range(len(cities)))
        model.addConstr(y[j] >= group_fte[j] - rounded_fte[j])
        model.addConstr(y[j] >= rounded_fte[j] - group_fte[j])
    
    # 约束3：组激活约束
    for j in range(num_groups):
        model.addConstr(gp.quicksum(x[i, j] for i in range(len(cities))) <= len(cities) * z[j])
        model.addConstr(gp.quicksum(x[i, j] for i in range(len(cities))) >= z[j])
        model.addConstr(group_fte[j] >= fte_lower_bound * z[j])
    
    # 约束4：计算组内距离和
    for j in range(num_groups):
        model.addConstr(
            group_distance[j] == gp.quicksum(
                distance_matrix_np[i, k] * x[i, j] * x[k, j]
                for i in range(len(cities))
                for k in range(i + 1, len(cities))
            )
        )
    
    # 设置求解参数
    model.setParam('Threads', 7)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Heuristics', 1)
    model.setParam('ImproveStartGap', 0.5)
    model.setParam('RINS', 5)
    model.setParam('MIPFocus', 1)
    model.setParam('ImproveStartTime', 0)
    model.setParam('PreSparsify', 1)
    
    # 求解模型
    model.optimize()
    
     #处理结果
    results = {
        'status': model.status,
        'groups': [],
        'objective_value': float('inf'),
        'model': model,  # 添加model
        'cities': cities,  # 添加cities
        'x': x,  # 添加决策变量x
        'num_groups': num_groups,  # 添加组数
        'group_fte': group_fte,  # 添加组FTE
        'group_distance': group_distance  # 添加组距离
    }

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        results['objective_value'] = model.objVal
        
        # 输出和结果收集
        print("Optimization Results:")
        for j in range(num_groups):
            members = [cities[i] for i in range(len(cities)) if x[i, j].x > 0.5]
            
            if members:  # 只处理非空组
                group_info = {
                    'members': members,
                    'fte_sum': group_fte[j].getValue(),
                    'distance_sum': group_distance[j].x,
                    'y_value': y[j].x
                }
                
                results['groups'].append(group_info)
                
                # 打印每个组的详细信息
                print(f"Group {j+1}: {members}")
                print(f"  FTE Sum: {group_info['fte_sum']:.2f}")
                print(f"  Distance Sum: {group_info['distance_sum']:.2f}")
                print(f"  Group Activation (y): {group_info['y_value']:.2f}\n")
    else:
        print("No optimal solution found.")
        results['groups'] = []
    
    return results

def format_optimization_results(results,df_orig):
    """
    格式化优化结果
    
    参数:
    results: dict, 包含优化结果的字典
    
    返回:
    DataFrame: 更新后的数据框，包含分组信息
    """
    model = results['model']
    cities = results['cities']
    x = results['x']
    num_groups = results['num_groups']
    group_fte = results['group_fte']
    group_distance = results['group_distance']

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        grouped_clusters = {}
        for j in range(num_groups):
            members = [cities[i] for i in range(len(cities)) if x[i, j].x > 0.5]
            
            if members:
                if len(members) > 1:
                    grouped_cluster_name = '&'.join(sorted(members))
                    for member in members:
                        grouped_clusters[member] = grouped_cluster_name
                else:
                    grouped_clusters[members[0]] = members[0]

        df_orig['db_cluster'] = df_orig['city'].map(grouped_clusters)
        df_orig['MR Pos'] = df_orig['db_cluster']

        # 打印分组信息
        for j in range(num_groups):
            members = [cities[i] for i in range(len(cities)) if x[i, j].x > 0.5]
            if members:
                print(f"Group {j+1}: {members}, FTE Sum: {group_fte[j].getValue():.2f}, "
                      f"Distance Sum: {group_distance[j].x:.2f}")

        print("\n分组结果:")
        for city, group_name in grouped_clusters.items():
            print(f"{city}: {group_name}")
    else:
        print("No optimal solution found.")
        df_orig['db_cluster'] = None
        df_orig['MR Pos'] = None

    return df_orig