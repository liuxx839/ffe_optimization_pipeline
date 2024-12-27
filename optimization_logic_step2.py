import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans
from util_function import generate_initial_solution
from gurobipy import *


# 定义状态含义的映射
status_meaning = {
    GRB.OPTIMAL: "Optimal solution found",        # 最优解
    GRB.INFEASIBLE: "Model is infeasible",        # 模型不可行
    GRB.INF_OR_UNBD: "Infeasible or unbounded",   # 不可行或无界
    GRB.UNBOUNDED: "Model is unbounded",          # 模型无界
    GRB.TIME_LIMIT: "Time limit reached",         # 达到时间限制
    GRB.INTERRUPTED: "Interrupted by user",       # 被用户中断
    GRB.SUBOPTIMAL: "Suboptimal solution found",  # 次优解
}

def optimize_territory_assignment(df, distance_threshold, n_territories, initial_solution, params=None):
    # 设置默认参数
    default_params = {
        'fte_weight': 8,
        'prod_weight': 5,
        'poten_weight': 1,
        'growth_weight': 1,
        'terri_weight': 50,
        'target_fte': 1,
        'growth_rate_threshold': 1.1821991827107088,
        'hco_count_avg': 43.869565217391305,
        'avg_productivity': 9.558150212899818,
        'avg_potential': 334.3831946335722,
        'time_limit_step2': 5
    }
    
    # 如果提供了参数，则更新默认值
    if params:
        default_params.update(params)
    
    # 从参数中获取值
    fte_weight = default_params['fte_weight']
    prod_weight = default_params['prod_weight']
    poten_weight = default_params['poten_weight']
    growth_weight = default_params['growth_weight']
    terri_weight = default_params['terri_weight']
    target_fte = default_params['target_fte']
    growth_rate_threshold = default_params['growth_rate_threshold']
    hco_count_avg = default_params['hco_count_avg']
    avg_productivity = default_params['avg_productivity']
    avg_potential = default_params['avg_potential']
#     avg_potential = avg_potential/1.06
    time_limit_step2 = default_params['time_limit_step2']

    print(fte_weight)
    # 其他固定参数
    fte_lower_bound = 0.85  
    fte_upper_bound = 1.15
    productivity_lower_bound = 0.5
    productivity_upper_bound = 1.5
    potential_lower_bound = 0.5
    potential_upper_bound = 3
    lambda_fte = 4.0
    lambda_dist = 1.0

    group_fte = df.groupby('group')['fte'].sum().values
    group_productivity = df.groupby('group')['productivity'].sum().values
    group_potential = df.groupby('group')['potential'].sum().values
#     df['log_potential'] = np.log1p(df['potential'])
#     group_potential = df.groupby('group')['log_potential'].sum().values
    group_productivity_ly = df.groupby('group')['productivity_ly'].sum().values
    group_coords = df.groupby('group')[['latitude', 'longitude']].mean().values
    group_distances = squareform(pdist(group_coords))
    
    # 创建group到index的映射
    groups = df['group'].unique()
    group_to_idx = {group: idx for idx, group in enumerate(groups)}
    n_groups = len(groups)
    cities = df['city'].unique()  # 提前获取所有唯一城市

    # 2. Create Gurobi Model
    model = gp.Model(f"Territory_Assignment_Threshold_{distance_threshold}")
    
    # 3. Define Decision Variables
    x = {}
    for i in range(n_territories):
        for j in range(n_groups):
            x[i,j] = model.addVar(vtype=GRB.BINARY, name=f"assign_{i}_{j}")
    
    # Satisfaction Variables
    fte_satisfied = {}
    productivity_satisfied = {}
    potential_satisfied = {}
    growth_satisfied = {}  # 新增：增长率满足条件的变量

    for i in range(n_territories):
        fte_satisfied[i] = model.addVar(vtype=GRB.BINARY, name=f"fte_sat_{i}")
        productivity_satisfied[i] = model.addVar(vtype=GRB.BINARY, name=f"prod_sat_{i}")
        potential_satisfied[i] = model.addVar(vtype=GRB.BINARY, name=f"pot_sat_{i}")
        growth_satisfied[i] = model.addVar(vtype=GRB.BINARY, name=f"growth_sat_{i}")  # 新增
        
    territory_city_count = {}  # 记录每个 territory 覆盖的城市数
    for i in range(n_territories):
        territory_city_count[i] = model.addVar(name=f"territory_city_count_{i}", vtype=GRB.INTEGER, lb=0)


    # Violation Ratio Variables
    fte_violation_ratio = model.addVar(name="fte_violation_ratio", lb=0, ub=1)
    productivity_violation_ratio = model.addVar(name="productivity_violation_ratio", lb=0, ub=1)
    potential_violation_ratio = model.addVar(name="potential_violation_ratio", lb=0, ub=1)
    growth_violation_ratio = model.addVar(name="growth_violation_ratio", lb=0, ub=1)  # 新增
    territory_violation_ratio = model.addVar(name="territory_violation_ratio", lb=0, ub=1)  # 新增：保持一致的命名
    
    # 2.4 城市选择变量
    city_selected = {}  # 用于判断territory是否选择了某个城市
    for i in range(n_territories):
        for c in cities:
            city_selected[i,c] = model.addVar(vtype=GRB.BINARY, name=f"city_selected_{i}_{c}")

    # 新增：与初始解的差异变量
    solution_diff = model.addVar(name="solution_difference_ratio", lb=0, ub=1)
    assignment_diff = {}
    for i in range(n_territories):
        for j in range(n_groups):
            assignment_diff[i,j] = model.addVar(vtype=GRB.BINARY, name=f"diff_{i}_{j}")


    model.update()
    

    # 4. Set Objective Function
    alpha = 50  # 调整这个权重来平衡与初始解的差异和其他目标
    model.setObjective(
        fte_weight * fte_violation_ratio + 
        prod_weight * productivity_violation_ratio + 
        poten_weight * potential_violation_ratio+
        growth_weight * growth_violation_ratio+
        terri_weight * territory_violation_ratio +
        alpha * solution_diff,  # 新增项,
        GRB.MINIMIZE)


    # 5. Add Constraints
    
    # Distance Constraint
    for i in range(n_territories):
        for j in range(n_groups):
            for k in range(j + 1, n_groups):
                if group_distances[j, k] > distance_threshold:
                    model.addConstr(x[i,j] + x[i,k] <= 1)

    print('here')

#     # Distance Constraint - 优化版本
#     for i in range(n_territories):
#         # 预先计算不兼容的组合
#         incompatible_pairs = [(j, k) for j in range(n_groups) 
#                              for k in range(j + 1, n_groups) 
#                              if group_distances[j, k] > distance_threshold]

#         # 只为不兼容的组合添加约束
#         for j, k in incompatible_pairs:
#             model.addConstr(x[i,j] + x[i,k] <= 1)
            
    

    # 修��� POV 数量约束部分
    # 将软约束改为硬约束
    for i in range(n_territories):
        total_pov_count = quicksum(x[i,j] for j in range(n_groups))
        model.addConstr(total_pov_count <= 3.0 * hco_count_avg)

    
    
    # Each group must be completely assigned
    for j in range(n_groups):
        model.addConstr(quicksum(x[i,j] for i in range(n_territories)) == 1)
        
    # 确保每个 territory 至少被分配到一个 group
    for i in range(n_territories):
        model.addConstr(quicksum(x[i, j] for j in range(n_groups)) >= 1)
    
    # Big M for constraint relaxation
    M = 1e10
    
    # FTE Constraints
    for i in range(n_territories):
        total_fte = quicksum(x[i,j] * group_fte[j] for j in range(n_groups))
        model.addConstr(total_fte >= fte_lower_bound * target_fte - M * (1 - fte_satisfied[i]))
        model.addConstr(total_fte <= fte_upper_bound * target_fte + M * (1 - fte_satisfied[i]))
    
    # Productivity Constraints
    for i in range(n_territories):
        total_productivity = quicksum(x[i,j] * group_productivity[j] for j in range(n_groups))
        model.addConstr(total_productivity >= productivity_lower_bound * avg_productivity - M * (1 - productivity_satisfied[i]))
        model.addConstr(total_productivity <= productivity_upper_bound * avg_productivity + M * (1 - productivity_satisfied[i]))
    
    # Potential Constraints
    for i in range(n_territories):
        total_potential = quicksum(x[i,j] * group_potential[j] for j in range(n_groups))
        model.addConstr(total_potential >= potential_lower_bound * avg_potential - M * (1 - potential_satisfied[i]))
        model.addConstr(total_potential <= potential_upper_bound * avg_potential + M * (1 - potential_satisfied[i]))
     # 增长率约束
    for i in range(n_territories):
        total_productivity = quicksum(x[i,j] * group_productivity[j] for j in range(n_groups))
        total_productivity_ly = quicksum(x[i,j] * group_productivity_ly[j] for j in range(n_groups))

        # 使用大M方法确保当growth_satisfied[i]=1时，增长率>=growth_rate_threhold
        model.addConstr(total_productivity >= growth_rate_threshold * total_productivity_ly - M * (1 - growth_satisfied[i]))
       # 为避免除零错误，使用乘法形式而不是除法
    
    # Violation Ratio Constraints
    model.addConstr(fte_violation_ratio == 1 - quicksum(fte_satisfied[i] for i in range(n_territories)) / n_territories)
    model.addConstr(productivity_violation_ratio == 1 - quicksum(productivity_satisfied[i] for i in range(n_territories)) / n_territories)
    model.addConstr(potential_violation_ratio == 1 - quicksum(potential_satisfied[i] for i in range(n_territories)) / n_territories)
    model.addConstr(growth_violation_ratio == 1 - quicksum(growth_satisfied[i] for i in range(n_territories)) / n_territories)


    # 单城市territory约束
    # 连接city_selected和x变量：当一个territory包含某个group时，对应的city必须被选中
    for i in range(n_territories):
        for j, group in enumerate(groups):  # 使用enumerate避免使用group_to_idx
            city = df.loc[df['group'] == group, 'city'].iloc[0]  # 获取该group对应的city
            model.addConstr(city_selected[i,city] >= x[i,j])

    # 计算每个 territory 包含的城市数量
    for i in range(n_territories):
        model.addConstr(
            territory_city_count[i] == quicksum(city_selected[i,c] for c in cities),
            name=f"city_count_constr_{i}"
        )

    # 修改违反比例的计算方式
    # 计算相对违反程度：(实际城市数 - 1) / (最大可能城市数 - 1)
    model.addConstr(
        territory_violation_ratio == 
        quicksum(territory_city_count[i] - 1 for i in range(n_territories)) / 
    #     (n_territories * (len(cities) - 1)) ,
        (n_territories * 5),
        name="territory_violation_ratio_constr"
    )


    # 新增：计算与初始解的差异
    if initial_solution is not None:
        for i in range(n_territories):
            for j in range(n_groups):
                # 如果分配与初始解不同，assignment_diff为1
                model.addConstr(assignment_diff[i,j] >= x[i,j] - initial_solution.iloc[i,j])
                model.addConstr(assignment_diff[i,j] >= initial_solution.iloc[i,j] - x[i,j])

        # 计算总体差异率
        total_assignments = n_territories * n_groups
        model.addConstr(solution_diff == quicksum(assignment_diff[i,j] 
                                                for i in range(n_territories) 
                                                for j in range(n_groups)) / total_assignments)


    # 6. 设置初始解作为warm start
    if initial_solution is not None:
        for i in range(n_territories):
            for j, group in enumerate(initial_solution.columns):
                if initial_solution.iloc[i, j] == 1:
                    x[i,j].start = 1
                else:
                    x[i,j].start = 0

    # Solve Parameters
    # 求解器参数设置
    model.setParam('TimeLimit', time_limit_step2)
    model.setParam('MIPGap', 0.01)
    model.setParam('Threads', 7)
    model.setParam('MIPFocus', 1)           # 将求解重点放在寻找可行解上(值为1)而不是提升下界
     # 设置启发式参数
    model.setParam('Heuristics', 1)  # 增加启发式搜索强度
    model.setParam('ImproveStartGap', 0.5)  # 当gap大于50%时就开始改进
    model.setParam('RINS', 5)                 # 每5个节点执行一次RINS(默认是25)

    # 设置数值稳定性参数
    model.setParam('ScaleFlag', 2)     # 激进的自动缩放

    model.setParam('ImproveStartTime', 0) # 立即开始改进初始解
    model.setParam('NumericFocus', 3)  # 最高数值稳定性

    model.setParam('OutputFlag', 1)  # Suppress output for cleaner loop
    
    # 8. Solve Model
    model.optimize()

    
    # 获取模型状态的含义
    model_status_description = status_meaning.get(model.status, "Unknown status")
        
    # 9. Get Results
    def safe_get_value(var):
        try:
            return var.x if var.x is not None else None
        except AttributeError:
            return None
    
    # 创建分配矩阵
    assignment_matrix = {}

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # 捕获分配矩阵
        for i in range(n_territories):
            for j in range(n_groups):
                try:
                    if x[i,j].x > 0.5:  # 二进制变量，接近1表示分配
                        assignment_matrix[(i,j)] = 1
                    else:
                        assignment_matrix[(i,j)] = 0
                except AttributeError:
                    assignment_matrix[(i,j)] = 0

        result = {
            'distance_threshold': distance_threshold,
            'objective_value': model.objVal if model.objVal is not None else None,
            'fte_violation_ratio': safe_get_value(fte_violation_ratio),
            'productivity_violation_ratio': safe_get_value(productivity_violation_ratio),
            'potential_violation_ratio': safe_get_value(potential_violation_ratio),
            'growth_violation_ratio': safe_get_value(growth_violation_ratio),
            'territory_violation_ratio': safe_get_value(territory_violation_ratio),
            'solution_diff':safe_get_value(solution_diff),
            'model_status': model_status_description,
            'best_bound': model.ObjBound if hasattr(model, 'ObjBound') else None
        }

        return result, assignment_matrix
    else:
        result = {
            'distance_threshold': distance_threshold,
            'objective_value': None,
            'fte_violation_ratio': None,
            'productivity_violation_ratio': None,
            'potential_violation_ratio': None,
            'growth_violation_ratio': None,
            'territory_violation_ratio': None,
            'solution_diff': None,
            'model_status': model_status_description,
            'best_bound': model.ObjBound if hasattr(model, 'ObjBound') else None
        }
        return result, None

def run_threshold_exploration(df, global_n_territories, output_dir, start=0.8, end=1.6, num_values=3, params=None):
    """
    添加 output_dir 参数来指定输出目录
    """
    results = []
    
    for n_territories in np.arange(global_n_territories-1, global_n_territories+2, 1):
        n_territories = max(n_territories, 1)
        print(f"\033[1mExploring for n_territories = {n_territories}\033[0m")
        
        print('\n生成kmeans初始解')
        initial_solution = generate_initial_solution(df, n_territories, weight=10, random_state=42)
        print('\n完成kmeans初始解')

        for threshold in np.linspace(start, end, num_values):
            print(f"\033[1mCurrent Distance Threshold: {threshold:.2f}\033[0m")
            
            result, assignment_matrix = optimize_territory_assignment(
                df, 
                threshold, 
                n_territories,
                initial_solution,
                params=params
            )

            result['n_territories'] = n_territories
            results.append(result)
            
            try:
                if result.get('model_status') in ['Optimal solution found', 'Time limit reached']:
                    assignments = {}
                    for (i, j), value in assignment_matrix.items():
                        if value > 0.5:
                            group_name = list(df['group'].unique())[j]
                            db_cluster = df.loc[df['group'] == group_name, 'db_cluster'].iloc[0]
                            assignments[group_name] = f"{db_cluster}_T{i}"

                    assignment_df = df.copy()
                    if 'MR Pos' in assignment_df.columns:
                        assignment_df.rename(columns={'MR Pos': 'MR Pos_old'}, inplace=True)

                    assignment_df['MR Pos'] = assignment_df['group'].map(assignments).fillna('Unassigned')
                    
                    # 直接使用传入的 output_dir
                    if output_dir:
                        db_cluster_prefix = "_".join(df['db_cluster'].unique())
                        filename = os.path.join(
                            output_dir, 
                            f"second_round_{db_cluster_prefix}_assignment_n{n_territories}_t{threshold:.2f}.xlsx"
                        )
                        assignment_df.to_excel(filename, index=False)
                        print(f"Saved assignment results to {filename}")

            except Exception as e:
                print(f"Could not save assignment results: {e}")
    
    return pd.DataFrame(results)