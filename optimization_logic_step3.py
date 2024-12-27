import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans
from util_function import generate_initial_solution
from gurobipy import *


def optimize_and_combine_results(result_df_all, n_territories_suggested, result_dir, file_name):
    """
    优化区域分配并合并结果
    
    参数:
    result_df_all: DataFrame, 包含所有优化结果的数据框
    n_territories_suggested: int, 建议的区域数量
    result_dir: str, 结果文件目录路径
    file_name: str, 原始文件名
    
    返回:
    DataFrame: 合并后的最终结果
    """
    try:
        # 删除NA行
        result_df_all = result_df_all.dropna()
        db_clusters = result_df_all['db_cluster'].unique()
        
        # 创建 Gurobi 模型
        model = gp.Model("db_cluster_optimization")
        
        # 定义变量
        x = model.addVars(result_df_all.index, vtype=GRB.BINARY, name="x")
        
        # 设置目标函数
        model.setObjective(
            .001 * gp.quicksum(result_df_all.loc[i, 'objective_value'] * x[i] for i in result_df_all.index) +
            gp.quicksum(result_df_all.loc[i, 'fte_violation_ratio'] * x[i] for i in result_df_all.index)+
            .1 * gp.quicksum(result_df_all.loc[i, 'territory_violation_ratio'] * x[i] for i in result_df_all.index), 
            GRB.MINIMIZE
        )
        
        # 添加约束条件
        # 约束1: 每个 db_cluster 只能选择一次
        for cluster in db_clusters:
            model.addConstr(
                gp.quicksum(x[i] for i in result_df_all.index if result_df_all.loc[i, 'db_cluster'] == cluster) == 1,
                name=f"unique_cluster_{cluster}"
            )
        
        # 约束2: n_territories 总和等于建议值
        model.addConstr(
            gp.quicksum(result_df_all.loc[i, 'n_territories'] * x[i] for i in result_df_all.index) == n_territories_suggested,
            name="territories_sum"
        )
        
        # 执行优化
        model.optimize()
        
        # 处理优化结果
        if model.status == GRB.OPTIMAL:
            selected_rows = [i for i in result_df_all.index if x[i].x > 0.5]
            print(f"优化后的结果行数: {len(selected_rows)}")
            print(f"选择的行: {selected_rows}")
            print(f"最小的目标函数值: {model.objVal}")
        else:
            print("未找到最优解")
            return None
            
        selected_df = result_df_all.loc[selected_rows]
        
        # 合并选中的结果文件
        result = pd.DataFrame()
        
        for index, row in selected_df.iterrows():
            file_path = os.path.join(
                result_dir, 
                f"second_round_{row['db_cluster']}_assignment_n{int(row['n_territories'])}_t{row['distance_threshold']:.2f}.xlsx"
            )
            
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在: {file_path}")
                continue
            
            city_assignment = pd.read_excel(file_path)
            result = pd.concat([result, city_assignment], ignore_index=True)
        
        # 排序并重置索引
        result_province = result.sort_values(by='group').reset_index(drop=True)
        
        # 保存最终结果
        result_file_name = os.path.join(
            result_dir, 
            f'{file_name.replace(".xlsx", "")}_n{n_territories_suggested}_gurobi_region_divided_for_plot.xlsx'
        )
        result_province.to_excel(result_file_name, index=False)
        
        print(f"结果已保存到: {result_file_name}")
        return result_province
        
    except Exception as e:
        print(f"优化过程中发生错误: {str(e)}")
        return None