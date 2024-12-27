import os
import uuid
import time
import shutil
import pathlib
from datetime import datetime

from flask import Flask, render_template, request, send_file, jsonify, session, Response,send_from_directory
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

from optimization_logic_step1 import optimize_region_assignment, format_optimization_results
from optimization_logic_step2 import run_threshold_exploration
from optimization_logic_step3 import optimize_and_combine_results

from util_function import calculate_inter_city_distances, calculate_city_hospital_distances
from util_function import compute_convex_hulls, calculate_and_sort_overlaps
from util_function import analyze_hco_groups,fitness,individual_summary,fitness_detail

from collections import defaultdict  # 添加这个导入

# 模拟进度更新
def generate_optimization_progress():
    progress = 0
    while progress < 100:
        time.sleep(0.5)  # 模拟计算时间
        progress += 10
        yield f"data: {progress}\n\n"

# 添加新的辅助函数
def create_result_directory(filename, session_id=None):
    # 移除文件扩展名
    base_filename = os.path.splitext(filename)[0]
    # 创建日期字符串
    date_str = datetime.now().strftime('%Y%m%d_%H%M')
    # 如果没有提供session_id，生成一个
    session_id = session_id or str(uuid.uuid4())
    # 创建目录名
    dir_name = f"{base_filename}_{date_str}_{session_id}"
    # 创建完整路径
    result_dir = os.path.join('results', dir_name)
    # 确保目录存在
    pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
    return result_dir

# 清理过期目录
def cleanup_old_result_directories(max_age_hours=24):
    now = datetime.now()
    results_path = 'results'
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    for dir_name in os.listdir(results_path):
        dir_path = os.path.join(results_path, dir_name)
        if os.path.isdir(dir_path):
            try:
                # 提取目录创建时间 
                parts = dir_name.split('_')
                if len(parts) >= 3:
                    date_str = f"{parts[-3]}_{parts[-2]}"
                    dir_time = datetime.strptime(date_str, '%Y%m%d_%H%M')
                    
                    # 删除超过指定时间的目录
                    if (now - dir_time).total_seconds() > max_age_hours * 3600:
                        shutil.rmtree(dir_path)
            except Exception as e:
                print(f"Error cleaning up directory {dir_name}: {e}")

STATIC_FOLDER = os.path.join(os.getcwd(), 'templates')
app = Flask(__name__, static_folder=STATIC_FOLDER)
# app = Flask(__name__)

# 配置会话和秘钥
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except FileNotFoundError:
        return f"File {filename} not found", 404

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件被上传'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 使用会话ID创建结果目录
        result_dir = create_result_directory(file.filename)
        
        df = pd.read_excel(file)
        required_columns = ['city', 'fte', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': f'缺少必要的列: {", ".join(missing_columns)}'}), 400
        
        distance_matrix = calculate_inter_city_distances(df)
        
        # 保存文件到新的目录
        input_file_path = os.path.join(result_dir, 'input_data.xlsx')
        distance_matrix_path = os.path.join(result_dir, 'distance_matrix.xlsx')
        
        df.to_excel(input_file_path, index=False)
        distance_matrix.to_excel(distance_matrix_path, index=True)
        
        # 将目录路径存储在会话中
        session['result_dir'] = result_dir
        
        return jsonify({'message': '文件上传成功', 'rows': len(df)}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        params = {
            'fte_lower_bound': float(request.form.get('fte_lower', 0.85)),
            'lambda_fte': float(request.form.get('lambda_fte', 4.0)),
            'lambda_dist': float(request.form.get('lambda_dist', 1.0)),
            'time_limit': int(request.form.get('time_limit', 30))
        }
        
        result_dir = session.get('result_dir')
        if not result_dir:
            return jsonify({'error': '未找到上传的文件信息'}), 400
            
        input_file_path = os.path.join(result_dir, 'input_data.xlsx')
        df_orig = pd.read_excel(input_file_path)
        distance_matrix = calculate_inter_city_distances(df_orig, method='min')

        result_df_orig = df_orig.groupby('city').agg({
            'fte': 'sum',
            'potential': 'sum',
            'productivity': 'sum',
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        results = optimize_region_assignment(
            result_df_orig=result_df_orig,
            distance_matrix=distance_matrix,
            **params
        )
        
        result_df = format_optimization_results(results,df_orig)
        
        # 添加凸包分析
        mr_hulls = compute_convex_hulls(result_df, col='MR Pos')
        mr_overlaps = calculate_and_sort_overlaps(mr_hulls)
        city_hulls = compute_convex_hulls(result_df, col='city')
        city_overlaps = calculate_and_sort_overlaps(city_hulls)
        
        # 计算总重叠面积
        mr_total_overlap = mr_overlaps['OverlapArea'].sum() if not mr_overlaps.empty else 0
        city_total_overlap = city_overlaps['OverlapArea'].sum() if not city_overlaps.empty else 0
        
        # 计算比值
        total_overlap_ratio = mr_total_overlap / city_total_overlap if city_total_overlap > 0 else 0
        # 判断并设置warning_msg
        if total_overlap_ratio > 5:
            warning_msg = f"当前分区的比例为{total_overlap_ratio}，出现隔城分配情况，请考虑提高距离权重或者降低FTE下限，重新优化"
        else:
            warning_msg = f"当前分区的比例为{total_overlap_ratio}，可以进行第二步优化"
        print(warning_msg)
        # 保存结果到新的文件
        result_filename = 'optimization_results_dbcluster.xlsx'
        result_path = os.path.join(result_dir, result_filename)
        result_df.to_excel(result_path, index=False)
        
        # 保存诊断信息到session中
        session['diagnostic_info'] = {
            'warning_msg': warning_msg,
            'mr_overlaps': mr_overlaps.to_dict('records') if not mr_overlaps.empty else []
        }
        
        return send_file(
            result_path,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            attachment_filename=result_filename
        )
    
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/optimize_second_round', methods=['POST'])
def optimize_second_round():
    try:
        # 获取参数
        params = {
            'distance_threshold': float(request.form.get('distance_threshold', 0.8)),
            'fte_weight': float(request.form.get('fte_weight', 8)),
            'prod_weight': float(request.form.get('prod_weight', 5)),
            'poten_weight': float(request.form.get('poten_weight', 1)),
            'growth_weight': float(request.form.get('growth_weight', 1)),
            'terri_weight': float(request.form.get('terri_weight', 50)),
            'target_fte': float(request.form.get('target_fte', 1)),
            'growth_rate_threshold': float(request.form.get('growth_rate_threshold', 1.1821991827107088)),
            'hco_count_avg': float(request.form.get('hco_count_avg', 43.869565217391305)),
            'avg_productivity': float(request.form.get('avg_productivity', 9.558150212899818)),
            'avg_potential': float(request.form.get('avg_potential', 334.383194633)),
            'time_limit_step2': int(request.form.get('time_limit_step2', 5))
        }
        # 将参数存储在session中以供第三步使用
        session['optimization_params'] = params

        result_dir = session.get('result_dir')
        if not result_dir:
            return jsonify({'error': '未找到上传的文件信息'}), 400
            
        # 读取第一轮优化结果
        first_round_results = os.path.join(result_dir, 'optimization_results_dbcluster.xlsx')
        if not os.path.exists(first_round_results):
            return jsonify({'error': '未找到第一轮优化结果'}), 400
            
        df_orig = pd.read_excel(first_round_results)
        
        
        # 计算距离阈值范围
        city_hospital_distances = calculate_city_hospital_distances(df_orig, 'db_cluster')
        result_cities = df_orig['db_cluster'].unique()
        result_cities_max_distances = {city: city_hospital_distances[city]['max_distance'] for city in result_cities}
        result_cities_max_distances_list = [city_hospital_distances[city]['max_distance'] for city in result_cities]
        start = min(result_cities_max_distances_list) + 0.1
        end = max(result_cities_max_distances_list) + 0.1
        
        result_dfs = []
        
        for cluster in df_orig['db_cluster'].unique():
            df = df_orig[df_orig['db_cluster'] == cluster]
            global_n_territories = int(np.ceil(df['fte'].sum()))
            
            # 更新全局变量
            globals().update(params)
            

            result_df = run_threshold_exploration(
                df, 
                global_n_territories, 
                output_dir=result_dir,
                start=start, 
                end=end,
                num_values=3,
                params=params
            )
            result_df['db_cluster'] = cluster
            result_dfs.append(result_df)
        
        # 合并结果
        result_df_all = pd.concat(result_dfs, ignore_index=True)
        
        # 保存结果
        output_file = os.path.join(result_dir, 'second_round_exploration.xlsx')
        result_df_all.to_excel(output_file, index=False)
        
        return send_file(
            output_file,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            attachment_filename='second_round_exploration.xlsx'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/optimize_third_round', methods=['POST'])
def optimize_third_round():
    try:
        # 获取建议的区域数量参数
        n_territories_suggested = int(request.form.get('n_territories_suggested', 25))
        # 保存到 session 中供后续使用
        session['n_territories_suggested'] = n_territories_suggested
        
        result_dir = session.get('result_dir')
        if not result_dir:
            return jsonify({'error': '未找到上传的文件信息'}), 400
            
        # 读取第二轮优化结果
        second_round_results = os.path.join(result_dir, 'second_round_exploration.xlsx')
        if not os.path.exists(second_round_results):
            return jsonify({'error': '未找到第二轮优化结果'}), 400
            
        result_df_all = pd.read_excel(second_round_results)
        
        # 运行第三轮优化
        final_result = optimize_and_combine_results(
            result_df_all=result_df_all,
            n_territories_suggested=n_territories_suggested,
            result_dir=result_dir,
            file_name='final_optimization'
        )
        
        if final_result is None:
            return jsonify({'error': '优化过程失败'}), 500
            
        # 构造输出文件路径
        output_file = os.path.join(
            result_dir, 
            f'final_optimization_n{n_territories_suggested}_gurobi_region_divided_for_plot.xlsx'
        )
        
        return send_file(
            output_file,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            attachment_filename=f'final_optimization_n{n_territories_suggested}.xlsx'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def optimization_status():
    return jsonify({'status': 'ready'})  # 示例状态，需根据实际逻辑动态调整

@app.route('/preview', methods=['GET'])
def preview_data():
    try:
        result_dir = session.get('result_dir')
        if not result_dir:
            return jsonify({'error': '未找到上传的文件信息'}), 400
            
        input_file_path = os.path.join(result_dir, 'input_data.xlsx')
        df = pd.read_excel(input_file_path)
        return jsonify({
            'columns': df.columns.tolist(),
            'data': df.head(10).to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/progress', methods=['GET'])
def optimization_progress():
    return Response(generate_optimization_progress(), content_type='text/event-stream')

@app.route('/list_files', methods=['GET'])
def list_files():
    try:
        result_dir = session.get('result_dir')
        if not result_dir:
            return jsonify({'error': '未找到上传的文件信息'}), 400
            
        files = []
        for file in os.listdir(result_dir):
            file_path = os.path.join(result_dir, file)
            if os.path.isfile(file_path):
                files.append({
                    'name': file,
                    'size': os.path.getsize(file_path),
                    'modified': os.path.getmtime(file_path)
                })
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        result_dir = session.get('result_dir')
        if not result_dir:
            return jsonify({'error': '未找到上传的文件信息'}), 400
            
        file_path = os.path.join(result_dir, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': '文件不存在'}), 404
            
        return send_file(
            file_path,
            as_attachment=True,
            attachment_filename=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 在应用启动时清理过期目录
@app.before_first_request
def cleanup_before_first_request():
    cleanup_old_result_directories()

# 定期清理任务 (如果使用后台任务调度器)
def schedule_cleanup():
    while True:
        try:
            cleanup_old_result_directories()
        except Exception as e:
            print(f"Scheduled cleanup error: {e}")
        time.sleep(24 * 3600)  # 每24小时运行一次

# 添加新的路由来获取诊断信息
@app.route('/get_diagnostic', methods=['GET'])
def get_diagnostic():
    # 获取所有诊断信息
    diagnostic_info = {
        'diagnostic_info': session.get('diagnostic_info', {}),  # 第一步的诊断信息
        'final_diagnostic_info': session.get('final_diagnostic_info', {})  # 最终诊断信息
    }
    return jsonify(diagnostic_info)

@app.route('/final_diagnosis', methods=['POST'])
def final_diagnosis():
    try:
        result_dir = session.get('result_dir')
        if not result_dir:
            return jsonify({'error': '未找到上传的文件信息'}), 400
        
        # 从session中获取第二步存储的参数
        optimization_params = session.get('optimization_params')
        if not optimization_params:
            return jsonify({'error': '未找到优化参数信息'}), 400
        # 读取第三轮优化结果
        final_result_file = os.path.join(
            result_dir, 
            'final_optimization_n{}_gurobi_region_divided_for_plot.xlsx'.format(
                session.get('n_territories_suggested', 25)
            )
        )
        
        if not os.path.exists(final_result_file):
            return jsonify({'error': '未找到第三轮优化结果'}), 400
            
        
        print(final_result_file)
        df = pd.read_excel(final_result_file)
        # print(df)
   
        
        # 进行分析
        (group2hco,
         individual, 
         group_productivity, 
         group_productivity_ly, 
         group_fte, 
         group_potential, 
         group_TH, 
         group_city_matrix) = analyze_hco_groups(df)
        # print(individual)
        
        # # 设置其他必要的全局变量
        # avg_productivity_territory = np.mean(group_productivity)
        # avg_productivity_growth = 1.1821991827107088  # 这个值可能需要根据实际情况调整
        # avg_potential_territory = np.mean(group_potential)
        # avg_TH = np.mean(group_TH)
        # workload_per_fte = 1.0  # 这个值可能需要根据实际情况调整

        # 从第二步的参数中获取值 从session中获取存储的参数
        avg_productivity_territory = optimization_params['avg_productivity']
        avg_productivity_growth = optimization_params['growth_rate_threshold']
        avg_potential_territory = optimization_params['avg_potential']
        avg_TH = optimization_params['hco_count_avg']
        workload_per_fte = 12 * 16.75 ####
        # 计算详细适应度
        fitness_detail_str = fitness_detail(
            individual,
            group_fte=group_fte,
            group_productivity=group_productivity,
            group_productivity_ly=group_productivity_ly,
            group_potential=group_potential,
            group_TH=group_TH,
            group_city_matrix=group_city_matrix,
            group2hco=group2hco,
            avg_productivity_territory=avg_productivity_territory,
            avg_productivity_growth=avg_productivity_growth,
            avg_potential_territory=avg_potential_territory,
            avg_TH=avg_TH,
            df=df,
            radius_base=0.7344073907957636,
            travel_base=4.017090516769585
        )
        print(fitness_detail_str)

        # 定义固定的列顺序
        column_order = [
            'score', 'workload_score', 'workload_penalty', 'workload_score_over80',
            'productivity_score', 'productivity_penalty_doublelow', 'productivity_penalty_lowpr',
            'potential_score', 'distance_score_radius', 'distance_score_travel',
            'distance_penalty_radius', 'distance_penalty_travel', 'distance_penalty_city'
        ]

        # 定义英文列名到中文的映射
        column_mapping = {
            'score': '总分',
            'workload_score': '工作量得分',
            'workload_penalty': '工作量惩罚',
            'workload_score_over80': '工作量达标率',
            'productivity_score': '生产力得分',
            'productivity_penalty_doublelow': '双低惩罚',
            'productivity_penalty_lowpr': '低产能惩罚',
            'potential_score': '潜力得分',
            'distance_score_radius': '半径距离得分',
            'distance_score_travel': '行程距离得分',
            'distance_penalty_radius': '半径惩罚',
            'distance_penalty_travel': '行程惩罚',
            'distance_penalty_city': '城市惩罚'
        }

        # 解析 fitness_detail_str 并创建有序字典
        metrics = {}
        for item in fitness_detail_str.split('; '):
            if ':' in item:
                key, value = item.split(':', 1)
                metrics[key.strip()] = float(value.strip())  # 转换为浮点数
        
        # 获取历史记录
        history_results = session.get('fitness_history', [])
        
        # 添加当前结果（保持列顺序）
        current_result = {column_mapping[col]: metrics.get(col, 0.0) for col in column_order}
        history_results.append(current_result)
        
        # 只保留最近5次结果
        if len(history_results) > 5:
            history_results = history_results[-5:]
        
        # 更新 session 中的历史记录
        session['fitness_history'] = history_results
        
        # 创建 DataFrame，添加次数列
        fitness_detail_str_df = pd.DataFrame(history_results)
        fitness_detail_str_df.insert(0, '次数', range(1, len(history_results) + 1))
        
        # 确保列的顺序（使用中文列名）
        column_order_chinese = ['次数'] + [column_mapping[col] for col in column_order]
        fitness_detail_str_df = fitness_detail_str_df[column_order_chinese]
        
        # 保存历史记录到结果目录
        history_file = os.path.join(result_dir, 'fitness_history.xlsx')
        fitness_detail_str_df.to_excel(history_file, index=False)
        print(fitness_detail_str_df)
        
        # 生成摘要
        individual_summary_df = individual_summary(
            individual=individual,
            group_fte=group_fte,
            group_productivity=group_productivity,
            group_productivity_ly=group_productivity_ly,
            group_potential=group_potential,
            group_TH=group_TH,
            group_city_matrix=group_city_matrix,
            group2hco=group2hco,
            avg_productivity_territory=avg_productivity_territory,
            avg_TH=avg_TH,
            avg_potential_territory = avg_potential_territory,
            df=df,
            workload_per_fte=workload_per_fte
        )        
        # 保存诊断结果
        output_file = os.path.join(result_dir, 'final_diagnosis.xlsx')
        individual_summary_df.to_excel(output_file, index=False)
        
        # 保存诊断信息到session（使用字典列表格式）
        session['final_diagnostic_info'] = {
            'warning': fitness_detail_str,
            'history': fitness_detail_str_df.to_dict('records')  # 包含次数列的完整记录
        }
        
        return send_file(
            output_file,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            attachment_filename='final_diagnosis.xlsx'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 可选：如果使用多线程后台任务
    # import threading
    # cleanup_thread = threading.Thread(target=schedule_cleanup, daemon=True)
    # cleanup_thread.start()
    print(f"Static folder path: {app.static_folder}")
    print(f"Available files: {os.listdir(app.static_folder)}")
    app.run(host='localhost', port=8889, debug=True)