import os
import json

import pandas as pd


def modify_libritts_list_files(directory_path: str, save_directory: str = None) -> None:
    """
    遍历指定目录下的所有.list文件，将每行的绝对文件路径的非文件名部分替换为指定的目录路径。
    
    Args:
        directory_path: 要替换成的目录路径
        save_directory: 可选，保存修改后文件的目录路径，默认为None（保存在原位置）
    """
    # 获取当前目录下所有.list文件
    list_names = [f for f in os.listdir(directory_path) if f.endswith('.list')]
    list_files = [os.path.join(directory_path, f) for f in list_names]
    
    for i, list_file in enumerate(list_files):
        try:
            modified_lines = []
            
            # 读取文件内容
            with open(list_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line:
                    # 获取文件名
                    file_name = os.path.basename(line)
                    # 构建新路径
                    new_path = os.path.join(directory_path, file_name)
                    modified_lines.append(new_path)
            
            # 确定保存路径
            if save_directory:
                # 确保保存目录存在
                os.makedirs(save_directory, exist_ok=True)
                output_file = os.path.join(save_directory, list_names[i])
            else:
                output_file = os.path.join(directory_path, list_names[i])
            
            # 写回文件
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in modified_lines:
                    f.write(line + '\n')
        
            print(f"已修改文件: {list_file} -> 保存于: {output_file}")
        except Exception as e:
            print(f"修改文件: {list_file} 失败，错误信息: {e}")


def modify_libritts_json_files(directory_path: str, save_directory: str = None) -> None:
    """
    遍历指定目录下的所有.json文件，将每个JSON中字典value中的文件路径的非文件名部分替换为指定的目录路径。
    
    Args:
        directory_path: 要替换成的目录路径
        save_directory: 可选，保存修改后文件的目录路径，默认为None（保存在原位置）
    """
    # 获取当前目录下所有.json文件
    json_names = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    json_files = [os.path.join(directory_path, f) for f in json_names]
    
    for i, json_file in enumerate(json_files):
        try:
            # 读取JSON文件内容
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
            # 修改字典中的路径
            for key, value in data.items():
                if isinstance(value, str) and os.path.basename(value) != value:  # 确认value是路径
                    # 获取文件名
                    file_name = os.path.basename(value)
                    # 构建新路径
                    new_path = os.path.join(directory_path, file_name)
                    data[key] = new_path
            
            # 确定保存路径
            if save_directory:
                # 确保保存目录存在
                os.makedirs(save_directory, exist_ok=True)
                output_file = os.path.join(save_directory, json_names[i])
            else:
                output_file = os.path.join(directory_path, json_names[i])
            
            # 写回文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"已修改文件: {json_file} -> 保存于: {output_file}")
        except Exception as e:
            print(f"修改文件: {json_file} 失败，错误信息: {e}")


def modify_libritts_tar_files(directory_path: str, new_wav_path: str, save_directory: str = None) -> None:
    """
    遍历指定目录下的所有.tar文件，将每个tar文件中的所有wav文件的路径替换为指定的目录路径。
    
    Args:
        directory_path: 要替换成的目录路径
        save_directory: 可选，保存修改后文件的目录路径，默认为None（保存在原位置）
    """
    # 获取当前目录下所有.tar文件
    tar_names = [f for f in os.listdir(directory_path) if f.endswith('.tar')]
    tar_files = [os.path.join(directory_path, f) for f in tar_names]

    for i, tar_file in enumerate(tar_files):
        try:
            data = pd.read_parquet(tar_file)

            for _, row in data.iterrows():
                wav_path = row['wav']
                # 获取文件名
                file_name = os.path.basename(wav_path)
                # 构建新路径
                new_path = os.path.join(new_wav_path, file_name)
                row['wav'] = new_path

            # 确定保存路径
            if save_directory:
                # 确保保存目录存在
                os.makedirs(save_directory, exist_ok=True)
                output_file = os.path.join(save_directory,  tar_names[i])
            else:
                output_file = os.path.join(directory_path, tar_names[i])
            
            data.to_parquet(output_file)

            print(f"已修改文件: {tar_file} -> 保存于: {output_file}")
        except Exception as e:
            print(f"修改文件: {tar_file} 失败，错误信息: {e}")


def modify_libritts_parquet(directory_path: str, new_wav_path: str, save_directory: str = None) -> None:
    modify_libritts_list_files(directory_path, save_directory)
    modify_libritts_json_files(directory_path, save_directory)
    modify_libritts_tar_files(directory_path, new_wav_path, save_directory)


if __name__ == "__main__":
    data_path = ""
    save_directory = ""
    new_wav_path = ""
    modify_libritts_parquet(data_path, new_wav_path, save_directory)