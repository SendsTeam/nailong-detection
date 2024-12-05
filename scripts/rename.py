import os


def rename_files_in_folder(folder_path):
    # 获取文件夹中所有文件
    files = os.listdir(folder_path)
    
    # 过滤出文件（排除文件夹）
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    
    # 遍历文件，按索引重命名
    for index, file_name in enumerate(files):
        # 获取文件扩展名
        file_extension = os.path.splitext(file_name)[1]
        print(file_name)
        
        # 创建新的文件名，例如 0.txt, 1.jpg 等
        new_name = f"{index}{file_extension}"
        
        # 获取原文件的完整路径和新文件的完整路径
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_name)
        if (os.path.exists(new_file_path)):
            continue

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {file_name} -> {new_name}")

# 示例：调用函数来重命名文件夹中的文件
folder_path = './data/valid/without-nailong'  # 替换成你要操作的文件夹路径
rename_files_in_folder(folder_path)
