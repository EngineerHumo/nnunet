import os
import re

def process_images_and_create_annotation():
    # 定义路径
    images_dir = '/home/wensheng/jiaqi/Zig-RiR/data/prp/images'
    annotations_dir = '/home/wensheng/jiaqi/Zig-RiR/data/prp/annotations'
    train_txt_path = os.path.join(annotations_dir, 'test.txt')
    
    # 确保annotations目录存在
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)
    
    # 获取所有png文件
    png_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    
    # 按文件名排序以确保顺序
    png_files.sort()
    
    processed_filenames = []
    
    for filename in png_files[141:]:
        # 删除.png之前的_0000
        new_filename = re.sub(r'_0000\.png$', '.png', filename)
        
        # 如果文件名有变化，则重命名文件
        if new_filename != filename:
            old_path = os.path.join(images_dir, filename)
            new_path = os.path.join(images_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_filename}")
        else:
            new_filename = filename
        
        # 获取不带后缀的文件名
        filename_without_ext = os.path.splitext(new_filename)[0]
        processed_filenames.append(filename_without_ext)
    
    # 将处理后的文件名写入train.txt
    with open(train_txt_path, 'w', encoding='utf-8') as f:
        for filename in processed_filenames:
            f.write(filename + '\n')
    
    print(f"处理完成！共处理 {len(processed_filenames)} 个文件")
    print(f"文件名已保存到: {train_txt_path}")

# 运行程序
if __name__ == "__main__":
    process_images_and_create_annotation()