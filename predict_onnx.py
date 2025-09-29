
#####测试版本
import onnx
import numpy as np
onnx_model = onnx.load("/home/wensheng/jiaqi/Zig-RiR/checkpoints/best_model.onnx")
onnx.checker.check_model(onnx_model)
import onnxruntime as ort
ort_session = ort.InferenceSession("/home/wensheng/jiaqi/Zig-RiR/checkpoints/best_model.onnx", providers=['CPUExecutionProvider']) # 创建一个推理session
x = np.random.randn(2,3,1024,1024).astype(np.float32)
print(x.shape)
ort_inputs = {ort_session.get_inputs()[0].name:x}
ort_loss, ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs[0].shape)


'''
import onnx
import numpy as np
import cv2
import os
from pathlib import Path
import onnxruntime as ort

def process_images(input_folder, output_folder, model_path):
    """
    批量处理PNG图像并进行模型预测
    
    参数:
        input_folder: 输入图像文件夹路径
        output_folder: 输出结果文件夹路径  
        model_path: ONNX模型文件路径
    """
    
    # 创建输出文件夹
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 加载ONNX模型
    print("正在加载ONNX模型...")
    ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # 获取支持的图像格式
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    # 遍历输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) 
                  if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    for i, filename in enumerate(image_files):
        print(f"处理第 {i+1}/{len(image_files)} 个图像: {filename}")
        
        # 读取图像
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"警告: 无法读取图像 {filename}, 跳过")
            continue
            
        # 检查图像尺寸
        if image.shape[:2] != (1240, 1240):
            print(f"警告: 图像 {filename} 的尺寸为 {image.shape}, 期望 (1240, 1240), 跳过")
            continue
            
        # 转换为RGB (OpenCV读取的是BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 预处理: 填充图像到1280x1280
        padded_image = np.pad(image_rgb, ((20, 20), (20, 20), (0, 0)), 
                             mode='constant', constant_values=0)
        
        # 转换图像格式为模型输入要求 (1, 3, 1280, 1280)
        input_array = padded_image.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
        input_array = input_array.astype(np.float32) / 255.0  # 归一化到0-1
        input_array = np.expand_dims(input_array, axis=0)  # 添加batch维度
        
        # 模型推理
        ort_inputs = {ort_session.get_inputs()[0].name: input_array}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # 获取预测结果 (1, 4, 1280, 1280)
        prediction = ort_outs[0][0]  # 移除batch维度, 得到(4, 1280, 1280)
        
        # 后处理: 将4个通道转换为4张二值图像
        # 使用softmax获取每个像素的类别概率
        exp_pred = np.exp(prediction - np.max(prediction, axis=0, keepdims=True))
        softmax_pred = exp_pred / np.sum(exp_pred, axis=0, keepdims=True)
        
        # 获取每个像素的预测类别 (0-3)
        class_map = np.argmax(softmax_pred, axis=0)
        
        # 创建4张二值图像 (每个类别一张)
        binary_masks = []
        for class_idx in range(4):
            # 创建二值掩码: 属于该类别的像素为255, 其他为0
            binary_mask = (class_map == class_idx).astype(np.uint8) * 255
            binary_masks.append(binary_mask)
        
        # 创建合并的灰度图像: 不同类别用不同灰度值表示
        gray_combined = np.zeros_like(class_map, dtype=np.uint8)
        # 设置不同类别的灰度值 (可以调整这些值以获得更好的对比度)
        gray_values = [0, 85, 170, 255]  # 类别0-3对应的灰度值
        for class_idx, gray_val in enumerate(gray_values):
            gray_combined[class_map == class_idx] = gray_val
        
        # 裁剪所有图像回1240x1240 (去掉填充的40像素)
        def crop_to_original(img):
            return img[20:1260, 20:1260]  # 从1280x1280裁剪回1240x1240
        
        # 裁剪所有结果图像
        cropped_binary_masks = [crop_to_original(mask) for mask in binary_masks]
        cropped_gray_combined = crop_to_original(gray_combined)
        
        # 保存结果图像
        base_name = os.path.splitext(filename)[0]
        
        # 保存4张二值图像
        for class_idx, binary_mask in enumerate(cropped_binary_masks):
            output_path = os.path.join(output_folder, f"{base_name}_class{class_idx}.png")
            cv2.imwrite(output_path, binary_mask)
        
        # 保存合并的灰度图像
        combined_path = os.path.join(output_folder, f"{base_name}_combined.png")
        cv2.imwrite(combined_path, cropped_gray_combined)
        
        print(f"已完成: {filename} -> 生成5张结果图像")
    
    print("所有图像处理完成!")

# 使用示例
if __name__ == "__main__":
    # 设置路径
    input_folder = "/home/wensheng/gjq_workspace/nnUNet/DATASET/nnUNet_raw/Dataset001_prp/imagesTs"  # 替换为你的输入图像文件夹路径
    output_folder = "/home/wensheng/gjq_workspace/nnUNet/output"  # 替换为你的输出文件夹路径
    model_path = "/home/wensheng/gjq_workspace/nnUNet/DATASET/nnUNet_trained_models/Dataset001_prp/nnUNetTrainer__nnUNetPlans__2d/onnx_exports/fold_all_checkpoint_final.onnx"
    
    # 处理图像
    process_images(input_folder, output_folder, model_path)

'''