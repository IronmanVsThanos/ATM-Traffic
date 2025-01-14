def rename_weights_keys(state_dict):
    """
    将权重字典中的'reins'替换为'atms'
    
    Args:
        state_dict: 模型的state_dict
        
    Returns:
        new_state_dict: 修改后的state_dict
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # 替换键名中的'reins'为'atms'
        new_key = key.replace('reins', 'atms')
        new_state_dict[new_key] = value
        
        # 打印修改的键名，方便检查
        if key != new_key:
            print(f'Renamed: {key} -> {new_key}')
    
    return new_state_dict

# 使用示例
if __name__ == '__main__':
    # 1. 从文件加载权重
    import torch
    
    # 加载权重文件
    checkpoint = torch.load('/data/DL/code/Rein（复件）/work_dirs/rein_dinov2_mask2former_1024x1024_bs4x2/iter_40000.pth', map_location='cpu')
    
    # 如果权重在'state_dict'键下
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # 重命名键
        new_state_dict = rename_weights_keys(state_dict)
        # 更新checkpoint
        checkpoint['state_dict'] = new_state_dict
    else:
        # 直接重命名checkpoint中的键
        new_checkpoint = rename_weights_keys(checkpoint)
        checkpoint = new_checkpoint
    
    # 保存修改后的权重
    torch.save(checkpoint, '/data/DL/code/Rein（复件）/work_dirs/rein_dinov2_mask2former_1024x1024_bs4x2/atm_iter_40000.pth')
    
    # 2. 打印一些示例键以验证
    for i, (k, v) in enumerate(checkpoint['state_dict'].items() if 'state_dict' in checkpoint else checkpoint.items()):
        if 'atms' in k:
            print(f"Example key {i+1}: {k}")
        if i >= 4:  # 只打印前5个包含'atms'的键
            break