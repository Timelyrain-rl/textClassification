import torch
from typing import Dict, List, Tuple, Set

class HierarchyBuilder:
    def __init__(self):
        self.label_to_id = {'l1': {}, 'l2': {}, 'l3': {}}
        self.id_to_label = {'l1': {}, 'l2': {}, 'l3': {}}
        self.l1_to_l2_map = {}
        self.l2_to_l3_map = {}
        
    def build_from_file(self, file_path: str) -> None:
        """从文件构建层级结构
        Args:
            file_path: 层级结构文件路径
        """
        l1_labels = set()
        l2_labels = set()
        l3_labels = set()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')[0].split('|')
                if len(parts) >= 2:  # 至少包含root和一级分类
                    l1 = parts[1]
                    l1_labels.add(l1)
                    
                    if len(parts) >= 3:  # 包含二级分类
                        l2 = parts[2]
                        l2_labels.add(l2)
                        if l1 not in self.l1_to_l2_map:
                            self.l1_to_l2_map[l1] = set()
                        self.l1_to_l2_map[l1].add(l2)
                        
                        if len(parts) >= 4:  # 包含三级分类
                            l3 = parts[3]
                            l3_labels.add(l3)
                            if l2 not in self.l2_to_l3_map:
                                self.l2_to_l3_map[l2] = set()
                            self.l2_to_l3_map[l2].add(l3)
        
        # 构建标签到ID的映射
        for idx, label in enumerate(sorted(l1_labels)):
            self.label_to_id['l1'][label] = idx
            self.id_to_label['l1'][idx] = label
            
        for idx, label in enumerate(sorted(l2_labels)):
            self.label_to_id['l2'][label] = idx
            self.id_to_label['l2'][idx] = label
            
        for idx, label in enumerate(sorted(l3_labels)):
            self.label_to_id['l3'][label] = idx
            self.id_to_label['l3'][idx] = label
    
    def get_mask_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成掩码矩阵
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (l1_to_l2_mask, l2_to_l3_mask)
        """
        num_l1 = len(self.label_to_id['l1'])
        num_l2 = len(self.label_to_id['l2'])
        num_l3 = len(self.label_to_id['l3'])
        
        l1_to_l2_mask = torch.zeros(num_l1, num_l2)
        l2_to_l3_mask = torch.zeros(num_l2, num_l3)
        
        # 填充一级到二级的掩码矩阵
        for l1, l2_set in self.l1_to_l2_map.items():
            l1_idx = self.label_to_id['l1'][l1]
            for l2 in l2_set:
                l2_idx = self.label_to_id['l2'][l2]
                l1_to_l2_mask[l1_idx, l2_idx] = 1
                
        # 填充二级到三级的掩码矩阵
        for l2, l3_set in self.l2_to_l3_map.items():
            l2_idx = self.label_to_id['l2'][l2]
            for l3 in l3_set:
                l3_idx = self.label_to_id['l3'][l3]
                l2_to_l3_mask[l2_idx, l3_idx] = 1
                
        return l1_to_l2_mask, l2_to_l3_mask
    
    def get_label_counts(self) -> Tuple[int, int, int]:
        """获取各层级的标签数量
        Returns:
            Tuple[int, int, int]: (num_l1, num_l2, num_l3)
        """
        return (
            len(self.label_to_id['l1']),
            len(self.label_to_id['l2']),
            len(self.label_to_id['l3'])
        )