import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from sklearn.preprocessing import MultiLabelBinarizer

class LabelChecker:
    def __init__(self, hierarchy_builder):
        self.hierarchy_builder = hierarchy_builder
        
    def check_raw_labels(self, raw_labels_l1: np.ndarray, raw_labels_l2: np.ndarray, 
                        raw_labels_l3: np.ndarray) -> Dict[str, Set[str]]:
        """检查原始标签数据
        Args:
            raw_labels_l1: 一级标签数组
            raw_labels_l2: 二级标签数组
            raw_labels_l3: 三级标签数组
        Returns:
            Dict[str, Set[str]]: 每个层级中的未知标签
        """
        unknown_labels = {
            'l1': set(),
            'l2': set(),
            'l3': set()
        }
        
        # 获取有效标签列表
        valid_l1_labels = set(self.hierarchy_builder.label_to_id['l1'].keys())
        valid_l2_labels = set(self.hierarchy_builder.label_to_id['l2'].keys())
        valid_l3_labels = set(self.hierarchy_builder.label_to_id['l3'].keys())
        
        # 检查每个层级的标签
        def check_level_labels(raw_labels, valid_labels, level):
            for labels in raw_labels:
                if isinstance(labels, str):
                    labels = labels.split(',')
                else:
                    labels = [str(labels)]
                for label in labels:
                    if label not in valid_labels and label != 'nan':
                        unknown_labels[level].add(label)
        
        check_level_labels(raw_labels_l1, valid_l1_labels, 'l1')
        check_level_labels(raw_labels_l2, valid_l2_labels, 'l2')
        check_level_labels(raw_labels_l3, valid_l3_labels, 'l3')
        
        return unknown_labels
    
    def check_hierarchy_consistency(self) -> Dict[str, List[Tuple[str, int]]]:
        """检查层级关系的一致性
        Returns:
            Dict[str, List[Tuple[str, int]]]: 每个父标签及其子标签数量
        """
        hierarchy_stats = {
            'l1_to_l2': [],
            'l2_to_l3': []
        }
        
        # 检查一级到二级的映射
        for l1, l2_set in self.hierarchy_builder.l1_to_l2_map.items():
            hierarchy_stats['l1_to_l2'].append((l1, len(l2_set)))
            
        # 检查二级到三级的映射
        for l2, l3_set in self.hierarchy_builder.l2_to_l3_map.items():
            hierarchy_stats['l2_to_l3'].append((l2, len(l3_set)))
            
        return hierarchy_stats
    
    def check_label_dimensions(self, labels_l1: np.ndarray, labels_l2: np.ndarray, 
                             labels_l3: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """检查标签维度
        Args:
            labels_l1: 处理后的一级标签矩阵
            labels_l2: 处理后的二级标签矩阵
            labels_l3: 处理后的三级标签矩阵
        Returns:
            Dict[str, Tuple[int, int]]: 每个层级的标签维度和预期维度
        """
        num_l1, num_l2, num_l3 = self.hierarchy_builder.get_label_counts()
        
        return {
            'l1': (labels_l1.shape[1], num_l1),
            'l2': (labels_l2.shape[1], num_l2),
            'l3': (labels_l3.shape[1], num_l3)
        }
    
    def check_duplicate_labels(self) -> Dict[str, List[Tuple[str, List[str]]]]:
        """检查重复的标签
        Returns:
            Dict[str, List[Tuple[str, List[str]]]]: 每个层级中的重复标签及其父标签
        """
        duplicates = {
            'l2': [],  # [(重复的二级标签, [对应的一级标签列表])]
            'l3': []   # [(重复的三级标签, [对应的二级标签列表])]
        }
        
        # 检查二级标签
        l2_to_l1 = {}
        for l1, l2_set in self.hierarchy_builder.l1_to_l2_map.items():
            for l2 in l2_set:
                if l2 not in l2_to_l1:
                    l2_to_l1[l2] = []
                l2_to_l1[l2].append(l1)
        
        for l2, l1_list in l2_to_l1.items():
            if len(l1_list) > 1:
                duplicates['l2'].append((l2, l1_list))
        
        # 检查三级标签
        l3_to_l2 = {}
        for l2, l3_set in self.hierarchy_builder.l2_to_l3_map.items():
            for l3 in l3_set:
                if l3 not in l3_to_l2:
                    l3_to_l2[l3] = []
                l3_to_l2[l3].append(l2)
        
        for l3, l2_list in l3_to_l2.items():
            if len(l2_list) > 1:
                duplicates['l3'].append((l3, l2_list))
        
        return duplicates

    def print_check_results(self, unknown_labels: Dict[str, Set[str]], 
                          hierarchy_stats: Dict[str, List[Tuple[str, int]]], 
                          label_dimensions: Dict[str, Tuple[int, int]]):
        """打印检查结果
        """
        print("=== 标签检查结果 ===")
        
        print("\n1. 未知标签检查:")
        for level, labels in unknown_labels.items():
            if labels:
                print(f"{level}级分类中的未知标签: {labels}")
            else:
                print(f"{level}级分类中没有未知标签")
        
        print("\n2. 层级关系检查:")
        print("一级到二级的映射:")
        for l1, count in sorted(hierarchy_stats['l1_to_l2']):
            print(f"  {l1}: {count}个二级分类")
        print("\n二级到三级的映射:")
        for l2, count in sorted(hierarchy_stats['l2_to_l3']):
            print(f"  {l2}: {count}个三级分类")
        
        print("\n3. 维度检查:")
        for level, (actual, expected) in label_dimensions.items():
            print(f"{level}级分类: 实际维度={actual}, 预期维度={expected}")
            if actual != expected:
                print(f"  警告: {level}级分类的维度不匹配!")
        
        # 添加重复标签检查结果的打印
        print("\n4. 重复标签检查:")
        duplicates = self.check_duplicate_labels()
        
        if duplicates['l2']:
            print("发现重复的二级标签:")
            for l2, l1_list in duplicates['l2']:
                print(f"  {l2} 出现在以下一级分类下: {', '.join(l1_list)}")
        else:
            print("没有发现重复的二级标签")
            
        if duplicates['l3']:
            print("发现重复的三级标签:")
            for l3, l2_list in duplicates['l3']:
                print(f"  {l3} 出现在以下二级分类下: {', '.join(l2_list)}")
        else:
            print("没有发现重复的三级标签")


def main():
    print("开始执行标签检查...")
    
    # 导入必要的模块
    import pandas as pd
    from hierarchy_utils import HierarchyBuilder
    
    # 初始化层级结构构建器
    hierarchy_builder = HierarchyBuilder()
    hierarchy_builder.build_from_file('../visualization/annot_tree_output/classification_tree.txt')
    
    # 加载数据
    print("\n加载数据...")
    df = pd.read_csv('../data/merged_data_cleaned.csv')
    
    # 获取原始标签
    raw_labels_l1 = df['一级分类'].values
    raw_labels_l2 = df['二级分类'].values
    raw_labels_l3 = df['三级分类'].values
    
    # 初始化标签检查器
    checker = LabelChecker(hierarchy_builder)
    
    # 执行检查
    print("\n执行标签检查...")
    unknown_labels = checker.check_raw_labels(raw_labels_l1, raw_labels_l2, raw_labels_l3)
    hierarchy_stats = checker.check_hierarchy_consistency()
    
    # 处理标签数据用于维度检查
    processed_labels_l1 = [[str(label)] for label in raw_labels_l1]
    processed_labels_l2 = [[str(label)] for label in raw_labels_l2]
    processed_labels_l3 = [[str(label)] for label in raw_labels_l3]
    
    # 使用预定义的类别列表
    valid_l1_labels = list(hierarchy_builder.label_to_id['l1'].keys())
    valid_l2_labels = list(hierarchy_builder.label_to_id['l2'].keys())
    valid_l3_labels = list(hierarchy_builder.label_to_id['l3'].keys())
    
    # 使用预定义的类别进行转换
    mlb_l1 = MultiLabelBinarizer(classes=valid_l1_labels)
    mlb_l2 = MultiLabelBinarizer(classes=valid_l2_labels)
    mlb_l3 = MultiLabelBinarizer(classes=valid_l3_labels)
    
    labels_l1 = mlb_l1.fit_transform(processed_labels_l1)
    labels_l2 = mlb_l2.fit_transform(processed_labels_l2)
    labels_l3 = mlb_l3.fit_transform(processed_labels_l3)
    
    label_dimensions = checker.check_label_dimensions(labels_l1, labels_l2, labels_l3)
    
    # 打印检查结果
    checker.print_check_results(unknown_labels, hierarchy_stats, label_dimensions)

if __name__ == '__main__':
    main()