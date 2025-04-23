# -*- coding: utf-8 -*-

"""
PDF转Markdown工具

将PDF文件转换为可读的Markdown格式，保留文档结构、表格、图片和LaTeX公式。
使用marker库进行PDF解析和转换。
"""

import os
import argparse
from pathlib import Path

# 导入marker库相关模块
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

from PIL import Image
import io


def convert_pdf_to_markdown(pdf_path, output_path=None, config=None):
    """
    将PDF文件转换为Markdown格式
    
    Args:
        pdf_path (str): PDF文件路径
        output_path (str, optional): 输出文件路径，默认为与PDF同名的.md文件
        config (dict, optional): 转换配置参数
        
    Returns:
        str: 生成的Markdown文本路径
    """
    # 如果未指定输出路径，则使用默认路径
    if output_path is None:
        pdf_file = Path(pdf_path)
        output_path = str(pdf_file.with_suffix('.md'))
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置默认配置
    default_config = {
        "output_format": "markdown",
        "extract_images": True,
        "extract_tables": True,
        "extract_formulas": True
    }
    
    # 合并用户配置
    if config:
        default_config.update(config)
    
    # 创建配置解析器
    config_parser = ConfigParser(default_config)
    
    print(f"正在转换 {pdf_path} 为Markdown格式...")
    
    # 创建PDF转换器
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )
    
    # 执行转换
    rendered = converter(pdf_path)
    
    # 从渲染结果中提取文本和图片
    text, _, images = text_from_rendered(rendered)
    
    # 保存Markdown文本
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"转换完成，Markdown文件已保存至: {output_path}")
    
    # 如果图片，保存图片
    if images:
        images_dir = os.path.join(os.path.dirname(output_path), 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        print(f"正在保存 {len(images)} 张图片到 {images_dir}...")
        for i, img_path in enumerate(images):
            try:
                # 判断是否为二进制数据（bytes）
                if isinstance(img_path, bytes):
                    img_data = img_path
                elif isinstance(img_path, str):
                    # 如果是路径，先判断文件是否存在
                    if os.path.exists(img_path):
                        with open(img_path, 'rb') as f:
                            img_data = f.read()
                    else:
                        # 尝试从rendered中查找同名图片的二进制数据
                        img_data = None
                        if hasattr(rendered, 'images') and isinstance(rendered.images, dict):
                            img_data = rendered.images.get(img_path, None)
                        if img_data is None:
                            print(f"图片路径不存在且未找到二进制数据（序号{i+1}）: {img_path}")
                            continue
                else:
                    print(f"未知图片数据类型（序号{i+1}）: {type(img_path)}")
                    continue
                
                # 如果img_data是Image对象，则转换为字节数据
                if isinstance(img_data, Image.Image):
                    img_byte_arr = io.BytesIO()
                    img_data.save(img_byte_arr, format='PNG')
                    img_data = img_byte_arr.getvalue()
                
                img = Image.open(io.BytesIO(img_data))
                img_save_path = os.path.join(images_dir, f"image_{i+1}.png")
                img.save(img_save_path)
            except Exception as e:
                print(f"保存图片失败（序号{i+1}）: {str(e)}")
        
        print(f"图片保存完成")
    
    return output_path


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将PDF文件转换为Markdown格式')
    parser.add_argument('pdf_path', help='PDF文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('--no-images', action='store_true', help='不提取图片')
    parser.add_argument('--no-tables', action='store_true', help='不提取表格')
    parser.add_argument('--no-formulas', action='store_true', help='不提取公式')
    
    args = parser.parse_args()
    
    # 设置配置
    config = {
        "extract_images": not args.no_images,
        "extract_tables": not args.no_tables,
        "extract_formulas": not args.no_formulas
    }
    
    # 执行转换
    convert_pdf_to_markdown(args.pdf_path, args.output, config)


if __name__ == "__main__":
    # 如果直接运行脚本，则转换默认的PDF文件
    pdf_file = os.path.join(os.path.dirname(__file__), 'original.pdf')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, 'original.md')
    
    if os.path.exists(pdf_file):
        convert_pdf_to_markdown(pdf_file, output_file)
    else:
        print(f"错误：找不到PDF文件 {pdf_file}")
        print("请将PDF文件放在脚本同目录下，并命名为'original.pdf'")
        main()