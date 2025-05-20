from PIL import Image, ImageDraw, ImageFont
import os
import math
import random

from .common_chars import common_chars_3500, common_symbol

# 选择支持中文的字体（请更换为你系统中的字体）
font_path = "tools/font/simsun.ttc"


def random_chinese_char():
    # 随机生成一个汉字的Unicode编码，范围是 \u4e00 到 \u9fff
    # unicode_code = random.randint(0x4e00, 0x62ff)
    # return chr(unicode_code)
    if random.random() < 0.8:
        # 随机生成一个汉字
        return random.choice(common_chars_3500)
    else:
        # 随机生成一个符号
        return random.choice(common_symbol)

class Generator:
    def __init__(self, min_length, max_length, image_size):
        self.min_length = min_length
        self.max_length = max_length
        self.image_size = image_size

    def generate(self):
        # 随机生成文本长度
        num_chars = random.randint(self.min_length, self.max_length)
        # 生成随机中文文本
        text = ''.join(random_chinese_char() for _ in range(num_chars))
        #print(f"Generated text: {text}")
        #print(f"num_char:{num_chars}")
        # 计算矩阵排布大小（√n）
        grid_size = math.ceil(math.sqrt(num_chars)) if num_chars > 0 else 1  # 每行字符数
        #print(f"grid_size:{grid_size}")
        font_size = int(self.image_size / grid_size)  # 计算字体大小
        #print(f"font_size:{font_size}")
        font = ImageFont.truetype(font_path, font_size)
        # 创建灰度图像
        img = Image.new('L', (self.image_size, self.image_size), color=255)
        draw = ImageDraw.Draw(img)
        # 计算每行的文本，并按行排列
        lines = [text[i:i + grid_size] for i in range(0, num_chars, grid_size)]
        # 获取每行的最大宽度和总高度
        line_heights = []
        max_width = 0
        for line in lines:
            text_bbox = draw.textbbox((0, 0), line, font=font)  # 使用 textbbox()
            line_width = text_bbox[2] - text_bbox[0]
            line_height = text_bbox[3] - text_bbox[1]
            line_heights.append(line_height)
            max_width = max(max_width, line_width)
        total_height = sum(line_heights)  # 所有行的总高度
        # 计算起始坐标，使文本居中
        y_offset = (self.image_size - total_height) // 2  # Y方向居中
        x_offset = (self.image_size - max_width) // 2  # X方向居中
        # 在图片上绘制文本
        for line, line_height in zip(lines, line_heights):
            draw.text((x_offset, y_offset), line, font=font, fill=0)
            y_offset += line_height  # 更新Y坐标，绘制下一行

        # 返回文件
        return img
