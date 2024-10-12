# -*- encoding: utf-8 -*-
"""
@File: trans_to_srt.py
@Modify Time: 2024/10/11 18:15       
@Author: Kevin-Chen
@Descriptions: 
"""
import subprocess

# 定义命令和参数
command = [
    "aeneas_execute_task",
    "tts.mp3",                # 音频文件
    "input.txt",              # 文本文件
    "task_language=zho|is_text_type=plain|os_task_file_format=srt",  # 任务配置
    "chinese_output.srt"      # 输出文件
]

# 调用命令
try:
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("Command output:", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error occurred:", e.stderr)
