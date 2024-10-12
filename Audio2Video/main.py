# -*- encoding: utf-8 -*-
"""
@File: main.py
@Modify Time: 2024/10/10 15:40       
@Author: Kevin-Chen
@Descriptions: 
"""
from Audio2Video.to_video import main
from Audio2Video.to_words import to_words_main
from Audio2Video.to_clean_trans import create_data_for_video

text_beauty = True
trans_yn = True
audio_path = "/Users/chenjunming/Downloads/描述性统计.wav"
output_file_name = "output.mp4"

if __name__ == '__main__':
    # 语音转文字
    text = to_words_main(appid="312e3129", secret_key="ec99dca05ab4a9a56e87c8ab388d83c9",
                         upload_file_path=audio_path)

    # 文字预处理
    final_text_list = create_data_for_video(text_beauty, trans_yn, text)

    # 生成视频
    main(
        audio_file=audio_path,
        transcript_file=final_text_list,
        output_file=output_file_name,
        background_color=(0, 0, 0),  # 黑色背景
        subtitle_font='PingFang SC',  # 字幕字体
        subtitle_color='white',  # 原文字幕颜色
        subtitle_position='center',  # 原文字幕位置
        subtitle_fontsize=40,  # 原文字幕字体大小
        trans_show=True if text_beauty and trans_yn else False,  # 是否显示翻译字幕
        trans_font='STHeiti',  # 翻译字幕字体
        trans_color='yellow',  # 翻译字幕颜色
        trans_position='bottom',  # 翻译字幕位置
        trans_fontsize=20,  # 翻译字幕字体大小
        fps=5,
        time_offset=0.5,
        resolution=(1280, 720)
    )
