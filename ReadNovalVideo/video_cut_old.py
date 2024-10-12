# -*- encoding: utf-8 -*-
"""
@File: video_cut.py
@Modify Time: 2024/10/12 09:27       
@Author: Kevin-Chen
@Descriptions: 视频剪切
"""

from moviepy.video.io.VideoFileClip import VideoFileClip


# 从原视频中截取指定时间段并保存为新的视频文件
def cut_video(input_video_path, output_video_path, start_time=None, end_time=None):
    """
    从原视频中截取指定时间段并保存为新的视频文件。
    当 start_time 或 end_time 为 None 时，自动使用视频的开始或结束时间。

    :param input_video_path: 原视频路径
    :param output_video_path: 输出视频路径
    :param start_time: 截取开始时间（单位：秒），默认为视频开始时间
    :param end_time: 截取结束时间（单位：秒），默认为视频结束时间
    """
    # 加载视频
    with VideoFileClip(input_video_path) as video:
        # 如果 start_time 为 None，则从视频开始
        if start_time is None:
            start_time = 0

        # 如果 end_time 为 None，则到视频结束
        if end_time is None:
            end_time = video.duration

        # 截取指定时间段
        new_video = video.subclip(start_time, end_time)
        # 保存到指定路径
        new_video.write_videofile(output_video_path, codec="libx264", remove_temp=True)


if __name__ == '__main__':
    # 示例用法
    input_video = "the_video.mp4"  # 输入视频文件路径
    output_video = "the_video_cut.mp4"  # 输出截取的视频文件路径
    the_start_time = 16  # 开始时间（秒）
    the_end_time = 20  # 结束时间（秒）

    cut_video(input_video, output_video, the_start_time, the_end_time)
