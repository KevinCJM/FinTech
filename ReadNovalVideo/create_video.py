# 导入moviepy库中的视频和音频处理模块
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips, TextClip
# 导入moviepy库中的字幕处理模块
from moviepy.video.tools.subtitles import SubtitlesClip


# 生成格式化文本，用于后续添加到视频中
def generator(txt, max_width=38):
    """
    将给定的文本按照最大宽度分割成多行

    :param txt: 待格式化的文本
    :param max_width: 每行的最大宽度，默认为38个字符
    :return: 格式化后的文本对象
    """
    lines = []
    while len(txt) > max_width:
        # 寻找合适的位置将文本分割成两部分
        split_index = txt[:max_width].rfind(' ')
        if split_index == -1:
            split_index = max_width
        lines.append(txt[:split_index])
        txt = txt[split_index:].strip()
    lines.append(txt)
    wrapped_text = '\n'.join(lines)
    # 返回按照指定样式格式化的文本对象
    return TextClip(wrapped_text, font='STHeiti', fontsize=50, color='#FFD700',
                    stroke_color='black', stroke_width=1)


# 创建带有音频和字幕的最终视频
def create_video(mp3_file="tts.mp3", mp4_file="the_video.mp4", srt_file="chinese_output.srt",
                 final_video_path="final_video.mp4"):
    """
    合并给定的视频、音频和字幕文件，生成最终的视频文件

    :param mp3_file: 音频文件路径
    :param mp4_file: 视频文件路径
    :param srt_file: 字幕文件路径
    :param final_video_path: 最终生成的视频文件路径
    """
    # 加载原始视频并去除音频
    original_video = VideoFileClip(mp4_file).without_audio()

    # 加载音频
    audio = AudioFileClip(mp3_file)

    # 计算所需的视频总时长
    required_duration = audio.duration + 1

    # 初始化视频片段列表
    clips = []
    current_duration = 0
    # 循环添加视频片段，直到总时长满足要求
    while current_duration < required_duration:
        clips.append(original_video)
        current_duration += original_video.duration

    # 拼接视频片段并设置时长
    video = concatenate_videoclips(clips).set_duration(required_duration)

    # 设置视频的音频
    video = video.set_audio(audio)

    # 创建字幕对象
    subtitles = SubtitlesClip(srt_file, generator)

    # 设置字幕的位置
    subtitles = subtitles.set_position(('center', original_video.h - 150))

    # 合成最终视频
    final_video = CompositeVideoClip([video, subtitles])

    # 导出最终视频文件
    final_video.write_videofile(final_video_path, codec="libx264", audio_codec="aac")


# 程序入口
if __name__ == '__main__':
    # 调用函数创建最终视频
    create_video(mp3_file="tts.mp3", mp4_file="the_video.mp4", srt_file="chinese_output.srt",
                 final_video_path="final_video.mp4")
