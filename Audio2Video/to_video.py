import csv
import time
from Audio2Video.to_clean_trans import create_data_for_video
from moviepy.editor import ColorClip, AudioFileClip, TextClip, CompositeVideoClip


def parse_transcript(csv_file):
    subtitles = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subtitles.append({
                'start': float(row['start_time']),
                'end': float(row['end_time']),
                'speaker': row['speaker'],
                'text': row['text'],
                'trans_text': row['trans_text']
            })
    return subtitles


def parse_position(position_str, video_size, text_size):
    x, y = 0, 0
    vw, vh = video_size
    tw, th = text_size

    position_str = position_str.lower()

    # 水平位置
    if 'left' in position_str:
        x = 0
    elif 'right' in position_str:
        x = vw - tw
    else:  # center
        x = (vw - tw) / 2

    # 垂直位置
    if 'top' in position_str:
        y = 0
    elif 'bottom' in position_str:
        y = vh - th
    else:  # center
        y = (vh - th) / 2

    return (x, y)


def create_subtitle_clips(subtitles, video_size, text_key, font, color, position, fontsize, time_offset):
    subtitle_clips = []
    if text_key == 'text':
        change_line = '\n'
    else:
        change_line = ''
    for subtitle in subtitles:
        text_content = f"{subtitle['speaker']}: {change_line}{subtitle[text_key]}"
        txt_clip = TextClip(
            text_content,
            fontsize=fontsize,
            font=font,
            color=color,  # 使用颜色名称字符串
            size=(video_size[0] - 100, None),
            method='caption',
            align='center',
            bg_color='transparent'  # 设置背景为透明
        ).set_duration(subtitle['end'] - subtitle['start'])

        # 解析位置
        txt_position = parse_position(position, video_size, txt_clip.size)

        # 设置位置和时间
        txt_clip = txt_clip.set_position(txt_position).set_start(subtitle['start'] + time_offset).set_end(
            subtitle['end'] + time_offset)
        txt_clip = txt_clip.to_RGB()
        subtitle_clips.append(txt_clip)
    return subtitle_clips


def main(
        audio_file,
        transcript_file,
        output_file='output.mp4',
        background_color=(0, 0, 0),  # 使用 RGB 元组指定背景颜色
        subtitle_font='PingFang SC',
        subtitle_color='white',
        subtitle_position='bottom',
        subtitle_fontsize=40,
        trans_font='PingFang SC',
        trans_color='yellow',
        trans_position='top',
        trans_fontsize=30,
        trans_show=False,
        fps=24,
        time_offset=0.0,
        resolution=(1280, 720)
):
    print("Start to create video...")
    # 加载音频文件
    audio_clip = AudioFileClip(audio_file)
    duration = audio_clip.duration

    # 创建背景视频剪辑
    video_clip = ColorClip(size=resolution, color=background_color, duration=duration)

    # 解析字幕文件
    # subtitles = parse_transcript(transcript_file)
    subtitles = transcript_file

    # 创建原文字幕剪辑
    subtitle_clips = create_subtitle_clips(
        subtitles,
        video_clip.size,
        text_key='text',
        font=subtitle_font,
        color=subtitle_color,
        position=subtitle_position,
        fontsize=subtitle_fontsize,
        time_offset=time_offset
    )

    # 创建翻译字幕剪辑
    if trans_show:
        trans_subtitle_clips = create_subtitle_clips(
            subtitles,
            video_clip.size,
            text_key='trans_text',
            font=trans_font,
            color=trans_color,
            position=trans_position,
            fontsize=trans_fontsize,
            time_offset=time_offset
        )

        final_clip = CompositeVideoClip([video_clip] + subtitle_clips + trans_subtitle_clips)
    else:
        final_clip = CompositeVideoClip([video_clip] + subtitle_clips)
    # 合成字幕剪辑
    final_clip = final_clip.set_audio(audio_clip)

    # 导出视频
    final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac', fps=fps)


if __name__ == '__main__':
    # text_beauty = True
    # trans_yn = True
    # final_text_list = create_data_for_video(text_beauty=text_beauty, trans_yn=trans_yn)
    # # 示例：调用 main 函数，并指定参数
    # main(
    #     audio_file='/Users/chenjunming/Downloads/A股大跌与展望.wav',
    #     transcript_file=final_text_list,
    #     output_file='output.mp4',
    #     background_color=(0, 0, 0),  # 黑色背景
    #     subtitle_font='PingFang SC',  # 字幕字体
    #     subtitle_color='white',  # 原文字幕颜色
    #     subtitle_position='center',  # 原文字幕位置
    #     subtitle_fontsize=40,  # 原文字幕字体大小
    #     trans_show=True if text_beauty and trans_yn else False,  # 是否显示翻译字幕
    #     trans_font='STHeiti',  # 翻译字幕字体
    #     trans_color='yellow',  # 翻译字幕颜色
    #     trans_position='bottom',  # 翻译字幕位置
    #     trans_fontsize=30,  # 翻译字幕字体大小
    #     fps=5,
    #     time_offset=0.5,
    #     resolution=(1280, 720)
    # )
    pass
