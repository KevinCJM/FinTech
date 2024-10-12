from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.tools.subtitles import SubtitlesClip
import pysrt


# Define function to generate subtitles from SRT file with word wrapping
def generator(txt, max_width=35):
    # Split text into lines that do not exceed max_width characters
    lines = []
    while len(txt) > max_width:
        split_index = txt[:max_width].rfind(' ')
        if split_index == -1:
            split_index = max_width
        lines.append(txt[:split_index])
        txt = txt[split_index:].strip()
    lines.append(txt)
    return '\n'.join(lines)


# Load original video without audio
original_video = VideoFileClip("the_video.mp4").without_audio()

# Load new audio file
audio = AudioFileClip("tts.mp3")

# Calculate the required video duration (audio duration + 1 second)
required_duration = audio.duration + 1

# Repeat the original video until it meets or exceeds the required duration
clips = []
current_duration = 0
while current_duration < required_duration:
    clips.append(original_video)
    current_duration += original_video.duration

# Concatenate the repeated clips to form a video of sufficient length
video = concatenate_videoclips(clips).set_duration(required_duration)

# Add new audio to the video
video = video.set_audio(audio)

# Load subtitles from SRT file using pysrt to properly handle timing
subs = pysrt.open("chinese_output.srt")
subtitles = []
for sub in subs:
    text = generator(sub.text.replace('\n', ' '))
    start_time = sub.start.seconds + sub.start.minutes * 60 + sub.start.hours * 3600 + sub.start.milliseconds / 1000.0
    end_time = sub.end.seconds + sub.end.minutes * 60 + sub.end.hours * 3600 + sub.end.milliseconds / 1000.0
    subtitle = (TextClip(text, font='STHeiti', fontsize=50, color='yellow')
                .set_start(start_time)
                .set_end(end_time)
                .set_position(
        ('center', original_video.h - (100 + 24 * text.count('\n')))))  # Adjust position based on number of lines
    subtitles.append(subtitle)

# Overlay subtitles onto video
final_video = CompositeVideoClip([video] + subtitles)

# Write the final video to a file
final_video.write_videofile("final_video.mp4", codec="libx264", audio_codec="aac")
