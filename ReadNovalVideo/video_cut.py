from flask import Flask, request, jsonify, render_template
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.all import crop, speedx
import os
import platform
import subprocess
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cut_video', methods=['POST'])
def cut_video():
    try:
        video_file = request.files['video']
        start_time = float(request.form.get('start_time', 0))
        end_time = float(request.form.get('end_time', None))
        x1 = int(request.form['x1'])
        y1 = int(request.form['y1'])
        x2 = int(request.form['x2'])
        y2 = int(request.form['y2'])
        speed = float(request.form.get('speed', 1))  # 获取速率参数，默认值为1

        # 保存上传的视频文件
        timestamp = int(time.time())  # 获取当前时间戳
        input_video_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{video_file.filename}")
        video_file.save(input_video_path)

        output_video_path = os.path.join(UPLOAD_FOLDER, f"output_{timestamp}_{video_file.filename}")

        # 加载视频并裁剪
        with VideoFileClip(input_video_path) as video:
            # 如果 end_time 为 None，则到视频结束
            if end_time is None or end_time > video.duration:
                end_time = video.duration

            # 截取指定时间段
            new_video = video.subclip(start_time, end_time)

            # 裁剪指定区域
            new_video = crop(new_video, x1=x1, y1=y1, x2=x2, y2=y2)

            # 调整视频速率
            new_video = speedx(new_video, factor=speed)

            # 保存新视频
            new_video.write_videofile(output_video_path, codec="libx264")

        # 删除原始视频文件
        os.remove(input_video_path)

        return jsonify({"message": "视频裁剪完成", "output_video": output_video_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/open_folder')
def open_folder():
    folder_path = os.path.abspath(UPLOAD_FOLDER)
    if platform.system() == "Windows":
        os.startfile(folder_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.Popen(["open", folder_path])
    else:  # Linux and other
        subprocess.Popen(["xdg-open", folder_path])
    return "文件夹已打开"


if __name__ == '__main__':
    app.run(debug=True)
