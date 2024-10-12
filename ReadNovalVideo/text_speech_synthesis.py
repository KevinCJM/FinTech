#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : iflytek


import sys
import hmac
import json
import time
import random
import base64
import hashlib
import requests
from time import mktime
from datetime import datetime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time


class TestTask():

    def __init__(self):
        self.host = HOST
        self.app_id = APP_ID
        self.api_key = API_KEY
        self.api_secret = API_SECRET

    # 生成鉴权的url
    def assemble_auth_url(self, path):
        params = self.assemble_auth_params(path)
        # 请求地址
        request_url = "http://" + self.host + path
        # 拼接请求地址和鉴权参数，生成带鉴权参数的url
        auth_url = request_url + "?" + urlencode(params)
        return auth_url

    # 生成鉴权的参数
    def assemble_auth_params(self, path):
        # 生成RFC1123格式的时间戳
        format_date = format_date_time(mktime(datetime.now().timetuple()))
        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + format_date + "\n"
        signature_origin += "POST " + path + " HTTP/1.1"
        # 进行hmac-sha256加密
        signature_sha = hmac.new(self.api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        # 构建请求参数
        authorization_origin = 'api_key="%s", algorithm="%s", headers="%s", signature="%s"' % (
            self.api_key, "hmac-sha256", "host date request-line", signature_sha)
        # 将请求参数使用base64编码
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        params = {
            "host": self.host,
            "date": format_date,
            "authorization": authorization
        }
        return params

    # 创建任务
    def test_create(self, the_text, the_speaker, the_speed, the_pitch):
        # 创建任务的路由
        create_path = "/v1/private/dts_create"
        # 拼接鉴权参数后生成的url
        auth_url = self.assemble_auth_url(create_path)
        # 合成文本
        encode_str = base64.encodebytes(the_text.encode("UTF8"))
        txt = encode_str.decode()
        # 请求头
        headers = {'Content-Type': 'application/json'}
        # 请求参数，字段具体含义见官网文档：https://aidocs.xfyun.cn/docs/dts/%E6%8E%A5%E5%8F%A3%E5%8D%8F%E8%AE%AEv3.html

        data = {
            "header": {
                "app_id": self.app_id,
                # "callback_url": "",
                # "request_id": ""
            },
            "parameter": {
                "dts": {
                    "vcn": the_speaker,  # 发音人代码
                    "language": "zh",   # zh：中文(默认); en：英文
                    "speed": the_speed,    # 语速, 取值范围[0-100]，默认50
                    "volume": 70,   # 音量, 取值范围[0-100]，默认50
                    "pitch": the_pitch,    # 音调, 取值范围[0低沉-100尖锐]，默认50
                    "bgs": 0,
                    "reg": 0,
                    "rdn": 0,
                    "scn": 0,
                    "audio": {
                        "encoding": "lame",  # 音频编码, 下方下载的文件后缀需要保持一致 (lame：mp3编码格式)
                        "sample_rate": 16000,   # 采样率，可选值：16000(默认)、8000、24000
                        "channels": 1,
                        "bit_depth": 16,
                        "frame_size": 0
                    },
                    "pybuf": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "plain"
                    }
                }
            },
            "payload": {
                "text": {
                    "encoding": "utf8",
                    "compress": "raw",
                    "format": "plain",
                    "text": txt
                }
            },
        }
        try:
            print("创建任务请求参数:", json.dumps(data))
            res = requests.post(url=auth_url, headers=headers, data=json.dumps(data))
            res = json.loads(res.text)
            return res
        except Exception as e:
            print("创建任务接口调用异常，错误详情:%s" % e)
            sys.exit(1)

    # 查询任务
    def test_query(self, the_task_id):
        # 查询任务的路由
        query_path = "/v1/private/dts_query"
        # 拼接鉴权参数后生成的url
        auth_url = self.assemble_auth_url(query_path)
        # 请求头
        headers = {'Content-Type': 'application/json'}
        # 请求参数，字段具体含义见官网文档：https://aidocs.xfyun.cn/docs/dts/%E6%8E%A5%E5%8F%A3%E5%8D%8F%E8%AE%AEv3.html
        data = {
            "header": {
                "app_id": self.app_id,
                "task_id": the_task_id
            }
        }
        try:
            print("\n查询任务请求参数:", json.dumps(data))
            res = requests.post(url=auth_url, headers=headers, data=json.dumps(data))
            res = json.loads(res.text)
            return res
        except Exception as e:
            print("查询任务接口调用异常，错误详情:%s" % e)
            sys.exit(1)


# 创建任务
def do_create(the_text, the_speaker, the_speed, the_pitch):
    # 调用创建任务接口
    test_task = TestTask()
    create_result = test_task.test_create(the_text, the_speaker, the_speed, the_pitch)
    print("create_response:", json.dumps(create_result))
    # 创建任务接口返回状态码
    the_code = create_result.get('header', {}).get('code')
    # 状态码为0，创建任务成功，打印task_id, 用于后续查询任务
    if the_code == 0:
        the_task_id = create_result.get('header', {}).get('task_id')
        print("创建任务成功，task_id: %s" % the_task_id)
        return the_task_id
    # 状态码非0，创建任务失败, 相关错误码参考官网文档：https://aidocs.xfyun.cn/docs/dts/%E6%8E%A5%E5%8F%A3%E5%8D%8F%E8%AE%AEv3.html
    else:
        print("创建任务失败，返回状态码: %s" % the_code)


# 查询任务
def do_query(the_task_id):
    test_task = TestTask()
    i = 0
    # 这里循环调用查询结果，当task_status状态为'5'（即大文本合成任务完成）时停止循环，循环次数和sleep时间可酌情定义
    while True:
        # 等待1-5秒
        time.sleep(random.randint(1, 5))
        # 调用查询任务接口
        query_res = test_task.test_query(the_task_id)
        print("query_response:", json.dumps(query_res))
        # 查询任务接口返回状态码
        the_code = query_res.get('header', {}).get('code')
        # 状态码为0，查询任务成功
        if the_code == 0:
            # 任务状态码：1-任务创建成功 2-任务派发失败 4-结果处理中 5-结果处理完成
            task_status = query_res.get('header', {}).get('task_status')
            if task_status == '5':
                audio = query_res.get('payload', {}).get('audio').get('audio')
                # base64解码audio，打印下载链接
                decode_audio = base64.b64decode(audio)
                print("查询任务成功，音频下载链接: %s" % decode_audio.decode())
                return decode_audio
            else:
                print("第%s次查询，处理未完成，任务状态码:%s" % (i + 1, task_status))
        else:
            print("查询任务失败，返回状态码: %s" % the_code)
            sys.exit(1)
        i += 1


if __name__ == "__main__":
    # 1、用户参数，相关参数注意修改
    HOST = "api-dx.xf-yun.com"
    APP_ID = "62021897"
    API_KEY = "c9b076d85ff2cfaf70d6b088b8673c19"
    API_SECRET = "ZDVjNzJmZDI3MDFjODVlOTY4NTI3ODgy"
    ''' 朗读者
中文名称	    参数名称（vcn=）	                音色	    语种/方言	风格
希涵	            x4_yeting	                女	    中文/普通话	游戏影视解说
关山-专题	    x4_guanyijie	            男	    中文/普通话	专题片纪录片
小鹏	            x4_pengfei	                男	    中文/普通话	新闻播报
千雪	            x4_qianxue	                女	    中文/普通话	阅读听书
聆伯松-老年男声	x4_lingbosong	            男	    中文/普通话	阅读听书
秀英-老年女声	    x4_xiuying	                女	    中文/普通话	阅读听书
明哥	            x4_mingge	                男	    中文/普通话	阅读听书
豆豆	            x4_doudou	                男	    中文/男童	阅读听书
聆小珊	        x4_lingxiaoshan_profnews	女	    中文/普通话	新闻播报
小果	            x4_xiaoguo	                女	    中文/普通话	新闻播报
小忠	            x4_xiaozhong	            男	    中文/普通话	新闻播报
小露	            x4_yezi	                    女	    中文/普通话	通用场景
超哥	            x4_chaoge	                男	    中文/普通话	新闻播报
飞碟哥	        x4_feidie	                男	    中文/普通话	游戏影视解说
聆飞皓-广告	    x4_lingfeihao_upbeatads     男	    中文/普通话	直播广告
嘉欣	            x4_wangqianqian	            女	    中文/普通话	直播广告
聆小臻	        x4_lingxiaozhen_eclives	    女	    中文/普通话	直播广告
    '''

    speaker = 'x4_mingge'
    speed = 70
    pitch = 50

    # 2、执行创建任务
    file = open("input.txt", encoding='utf-8')
    text = file.read()
    task_id = do_create(text, speaker, speed, pitch)
    file.close()

    # 3、执行查询任务 (创建任务执行成功后，由返回的task_id执行查询任务)
    if task_id:
        query_result = do_query(task_id)
    else:
        raise Exception("创建任务失败！task_id为空！")

    # 4、下载到本地
    download_address = query_result
    print("\n音频下载地址:", download_address)
    f = requests.get(download_address)
    # 下载文件，根据需要更改文件后缀
    filename = "tts.mp3"
    with open(filename, "wb") as code:
        code.write(f.content)
    if filename:
        print("\n音频保存成功！")
