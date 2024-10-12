# -*- encoding: utf-8 -*-
"""
@File: to_clean_trans.py
@Modify Time: 2024/10/10 15:31       
@Author: Kevin-Chen
@Descriptions: 
"""
import re
import ast
import json
import time
import random
import requests
from openai import OpenAI


# 读取语言识别结果, 并合并同一个人说话的句子
def data_clean(text=None):
    # 读取语言识别结果
    if not text:
        text = open('01_to_words.txt').read()
    text = text.replace("[", '')
    text = text.replace("]", '')
    text_list = text.split('},')
    final_list = []

    for data in text_list:
        # 判断字符串是否 } 结尾
        data = data + '}' if not data.endswith('}') else data
        data = json.loads(data)
        final_list.append(data)

    # 合并同一个人说话的句子
    merged_list = []
    current = None

    for entry in final_list:
        if current is None:
            current = entry
        else:
            # 检查说话人是否相同
            if current['speaker'] == entry['speaker']:
                # 合并对话内容，更新结束时间
                current['onebest'] += entry['onebest']
                current['ed'] = entry['ed']
            else:
                # 说话人不同，保存之前的记录并开始新的记录
                merged_list.append(current)
                current = entry

    # 将最后一个说话人的记录添加到结果中
    if current:
        merged_list.append(current)
    return merged_list


# 数据拆分, 以便分批调用接口进行语法优化
def split_text(merged_list):
    # 循环, 每次取出10条数据
    text_list = []
    for i in range(0, len(merged_list), 10):
        sub_text_dict = []
        for j in merged_list[i:i + 10]:
            sub_text_dict.append({'speaker': j['speaker'], 'onebest': j['onebest']})
        text_list.append(sub_text_dict)
    return text_list


# 调用openai接口, 修复语法错误
def fix_text_by_ai(text_list):
    # 初始化OpenAI客户端
    client = OpenAI(base_url='https://api.gptsapi.net/v1',
                    api_key='sk-CTYc31030e7efaeae5bb8cf075320b6308f28796ca3G85jT')

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"text_list 是一个列表, 里面都是字典数据. \n"
                           f"数据是两个人的对话记录, 字典内的speaker键是说话的人, 字典内的onebest是说话的文字内容. \n"
                           f"但是说话文字内容中有 语法错误 或 单词错误, 请修复. \n"
                           f"注意, 对话中有些单词出现错位, 本来应该在是下一个人说话的开头却是变成了上一个人说话的结尾, 请修复. \n"
                           f"不能合并句子, 也不能重建新句子. 也就是不要增加或减少列表中的总对话数据. \n"
                           f"以下是text_list列表内的数据:\n {text_list}\n"
                           f"请返回处理后的 text_list 列表数据. 列表内仍然是字典数据, \n"
                           f"再次提醒! 修改后的列表的总数据量一定要和原本的 text_list 一致, 绝对不要增加或减少列表中的总对话数据.\n"
                           f"本次我给你的列表有 {len(text_list)} 条对话数据, 你返回的列表也必须有 {len(text_list)} 条对话数据.\n"
            }
        ],
        temperature=0.5,
        max_tokens=4096,
        top_p=1
    )

    res_text = response.choices[0].message.content
    return res_text


# 取出修复后的句子
def get_fixed_cell(res_text):
    fixed_list = []

    # 提取方括号中的内容并将其转为Python数据结构
    matches = re.search(r'\[.*\]', res_text, re.DOTALL)
    if matches:
        content = matches.group(0)
        # 将字符串转换为Python对象
        dialogue_list = ast.literal_eval(content)
        # 提取每个'dialogue'中的'onebest'字段
        for item in dialogue_list:
            fixed_list.append(item['onebest'].strip())
    return fixed_list


# 定义翻译函数
def translate_to_chinese(text):
    # 有道翻译API地址
    url = f'https://dict.youdao.com/jsonapi?q={text}'
    response = requests.get(url)
    data = response.json()
    # 随机等待1-4秒, 避免请求过于频繁被封IP
    time.sleep(random.randint(1, 4))

    # 从返回结果中提取翻译内容
    if 'fanyi' in data and 'tran' in data['fanyi']:
        return data['fanyi']['tran']
    elif data['web_trans']['web-translation'][0]['trans'][0]['value']:
        return data['web_trans']['web-translation'][0]['trans'][0]['value']
    else:
        return ""


# 对话数据处理主函数
def main_data_clean(text_beauty=False, text=None):
    cleaned_data = data_clean(text)
    print(f"完成读取语言识别结果, 并合并同一个人说话的句子, 合并后的对话总共有{len(cleaned_data)}条")
    separated_data = split_text(cleaned_data)
    print(f"对话句子拆分完毕, 总共{len(separated_data)}个批次, 准备调用openai接口进行语法优化")
    print(separated_data)

    # 调用openai接口做文字美化
    if text_beauty:
        final_text_list = []
        for i, sentence_list in enumerate(separated_data):
            sub_fixed_text = fix_text_by_ai(sentence_list)
            fixed_text = get_fixed_cell(sub_fixed_text)
            print(f"第{i + 1}批次修复完成, 本批次内有{len(fixed_text)}条对话")
            if len(fixed_text) != len(sentence_list):
                # 如果修复后的对话数量和原始对话数量不一致, 则直接使用原始对话
                fixed_text = [item['onebest'] for item in sentence_list]
                print(f"第{i + 1}批次修复失败, 修复后的对话数量和原始对话数量不一致, 使用原始对话")
            final_text_list = final_text_list + fixed_text
    else:
        final_text_list = [item['onebest'] for item in cleaned_data]

    print(f"全部修复完成, 修复后的对话总共有{len(final_text_list)}条")
    with open('02_sub_final_result.txt', 'w') as f:
        for cc in final_text_list:
            f.write(cc)
            f.write('\n')
    return cleaned_data, final_text_list


# 创建视频的字幕数据
def create_data_for_video(text_beauty=True, trans_yn=False, text=None):
    # 如果不做文字美化, 则不做翻译
    if not text_beauty:
        trans_yn = False

    # 对话内容处理 & 调用ai接口做文字美化 & 保存文档做备份
    cleaned_data, lines = main_data_clean(text_beauty=text_beauty, text=text)

    # 同步循环 cleaned_data 和 lines; 生成字幕数据
    print(f"开始生成字幕数据, 并进行翻译处理, 总共有 {len(cleaned_data)} 条对话数据")
    final_text_list = []
    for cleaned, line in zip(cleaned_data, lines):
        the_dict = dict()
        the_dict['start'] = float(cleaned['bg']) / 1000
        the_dict['end'] = float(cleaned['ed']) / 1000
        the_dict['speaker'] = 'Kevin' if cleaned['speaker'] == '1' else 'Mia'
        the_dict['text'] = line.strip()
        # 调用翻译接口
        the_dict['trans_text'] = translate_to_chinese(line.strip()) if trans_yn else ''
        final_text_list.append(the_dict)
    print(f"字幕数据生成完毕, 总共有 {len(final_text_list)} 条字幕数据")
    with open('03_final_result.txt', 'w') as f:
        for the_c in final_text_list:
            f.write(str(the_c))
            f.write('\n')
    return final_text_list


if __name__ == '__main__':
    pass
