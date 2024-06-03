#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/05/21 17:49:18
@Author  :   ChenHao
@Description  :   测试脚本
@Contact :   jerrychen1990@gmail.com
'''

from http import HTTPStatus
from io import BytesIO
import time
import traceback
from PIL import Image
import base64
import os
from tqdm import tqdm
from snippets import load, dump


# prompt = """
# 答
# """

_system = """
你是一个高智商的天才，你正在接受一场智商测试。
你会看到一个图片，图片的左边是3*3排列的9张小图，最后一张小图的内容缺失
图片的右边是6张小图，标号分别是1,2,3,4,5,6。
你需要选择一张最符合左边图片规律的小图，填入左边缺失的小图的空挡中。
"""

prompt = """
请遵循如下格式输出
观察：左边8张小图呈现出来的规律
候选项分析：右边6个候选项表达出来的信息
选择理由：考量#观察#的结果和#候选项分析#的结果，选择最适合的候选项，给出理由
最终选择：候选项的标号,在1-6中选择
"""

SEED = 1

option_map = "ABCDEFG"
option_map = {ch: i for i, ch in enumerate(option_map, 1)}
# print(option_map)


def image_to_base64(image_path):
    # 打开图像文件
    with Image.open(image_path) as image:
        # 创建一个字节流对象
        buffered = BytesIO()
        # 将图像保存到字节流对象中
        image.save(buffered, format="PNG")
        # 获取字节流中的字节数据
        img_bytes = buffered.getvalue()
        # 将字节数据编码为Base64字符串
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


def extract_option(text):
    try:
        import re
        print("content")
        print(text+"\n")
        print("*"*50)
        items = list(re.findall("最终选择[:：](.*)", text, re.DOTALL))
        if items:
            option = items[0].strip()
            options = list(re.findall("(\d|A-Za-z)", option))
            if not options:
                option = None
            else:
                option = options[0]
                option = int(option_map.get(option, option))

        else:
            print("no option found, return option None")
            option = None

        items = list(re.findall("选择理由[:：](.*?)最终选择", text, re.DOTALL))
        if items:
            reason = items[0].strip()
        else:
            print("no reason found, return reason None")
            reason = None
        return option, reason

    except Exception as e:
        traceback.print_exc()
        return None, ""


def test_with_glm(model, base64_str):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=os.environ["ZHIPU_API_KEY"])  # 填写您自己的APIKey
    messages = [
        {"role": "system", "content": _system},
        {
            "role": "user",
            "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_str
                        }
                    }
            ]
        }
    ]

    response = client.chat.completions.create(
        model=model,  # 填写需要调用的模型名称
        messages=messages,
        do_sample=False
    )
    # print(jdumps(messages))
    content = response.choices[0].message.content

    return dict(content=content, messages=messages)


def test_with_qwen(model, base64_str):
    import dashscope
    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    # print(base64_str)
    _prompt = _system + "\n" + prompt

    # print(_prompt)
    messages = [
        # {"role": "system", "content": _system},
        {
            "role": "user",
            "content": [
                {"image": base64_str},
                {"text": _prompt}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(model=model,
                                                     messages=messages,
                                                     top_k=1,
                                                     seed=SEED)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        content = response["output"]["choices"][0]["message"]["content"][0]["text"]
    else:
        # print(response.code)  # The error code.
        content = response.message  # The error message.
    return dict(content=content, messages=messages)

def test_with_customize_openai(model_name, base64_str, base_url, api_key):
    """
        openai client without system role
    """
    API_BASE = base_url

    from openai import OpenAI
    client = OpenAI(base_url=API_BASE, api_key=api_key)  # 填写您自己的APIKey
    _prompt = _system + "\n" + prompt

    image = f"data:image/jpeg;base64,{base64_str}"

    messages = [
        {

            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": _prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image
                    }
                }
            ]
        },
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.,
        top_p=1.
    )
    content = response.choices[0].message.content
    print(content)
    return dict(content=content, messages=messages)



def test_with_gpt(model, base64_str, base_url = "", api_key = ""):
    """ standard openai client. """
    from openai import OpenAI
    client = None
    if len(api_key) == 0:
        api_key = os.environ["OPENAI_API_KEY"]

    if len(base_url) == 0:
        client = OpenAI(api_key=api_key)  # 填写您自己的APIKey
    else:
        client = OpenAI(api_key=api_key, base_url=base_url)  # 填写您自己的APIKey

    messages = [
        {"role": "system", "content": _system},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_str}"
                    }
                }
            ]
        },

    ]
    # print(jdumps(messages))

    response = client.chat.completions.create(
        model=model,  # 填写需要调用的模型名称
        messages=messages,
        temperature=0.,
        top_p=1.
    )
    content = response.choices[0].message.content
    # print(content)
    return dict(content=content, messages=messages)


def test_with_step(model, base64_str):
    return test_with_gpt(model, base64_str, base_url="https://api.stepfun.com/v1", api_key=os.environ["STEP_API_KEY"])

def test_with_yi(model, base64_str):
    API_BASE = "https://api.stepfun.com/v1"
    return test_with_customize_openai("step-1-8k", base64_str, API_BASE, api_key=os.environ["YI_API_KEY"])


model_map = {
    "glm-4v": test_with_glm,
    "gpt-4-turbo": test_with_gpt,
    "gpt-4o": test_with_gpt,
    "qwen-vl-plus": test_with_qwen,
    "qwen-vl-max": test_with_qwen,
    "yi-vision": test_with_yi,
    "step-1v-8k" : test_with_step,
}


def test_model(item, model):
    st = time.time()
    func = model_map.get(model)
    image_path = os.path.join("data/images", item["image"])
    base64_str = image_to_base64(image_path)
    if "qwen" in model:
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        base64_str = os.path.join(cur_dir, "data/images", item["image"])
    rs_item = dict(**item)
    model_rs = func(model, base64_str=base64_str)
    # print(model_rs)

    rs_item.update(**model_rs)

    option, reason = extract_option(rs_item["content"])
    score = 1 if option == item["gold_option"] else 0
    rs_item.update(model=model, cost=time.time() - st, option=option, score=score, reason=reason)

    return rs_item


if __name__ == "__main__":
    data = load("data/test.jsonl")
    #data = load("data/result_1717065081.2147741.jsonl")
    data = data[:]
    # old_result = load("data/result_old.jsonl")
    models = ["glm-4v", "gpt-4o", "qwen-vl-max", "yi-vision", "step-1v-8k"]
    # old_map = {item["image"]: item for item in old_result}

    try_times = 3
    result = []
    for item in tqdm(data):
        try:
            print(f"testing {item}")
            rs_item = dict(**item)
            for model in models:
                print(f"testing {model=}")
                keys = ["cost", "option", "score", "reason"]
                update_dict = {f"{model}_{key}": [] for key in keys}
                update_dict = {k: [None] * try_times if not item.get(k) else item[k] for k in update_dict}

                for i in range(try_times):
                    if update_dict[f"{model}_option"][i]:
                        continue

                    try:
                        test_rs = test_model(item, model)
                        # result.append(rs_item)
                        for k in keys:
                            update_dict[f"{model}_{k}"][i] = test_rs[k]
                    except Exception as e:
                        print(e)
                        i -= 1
                print(update_dict)

                rs_item.update(**update_dict)
                print(f"\n{'*'*50}\n")
            result.append(rs_item)
        except Exception as e:
            print(e.with_traceback())

    dist_path = f"data/result_{int(time.time())}.jsonl"
    print(f"dumping to {dist_path}")

    dump(result, dist_path)
