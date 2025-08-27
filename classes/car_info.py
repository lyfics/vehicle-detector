import requests
import json
# 定义服务器端的地址
server_url = 'http://127.0.0.1:5000'

def get_info(img):
    # 构造文件数据
    files = {'image': img}  # 替换为你的图片路径

    # 发送POST请求，确保文件键名为'image'
    response = requests.post(server_url, files=files)

    # 假设response是你得到的响应字符串

    # 解析响应字符串为Python字典
    response_dict = json.loads(response.text)

    # 获取'cars'键对应的值，即包含所有车辆信息的列表
    cars = response_dict['cars']

    # 遍历所有车辆信息
    for car in cars:
        # 获取每辆车的颜色和制造商
        color = car['color']
        make = car['make']

        # 打印颜色和制造商
        print("Color:", color)
        print("Make:", make)
        return color,make

