import subprocess
import datetime
import os
from time import sleep

# 定义保存路径
save_path = "./chrome"  # 替换为你想要保存图片的路径

# 确保保存路径存在
if not os.path.exists(save_path):
    os.makedirs(save_path)

# # 设置Chrome选项
# chrome_options = Options()
# chrome_options.add_argument("--headless")  # 如果不需要看到浏览器窗口，可以使用headless模式
# # chrome_options.add_argument("--window-size=3024,1964")  # 设置窗口大小

# # 启动Chrome浏览器
# driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)

# # 打开要截屏的网页
# driver.get("https://iqmentor.io/zh-Hans/iqtest")  # 替换为你要截取的网页URL

# # __next > div.pointer-events-none.fixed.inset-0.flex.items-end.px-4.py-6.sm\:items-start.sm\:p-6

# # 定义截屏区域 (left, top, width, height)
# element = driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div[2]/div')  # 替换为你要截取的元素的CSS选择器
# location = element.location
# size = element.size

# bbox = (location['x'], location['y'], size['width'], size['height'])
# # print(bbox, size)

# # 生成文件名，格式为screenshot_yyyyMMdd_HHmmss.png
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"screenshot_{timestamp}.png"
file_path = os.path.join(save_path, file_name)

# # 截取浏览器页面并保存
# screenshot = driver.get_screenshot_as_png()

# # 保存指定区域的截图

# im = Image.open(BytesIO(screenshot))
# w, h = im.size


# # print(im.)
# # im_cropped = im
# # print((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
# bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
# print(f"bbox:{bbox}")
# # im_cropped = im.crop(bbox)
# im_cropped = im
# im_cropped.save(file_path)

# driver.quit()

# 获取所有屏幕的信息

def get_screens_info():
    result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True)
    return result.stdout

# 截取指定屏幕的截图


def capture_screen_region(screen_number, region, save_path):
    command = f"screencapture -D {screen_number} -R {region} {save_path}"
    subprocess.run(command, shell=True)


# 打印所有屏幕的信息
print(get_screens_info())


for i in range(1, 41):
    file_name = f"IQ{i}.png"
    file_path = os.path.join(save_path, file_name)
    # print(file_path)

    # 示例：截取第1个屏幕
    screen_number = 1
    # save_path = "screenshot_screen1.png"
    region = "40,300,1450,550"
    capture_screen_region(screen_number, region, file_path)

    print(f"Screenshot saved to {file_path}")

    sleep(2.5)
    

