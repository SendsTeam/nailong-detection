from selenium import webdriver
import time
import os
import requests
from selenium.webdriver.common.by import By

# 搜索关键词
keyword = '奶龙'
# URL路径
url = 'https://www.google.com/search?q=' + keyword + '&tbm=isch'
# 内容保存的路径
dir = './data/all/'
# 要爬取的轮数(页数)
round = 200


class Chrome_Image_Crawler:
    # 初始化
    def __init__(self):
        self.url = url

    # 获得浏览器驱动，并访问url
    def init_browser(self):
        browser = webdriver.Chrome()
        # 访问url
        browser.get(self.url)
        # 最大化窗口，之后需要爬取窗口中所见的所有图片
        browser.maximize_window()
        return browser

    # 下载图片
    def download_images(self, browser, round=2):
        # 路径不存在时创建一个
        if not os.path.exists(dir): os.makedirs(dir)
        # 图片序号
        count = 0
        pos = 0
        for i in range(round):
            print('正在进行第'+str(i+1)+'轮爬取')
            # 记录爬取当前的所有url
            img_url_dic = []
            pos += 500
            # 向下滑动
            js = 'var q=document.documentElement.scrollTop=' + str(pos)
            # 执行js代码，使滚动条每次滚动500像素
            browser.execute_script(js)
            # 执行完滚动条之后等待3秒
            time.sleep(3)
            # 获取谷歌图片所在的标签名，即'img'
            img_elements = browser.find_elements(by=By.TAG_NAME, value='img')
            # 遍历抓到的所有webElement
            for img_element in img_elements:
                # 获取每个标签元素内部的url所在连接
                img_url = img_element.get_attribute('src')
                if isinstance(img_url, str):
                    # 过滤掉无效的url
                    if len(img_url) <= 200:
                        # 将无效goole图标筛去
                        if 'images' in img_url:
                            # 每次爬取当前窗口，或许会重复，因此进行去重
                            if img_url not in img_url_dic:
                                try:
                                    img_url_dic.append(img_url)
                                    # 下载并保存图片到当前目录下
                                    filename = dir + str(count) + ".jpg"
                                    r = requests.get(img_url)
                                    with open(filename, 'wb') as f:
                                        f.write(r.content)
                                    f.close()
                                    count += 1
                                    print('成功爬取第' + str(count) + '张图片')
                                    # 防止反爬机制
                                    time.sleep(0.5)
                                except:
                                    print('爬取失败.')

    # 开始爬取
    def BanG_Dream(self, round:int):
        self.__init__()
        browser = self.init_browser()
        # 可以修改爬取的页面数，基本10页是100多张图片
        self.download_images(browser, round)
        browser.close()
        print("爬取完成")


if __name__ == '__main__':
    craw = Chrome_Image_Crawler()
    craw.BanG_Dream(round)