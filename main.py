from Video2Ascii import create_char_gray_dict,video_to_ascii_video,image_to_ascii_image
from string import printable
import cv2
from threading import Thread
import multiprocessing


class AsyncShow:
    def __init__(self):
        self.q = multiprocessing.Queue(1)
        self.thread = multiprocessing.Process(target = self.main,daemon = True)
        self.done = False
        self.thread.start()
        
    def show(self,frame):
        if not self.q.full():
            self.q.put_nowait(frame)

    def main(self):
        cv2.namedWindow("Show", cv2.WINDOW_NORMAL)
        while not self.done:
            frame = self.q.get()
            cv2.imshow("Show", frame)
            cv2.waitKey(1)
    
    def stop(self):
        self.done = True
    

# 使用示例
if __name__ == "__main__":
    # 中文字符集
    with open("zhLib_x3499.txt","r",encoding = "utf-8") as f:
        char_list = list(f.read())
    # ASCII字符集
    char_list.extend(printable)
    # 常用字符集
    char_list.extend("wmzвгдеёжзийклмнопрсoahkbdpqjftZO0QLCJUYX/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
    # 全色字符集
    s = ' ' + \
        '¡¢£¤¥¦§¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ' + \
        '▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟' + \
        '■□▢▣▤▥▦▧▨▩▪▫▬▭▮▯▰▱▲△▴▵▶▷▸▹►▻▼▽▾▿◀◁◂◃◄◅◆◇◈◉◊○◌◍◎●◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯' + \
        '←↑→↓↔↕↖↗↘↙∈∋∑∏−∕∗∘∙√∝∞∟∠∣∥∩∫∴∼≅≈≠≡≤≥' + \
        ''.join([chr(i) for i in range(0x2800, 0x2900)])
    char_list.extend(s)
    
    # 创建字符集
    char_dict = create_char_gray_dict(char_list, font_size = 20,font_path = "msyh")
    
    ashow = AsyncShow() # （可选）实时显示结果
    # 转换视频（视频模式）
    video_to_ascii_video(
        input_video_path = "test.mp4",
        output_video_path = "ascii.mp4",
        char_dict = char_dict,
        font_path = "msyh",
        fps = None, # 手动控制fps，默认与原视频相等
        char_size = 2,  # 字符宽高相同
        output_width_chars = 1024,  # 字符数
        codec = 'avc1',  # 兼容性更好的编码
        progress = True, # 显示进度条
        showCallback = ashow.show,  # （可选）实时显示结果
        text = False  # （可选）是否输出为纯文本生成器
    )
    ashow.stop()
    
    # 转换视频（文本模式）
    r = video_to_ascii_video(
        input_video_path = "test.mp4",
        output_video_path = "ascii.mp4",
        char_dict = char_dict,
        font_path = "msyh",
        fps = None,  # 手动控制fps，默认与原视频相等
        char_size = 2,  # 字符宽高相同
        output_width_chars = 1024,  # 字符数
        codec = 'avc1',  # 兼容性更好的编码
        progress = True,  # 显示进度条
        text = True  # （可选）是否输出为纯文本生成器
    )
    for frame in r:
        print(len(frame))
    '''
    # 转换图片
    image_to_ascii_image(
        "icon.png",
        "icon_.png",
        char_dict
    )'''
    