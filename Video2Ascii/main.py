from Video2Ascii import create_char_gray_dict,video_to_ascii_video
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
    
    def done(self):
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
    
    # ashow = AsyncShow() # （可选）实时显示结果
    # 2. 直接转换视频
    video_to_ascii_video(
        input_video_path = "test.mp4",
        output_video_path = "ascii.mp4",
        char_dict = char_dict,
        font_path = "msyh",
        char_size = 16,  # 字符宽高相同
        output_width_chars = 256,  # 字符数
        codec = 'avc1',  # 兼容性更好的编码
        showCallback = None  # ashow.show  # （可选）实时显示结果
    )