import os
import logging
from typing import Optional

class Logger:
    def __init__(self, log_name: str = "log.txt"):
        # 获取当前工作目录
        cwd: str = os.getcwd()
        data_dir: str = os.path.join(cwd, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        log_path = os.path.join(data_dir, log_name)
        self.logger = logging.getLogger("CS336Logger")
        self.logger.setLevel(logging.INFO)
        # 防止重复添加handler
        if not self.logger.handlers:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def log(self, message: str, caller_name: Optional[str] = None):
        if caller_name is None:
            import inspect
            frame = inspect.currentframe()
            outer_frames = inspect.getouterframes(frame)
            caller_name = outer_frames[1].function
        log_message = f"[{caller_name}] {message}"
        self.logger.info(log_message)
        print(log_message)

def test_logger():
    print("测试Logger...")
    logger = Logger("test_log.txt")
    logger.log("这是一条测试日志。")
    logger.log("第二条日志。")
    print("请检查 data/test_log.txt 文件，确认日志内容已写入。")
