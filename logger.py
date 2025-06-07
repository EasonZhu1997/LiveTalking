import logging
import sys
 
# 配置日志器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 文件处理器 - 使用UTF-8编码
fhandler = logging.FileHandler('livetalking.log', encoding='utf-8')
fhandler.setFormatter(formatter)
fhandler.setLevel(logging.INFO)
logger.addHandler(fhandler)

# 控制台处理器 - 使用UTF-8编码
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
sformatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(sformatter)
# 设置控制台输出编码为UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
logger.addHandler(handler)