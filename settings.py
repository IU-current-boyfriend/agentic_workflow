import os
from os.path import join, dirname
from dotenv import load_dotenv

# 加载环境配置文件
load_dotenv()
# LLM_NAME
LLM_NAME = "LLM_NAME"
LLM_URL = "LLM_URL"
LLM_API_KEY = "LLM_API_KEY"
EMBED_API = "EMBED_API"
EMBED_NAME = "EMBED_NAME"
EMBED_URL = "EMBED_URL"
REDIS_DB = "REDIS_DB"
REDIS_HOST = "REDIS_HOST"
REDIS_PORT = "REDIS_PORT"
REDIS_SPACE = "REDIS_SPACE"
REDIS_PIPELINE = "REDIS_PIPELINE"


# 获取项目根目录
ROOT_PATH = dirname(__file__)
# 获取数据源加载的路径
DATA_SOURCE_PATH = join(ROOT_PATH, 'data')
# 大语言模型命名空间
LLM_NAMESPACE = {
    LLM_NAME:os.getenv("LLM_NAME", None),
    LLM_API_KEY:os.getenv("LLM_API_TOKEN", None),
    LLM_URL:os.getenv("LLM_BASE_URL", None)
}
# 嵌入模型命名空间
EMBED_NAMESPACE = {
    EMBED_NAME:os.getenv("EMBED_MODEL_NAME", None),
    EMBED_API:os.getenv("EMBED_MODEL_API", None),
    EMBED_URL:os.getenv("EMBED_MODEL_URL", None),
}
# redis命名空间
REDIS_NAMESPACE = {
    REDIS_HOST:os.getenv("REDIS_HOST", None),
    REDIS_PORT:os.getenv("REDIS_PORT", None),
    REDIS_DB: os.getenv("REDIS_DB", None),
    REDIS_SPACE: os.getenv("REDIS_NAMESPACE", None),
    REDIS_PIPELINE: os.getenv("REDIS_PIPELINE", None),
}

