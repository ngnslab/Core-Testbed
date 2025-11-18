import os
from string import Template

# template 파일을 읽어서 값을 치환하는 유틸리티 함수들

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "..", "configs")

# template 파일을 로드하는 함수
def load_template(name: str) -> Template:
    path = os.path.join(CONFIG_DIR, name)
    with open(path, "r") as f:
        return Template(f.read())

# template 파일에 값을 치환하여 문자열로 반환하는 함수
def write_config(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
