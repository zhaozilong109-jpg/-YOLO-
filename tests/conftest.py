"""pytest 公共夹具：把项目根目录加入 sys.path，并提供临时数据目录的测试客户端。"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture()
def client(data_dir: Path):
    app = create_app(data_dir)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture()
def image_bytes():
    from io import BytesIO

    from PIL import Image

    def _make(width: int = 640, height: int = 480, color=(200, 120, 60)):
        buffer = BytesIO()
        Image.new("RGB", (width, height), color).save(buffer, format="PNG")
        buffer.seek(0)
        return buffer

    return _make
