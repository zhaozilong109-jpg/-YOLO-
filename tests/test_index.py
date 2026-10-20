"""上传去重索引的单元测试。"""

from core.index import UploadIndex, bytes_sha256, file_sha256


def test_add_find_remove(tmp_path):
    index = UploadIndex(tmp_path / "index.json")
    sha = bytes_sha256(b"abc")
    assert index.find(sha) is None

    index.add("a.jpg", sha, 3)
    assert index.find(sha) == "a.jpg"
    assert index.size() == 1

    index.remove("a.jpg")
    assert index.find(sha) is None


def test_persists_across_instances(tmp_path):
    path = tmp_path / "index.json"
    first = UploadIndex(path)
    first.add("a.jpg", bytes_sha256(b"abc"), 3)

    second = UploadIndex(path)
    assert second.find(bytes_sha256(b"abc")) == "a.jpg"
    assert second.size() == 1


def test_sync_drops_missing_files(tmp_path):
    path = tmp_path / "index.json"
    index = UploadIndex(path)
    index.add("a.jpg", bytes_sha256(b"a"), 1)
    index.add("b.jpg", bytes_sha256(b"b"), 1)

    assert index.sync({"a.jpg"}) == 1
    assert index.size() == 1
    assert UploadIndex(path).size() == 1


def test_clear(tmp_path):
    index = UploadIndex(tmp_path / "index.json")
    index.add("a.jpg", bytes_sha256(b"a"), 1)
    index.clear()
    assert index.size() == 0
    assert index.find(bytes_sha256(b"a")) is None


def test_file_sha256_matches_bytes(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"hello world")
    assert file_sha256(path) == bytes_sha256(b"hello world")


def test_corrupt_index_is_ignored(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("{bad json", encoding="utf-8")
    index = UploadIndex(path)
    assert index.size() == 0
    # 仍然可写
    index.add("a.jpg", bytes_sha256(b"a"), 1)
    assert index.find(bytes_sha256(b"a")) == "a.jpg"
