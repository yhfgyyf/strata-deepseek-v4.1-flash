"""Resumable downloads of the release's exact model assets, using only stdlib."""
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from tools.deepseek41.startup_assets import GSQ_VARIANT, TOKENIZER_SHA256, VARIANTS

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("modelscope", "huggingface", "hf-mirror")


def private_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def catalog(variant, mmproj=True):
    if variant not in VARIANTS:
        raise ValueError("unsupported model variant")
    ud = read_json(ROOT / "data/models-1017.json")
    main = read_json(ROOT / "data/models-gsq-rco-1017.json") if variant == GSQ_VARIANT else ud
    selected = []
    for entry in main["files"]:
        if variant != GSQ_VARIANT and not entry["path"].startswith(variant + "/"):
            continue
        selected.append(dict(entry, repository=main["repository"], revision=main["huggingface_revision"], role="main"))
    for name, role in (("dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf", "draft"), ("mmproj-BF16.gguf", "mmproj")):
        if role == "mmproj" and not mmproj:
            continue
        entry = next(x for x in ud["files"] if x["path"] == name)
        selected.append(dict(entry, repository=ud["repository"], revision=ud["huggingface_revision"], role=role))
    selected.append(dict(path="tokenizer.json", bytes=6367257, sha256=TOKENIZER_SHA256, role="tokenizer",
                         repository="deepseek-ai/DeepSeek-V4.1-Flash",
                         revision="314113f8a7dadb388215396ba5fbee2c610a04d6"))
    return selected


def asset_url(asset, source):
    if source not in SOURCES:
        raise ValueError("unsupported download source")
    base = {"modelscope": "https://modelscope.cn/models", "huggingface": "https://huggingface.co",
            "hf-mirror": "https://hf-mirror.com"}[source]
    revision = "master" if source == "modelscope" else asset["revision"]
    return "/".join((base, asset["repository"], "resolve", revision, urllib.parse.quote(asset["path"], safe="/")))


def assets_for_config(cfg):
    assets = catalog(cfg["variant"], bool(cfg.get("mmproj")))
    first = next(x for x in assets if x["role"] == "main")
    if Path(cfg["model"]).name != Path(first["path"]).name:
        raise ValueError("model must name the first original shard for " + cfg["variant"])
    result = []
    for asset in assets:
        destination = (Path(cfg["model"]).parent / Path(asset["path"]).name
                       if asset["role"] == "main" else Path(cfg[asset["role"]]))
        result.append(dict(asset, destination=str(destination)))
    return result


def verify_file(path, asset, receipts, progress=lambda **_: None):
    path = Path(path)
    before = identity(path)
    if before[2] != asset["bytes"]:
        raise ValueError("file size differs from the release manifest: " + str(path))
    key = str(path.absolute())
    receipt = receipts.get(key, {})
    if receipt.get("identity") == before and receipt.get("sha256") == asset["sha256"]:
        progress(phase="verified", file=path.name, bytes=before[2], total=before[2], cached=True)
        return
    digest = hashlib.sha256()
    done, last = 0, 0.0
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
            done += len(block)
            if time.monotonic() - last >= 1:
                progress(phase="verifying", file=path.name, bytes=done, total=before[2])
                last = time.monotonic()
    if before != identity(path):
        raise ValueError("file changed during SHA256 verification: " + str(path))
    if digest.hexdigest() != asset["sha256"]:
        raise ValueError("SHA256 mismatch; file was preserved: " + str(path))
    receipts[key] = dict(identity=before, sha256=digest.hexdigest())
    progress(phase="verified", file=path.name, bytes=done, total=done, cached=False)


def download_file(asset, source, receipts, no_proxy=False, progress=lambda **_: None, url=None):
    """Never replace an existing file. A failed/interrupted download retains its .part."""
    target = Path(asset["destination"])
    if target.exists():
        verify_file(target, asset, receipts, progress)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > asset["bytes"]:
        raise ValueError("partial file is larger than the manifest; preserved: " + str(partial))
    if offset < asset["bytes"]:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if no_proxy else urllib.request.build_opener()
        headers = {"User-Agent": "Strata-DeepSeek-Launcher/1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = "bytes=" + str(offset) + "-"
        request = urllib.request.Request(url or asset_url(asset, source), headers=headers)
        try:
            response = opener.open(request, timeout=45)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{source} returned HTTP {error.code} for {target.name}; partial file preserved. "
                               "Choose another source explicitly if the provider limits large HTTP downloads.") from None
        with response:
            if response.status not in (200, 206):
                raise ValueError("unexpected HTTP status: " + str(response.status))
            if offset or response.status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if (response.status != 206 or not match or int(match[1]) != offset
                        or int(match[2]) != asset["bytes"] - 1 or int(match[3]) != asset["bytes"]):
                    raise ValueError("server did not honor the exact resume range; partial file preserved")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != asset["bytes"] - offset:
                raise ValueError("download size differs from the release manifest")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("encoded download cannot be resumed safely")
            # The caller serializes downloads. Do not append to a partial changed while the request was open.
            actual = partial.stat().st_size if partial.exists() else 0
            if actual != offset:
                raise ValueError("partial file changed while opening the download")
            done, last = offset, 0.0
            with partial.open("ab" if partial.exists() else "xb") as stream:
                while block := response.read(8 << 20):
                    if done + len(block) > asset["bytes"]:
                        raise ValueError("server sent more bytes than the release manifest")
                    stream.write(block)
                    done += len(block)
                    if time.monotonic() - last >= 1:
                        progress(phase="downloading", file=target.name, bytes=done, total=asset["bytes"])
                        last = time.monotonic()
                stream.flush()
                os.fsync(stream.fileno())
            if done != asset["bytes"]:
                raise ValueError("incomplete download; rerun to resume: " + str(partial))
    verify_file(partial, asset, receipts, progress)
    # link() provides no-clobber publication on the same filesystem; the verified inode is retained.
    os.link(partial, target)
    partial.unlink()
    receipts.pop(str(partial.absolute()), None)
    receipts[str(target.absolute())] = dict(identity=identity(target), sha256=asset["sha256"])
