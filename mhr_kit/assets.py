# Copyright (c) 2026 -- helper utilities for the MHR body model. Apache-2.0, like MHR itself.
"""Selective downloader for the MHR release assets.

Why this exists
---------------
The official ``mhr-download-assets`` command pulls the whole 199 MB ``assets.zip``
release archive -- rig and pose correctives for *all seven* levels of detail plus
the TorchScript export -- and then unzips it. Almost every workflow touches a
single LOD, so this module downloads *only the members it is asked for*.

It does that with HTTP range requests, which GitHub release storage supports:

1. read the tail of the archive  -> the ZIP "end of central directory" record,
   which says where the central directory lives;
2. read the central directory    -> name, size, CRC and byte offset of every member;
3. read one member at a time     -> compressed bytes, inflated locally with
   :mod:`zlib` and verified against the stored CRC-32.

For the default LOD 1 that is ~27 MB instead of ~199 MB, and the archive is never
staged on disk. All of the reads share a single HTTPS connection.

Command line::

    python -m mhr_kit.assets --lod 1                 # rig + correctives for LOD 1
    python -m mhr_kit.assets --lod 4 --torchscript   # coarse LOD + TorchScript model
"""

from __future__ import annotations

import argparse
import http.client
import struct
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

# Release this project was written against. Pinning keeps the bytes reproducible.
DEFAULT_RELEASE = "v1.0.1"
ARCHIVE_URL = "https://github.com/facebookresearch/MHR/releases/download/{release}/assets.zip"

# The SAM 3D Body example predictions live in the git tree, not the release
# archive, so they are fetched individually from raw.githubusercontent.com.
SAM3D_URL = (
    "https://raw.githubusercontent.com/facebookresearch/MHR/{release}"
    "/tools/mhr_smpl_conversion/data/sam3d_body_outputs/{name}"
)
SAM3D_FILES = ("img_subj00.npz", "img_subj01.npz", "img_subj02.npz", "img_subj03.npz")

# Project-relative default locations, so every script agrees on where things are.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"

# Members every LOD needs: the parameter transform (which of the 204 model
# parameters drives which joint degree of freedom) and the sparse activation
# pattern of the pose-corrective network.
SHARED_MEMBERS = (
    "assets/compact_v6_1.model",
    "assets/corrective_activation.npz",
    "assets/LICENSE.txt",
)
TORCHSCRIPT_MEMBER = "assets/mhr_model.pt"

# Uncompressed size of the pose-corrective blendshape matrix per LOD, in MB. It
# is held in RAM in full, so it dominates the memory footprint of a loaded model.
LOD_BLENDSHAPE_MB = {0: 2651, 1: 664, 2: 384, 3: 176, 4: 89, 5: 35, 6: 21}


def members_for_lod(lod: int) -> tuple[str, ...]:
    """Return the archive members required to build an :class:`mhr.mhr.MHR` at ``lod``."""
    return (
        f"assets/lod{lod}.fbx",  # skinned rig + identity/expression blendshapes
        f"assets/corrective_blendshapes_lod{lod}.npz",  # non-linear pose correctives
        *SHARED_MEMBERS,
    )


# --------------------------------------------------------------------------- #
# Reading byte ranges of a remote file over one connection
# --------------------------------------------------------------------------- #


class _RangeReader:
    """Random-access reader for a remote file, using HTTP range requests.

    A single keep-alive connection is used for every read, which matters: the
    TLS handshake to GitHub's release storage can take far longer than the
    transfers themselves.
    """

    def __init__(self, url: str, timeout: float = 300.0) -> None:
        final_url = _resolve_redirects(url)
        parts = urllib.parse.urlsplit(final_url)
        self._path = parts.path + (f"?{parts.query}" if parts.query else "")
        self._conn = http.client.HTTPSConnection(parts.hostname, parts.port or 443, timeout=timeout)
        # A one-byte read is the cheapest way to learn the total length and to
        # confirm the server honours ranges (it must answer 206 + Content-Range).
        content_range, _ = self._request(0, 1)
        self.size = int(content_range.split("/")[-1])

    def _request(self, start: int, length: int) -> tuple[str, bytes]:
        """Fetch ``[start, start + length)`` and return ``(Content-Range, body)``."""
        headers = {
            "User-Agent": "mhr-kit",
            "Range": f"bytes={start}-{start + length - 1}",
            "Accept-Encoding": "identity",  # never gzip: byte offsets must stay exact
        }
        self._conn.request("GET", self._path, headers=headers)
        response = self._conn.getresponse()
        try:
            if response.status != 206:
                raise RuntimeError(f"server did not honour range request (HTTP {response.status})")
            return response.headers.get("Content-Range", ""), response.read()
        finally:
            response.close()

    def read(self, start: int, length: int) -> bytes:
        """Return ``length`` bytes of the remote file starting at ``start``."""
        _, data = self._request(start, length)
        if len(data) != length:
            raise RuntimeError(f"short read: asked for {length} bytes at {start}, got {len(data)}")
        return data

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "_RangeReader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _resolve_redirects(url: str) -> str:
    """Return the final URL of ``url`` without downloading its body."""

    class _Stop(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102, ANN001
            return None  # stop, so we can read Location ourselves

    opener = urllib.request.build_opener(_Stop)
    request = urllib.request.Request(url, headers={"User-Agent": "mhr-kit"})
    try:
        with opener.open(request) as response:
            return response.url
    except urllib.error.HTTPError as error:
        location = error.headers.get("Location")
        if error.code in (301, 302, 303, 307, 308) and location:
            return urllib.parse.urljoin(url, location)
        raise


# --------------------------------------------------------------------------- #
# Minimal partial-ZIP reader
# --------------------------------------------------------------------------- #

_EOCD_SIGNATURE = b"PK\x05\x06"  # end of central directory record
_CD_SIGNATURE = b"PK\x01\x02"  # central directory file header
_ZIP64_SENTINEL = 0xFFFFFFFF


@dataclass(frozen=True)
class _Member:
    """One entry of the archive's central directory."""

    name: str
    offset: int  # byte offset of the member's local file header
    compress_size: int
    file_size: int
    compress_type: int  # 0 = stored, 8 = deflate
    crc: int


def _read_central_directory(reader: _RangeReader) -> dict[str, _Member]:
    """Parse the archive's central directory into ``{member name: _Member}``."""
    # The end-of-central-directory record is 22 bytes plus an optional comment,
    # so reading the last kilobyte is always enough to find it.
    tail_length = min(22 + 1024, reader.size)
    tail = reader.read(reader.size - tail_length, tail_length)
    eocd = tail.rfind(_EOCD_SIGNATURE)
    if eocd < 0:
        raise RuntimeError("no ZIP end-of-central-directory record found")
    cd_size, cd_offset = struct.unpack("<II", tail[eocd + 12 : eocd + 20])
    if _ZIP64_SENTINEL in (cd_size, cd_offset):
        raise RuntimeError("ZIP64 archives are not supported by this minimal reader")

    blob = reader.read(cd_offset, cd_size)
    members: dict[str, _Member] = {}
    pos = 0
    while blob.startswith(_CD_SIGNATURE, pos):
        # Fixed 46-byte header, little endian:
        #   0 signature, 10 compression method, 16 crc32, 20 compressed size,
        #   24 uncompressed size, 28/30/32 name/extra/comment lengths,
        #   42 offset of the local file header.
        (compress_type,) = struct.unpack("<H", blob[pos + 10 : pos + 12])
        crc, compress_size, file_size = struct.unpack("<III", blob[pos + 16 : pos + 28])
        name_len, extra_len, comment_len = struct.unpack("<HHH", blob[pos + 28 : pos + 34])
        (offset,) = struct.unpack("<I", blob[pos + 42 : pos + 46])
        name = blob[pos + 46 : pos + 46 + name_len].decode("utf-8")
        members[name] = _Member(name, offset, compress_size, file_size, compress_type, crc)
        pos += 46 + name_len + extra_len + comment_len
    return members


def _fetch_member(reader: _RangeReader, member: _Member, destination: Path) -> None:
    """Download one archive member and write its inflated bytes to ``destination``."""
    # The local file header repeats the name and extra fields, and only the
    # header itself tells us how long they are, so read its fixed part first.
    header = reader.read(member.offset, 30)
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    raw = reader.read(member.offset + 30 + name_len + extra_len, member.compress_size)

    if member.compress_type == 8:
        data = zlib.decompress(raw, -zlib.MAX_WBITS)  # negative wbits = raw deflate
    elif member.compress_type == 0:
        data = raw
    else:
        raise RuntimeError(f"unsupported compression method {member.compress_type} for {member.name}")
    if len(data) != member.file_size or zlib.crc32(data) != member.crc:
        raise RuntimeError(f"corrupt download for {member.name}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def download_assets(
    lod: int = 1,
    torchscript: bool = False,
    dest: Path = ASSET_DIR,
    release: str = DEFAULT_RELEASE,
    force: bool = False,
) -> Path:
    """Download exactly the release assets needed for ``lod`` into ``dest``.

    Args:
        lod: Level of detail, 0 (densest) to 6 (coarsest).
        torchscript: Also fetch ``mhr_model.pt`` (26 MB), the self-contained
            TorchScript export of the LOD 1 model.
        dest: Destination folder; the members are written flat, without the
            ``assets/`` prefix they carry inside the archive.
        release: Release tag of the MHR repository to pull from.
        force: Re-download files that are already present.

    Returns:
        The folder the assets were written to; pass it to :func:`mhr_kit.model.load_mhr`.
    """
    if lod not in LOD_BLENDSHAPE_MB:
        raise ValueError(f"lod must be one of {sorted(LOD_BLENDSHAPE_MB)}, got {lod}")

    wanted = list(members_for_lod(lod))
    if torchscript:
        wanted.append(TORCHSCRIPT_MEMBER)

    dest.mkdir(parents=True, exist_ok=True)
    todo = [name for name in wanted if force or not (dest / Path(name).name).exists()]
    if not todo:
        print(f"All assets for LOD {lod} are already in {dest}")
        return dest

    url = ARCHIVE_URL.format(release=release)
    print(f"Reading archive index: {url}")
    with _RangeReader(url) as reader:
        catalog = _read_central_directory(reader)
        missing = [name for name in todo if name not in catalog]
        if missing:
            raise RuntimeError(f"release {release} does not contain: {', '.join(missing)}")

        total_mb = sum(catalog[name].compress_size for name in todo) / 1e6
        print(f"Fetching {len(todo)} of {len(catalog)} members ({total_mb:.1f} MB compressed):")
        for name in todo:
            member = catalog[name]
            print(
                f"  {Path(name).name:38s}"
                f" {member.compress_size / 1e6:6.1f} MB ->{member.file_size / 1e6:8.1f} MB on disk"
            )
            _fetch_member(reader, member, dest / Path(name).name)

    print(f"Assets ready in {dest}")
    if lod in (0, 1):
        print(
            f"Note: the LOD {lod} pose correctives need ~{LOD_BLENDSHAPE_MB[lod]} MB of RAM once loaded; "
            "use a coarser LOD (e.g. --lod 4) for lightweight experiments."
        )
    return dest


def download_sam3d_examples(dest: Path = DATA_DIR, release: str = DEFAULT_RELEASE, force: bool = False) -> Path:
    """Download the four example SAM 3D Body predictions used by the image/video demos.

    Each ``.npz`` holds the MHR parameters that SAM 3D Body regressed from a single
    photograph (see :mod:`mhr_kit.sam3d`), which gives the demos genuine
    "image -> MHR" inputs without a GPU or the SAM 3D Body checkpoint.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for name in SAM3D_FILES:
        target = dest / name
        if target.exists() and not force:
            continue
        url = SAM3D_URL.format(release=release, name=name)
        print(f"Downloading {name}")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mhr-kit"})) as response:
            target.write_bytes(response.read())
    print(f"SAM 3D Body examples ready in {dest}")
    return dest


def main() -> None:
    """Command line entry point: ``python -m mhr_kit.assets``."""
    parser = argparse.ArgumentParser(description="Download only the MHR assets you need.")
    parser.add_argument(
        "--lod", type=int, default=1, choices=sorted(LOD_BLENDSHAPE_MB), help="level of detail to fetch (default: 1)"
    )
    parser.add_argument("--torchscript", action="store_true", help="also fetch the TorchScript model (26 MB)")
    parser.add_argument("--dest", type=Path, default=ASSET_DIR, help="destination folder for the assets")
    parser.add_argument("--release", default=DEFAULT_RELEASE, help=f"MHR release tag (default: {DEFAULT_RELEASE})")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--skip-examples", action="store_true", help="do not fetch the SAM 3D Body example data")
    args = parser.parse_args()

    download_assets(args.lod, args.torchscript, args.dest, args.release, args.force)
    if not args.skip_examples:
        download_sam3d_examples(release=args.release, force=args.force)


if __name__ == "__main__":
    main()
