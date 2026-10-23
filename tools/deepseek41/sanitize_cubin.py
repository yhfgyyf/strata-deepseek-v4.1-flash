"""Remove build-machine paths from ELF debug sections without changing code."""
import re
import struct


def sections(blob):
    if blob[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected a little-endian ELF64 cubin")
    offset = struct.unpack_from("<Q", blob, 40)[0]
    size, count, names_index = struct.unpack_from("<HHH", blob, 58)
    headers = [struct.unpack_from("<IIQQQQIIQQ", blob, offset + i * size) for i in range(count)]
    names = headers[names_index]
    strings = blob[names[4]:names[4] + names[5]]
    for h in headers:
        name = strings[h[0]:].split(b"\0", 1)[0].decode()
        yield name, h[2], h[4], h[5]


def sanitize(blob):
    result = bytearray(blob)
    for name, flags, offset, size in sections(blob):
        if name not in (".debug_line", ".nv.merc.debug_line"):
            continue
        if flags & 2:
            raise ValueError("refuse to edit an allocated debug section")
        original = blob[offset:offset + size]

        def replace(match):
            path = match[0]
            relative = b"triton/language" if b"triton/language" in path else b"tools/deepseek41"
            if len(relative) > len(path):
                relative = b"source"
            # Keep all ELF/DWARF offsets and string lengths unchanged.
            return relative.ljust(len(path), b" ")

        result[offset:offset + size] = re.sub(rb"/(?:home|Users)/[^\0\n]+", replace, original)
    result = bytes(result)
    if re.search(rb"/(?:home|Users)/", result):
        raise ValueError("private path outside recognized debug sections")
    for name, _, offset, size in sections(blob):
        if name not in (".debug_line", ".nv.merc.debug_line") and blob[offset:offset + size] != result[offset:offset + size]:
            raise ValueError("non-debug cubin section changed: " + name)
    return result
