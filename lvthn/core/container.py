import struct

MAGIC = b'LVTHN'
VERSION = 1

MEDIA_AUDIO = 0x01
MEDIA_IMAGE = 0x02
MEDIA_VIDEO = 0x03

HDR_FMT = '<5sHB'
HDR_SIZE = struct.calcsize(HDR_FMT)

def wrap(media_type, payload):
    header = struct.pack(HDR_FMT, MAGIC, VERSION, media_type)
    return header + payload

def unwrap(data):
    if len(data) < HDR_SIZE:
        raise ValueError('File too small')
    magic, version, media_type = struct.unpack_from(HDR_FMT, data, 0)
    if magic != MAGIC:
        raise ValueError('Not a Leviathan file')
    if version != VERSION:
        raise ValueError(f'Unsupported version {version}')
    return media_type, data[HDR_SIZE:]