# String Table
# Value Table

# TODO: alignment?

# format:
# "jnd-"
# length:64
# data_offset:64 data_length:64
# strings_offset:64 strings_length:64
# objects_offset:64 objects_length:64

# data indices are either u64, u32, u16, or u8, depending on data_length
# data length is bytes

# strings_offset is in bytes
# strings_length is entries
# where each entry is a [data_index, data_index] pair.

# string indices are either u64, u32, u16, or u8, depending on strings_length

# objects_offset is in bytes
# objects_length is in entries
# where each entry an [object_length, object_start] pair
# where object_length is a string index, object_start is a data index.
# at object_start, there are object_length sorted string indices

#values: 

# 0000 0000 - null
# 0000 0001 - false
# 0000 0010 - true

# 0001 00nn [n bytes]           - int<2**(n+3)>
# 0010 kk00 [k bytes]           - float or double
# 0011 kknn [k bytes] [n bytes] - string: offset, length into string table
# 0100 kknn                     - short array, k element references of n bytes each
# 0101 kknn [k bytes]           - long array, length follows tag, then that many elements of n bytes
# 1000 kknn [k bytes]           - object, object index offset follows, then <length> values of width n

# arrays and objects are followed by N entries of <value index> size, each referencing an element in the value table

# possible CBOR-style sketch

# 000 special
#     00000 - null
#     00001 - false
#     00002 - true
#     00003 - IEEE half-precision float
#     00004 - IEEE single-precision float
#     00005 - IEEE double-precision float
# 001 int
#     nnnnn
# 010 negative int
#     nnnnn
# 011 string
#     nnnnn
# 100 array
#     0xxyy - short array, x elements, y width
#     1xxyy [sizetag x] - long array, y width
# 101 object
#     0xxyy - 

import ctypes
import functools
import sys
import json
import struct

import zlib
import brotli
import zstd

import cbor
import msgpack
import bson
import ubjson

def encode(value):
    value_data = [] # list<bytes>
    object_data = [] # list<bytes>
    object_data_offset = {} # frozenset<str> -> object_data offset
    string_data = [] # list<bytes>
    string_data_offset = {} # str -> string_data offset
    string_value_offset = {} # str -> value_data offset
    int_value_offset = {}
    float_value_offset = {}

    def tagbyte(high_nibble, k, n):
        assert high_nibble < 16
        assert k < 4
        assert n < 4
        return chr((high_nibble << 4) + (k << 2) + n)

    def sizetag(v):
        if v < 1<<8: return 0
        if v < 1<<16: return 1
        if v < 1<<32: return 2
        return 3
        #if v < 1<<64: return 3

    def encode_uint(value, max_value=None):
        if max_value is None:
            max_value = value
        if max_value < 1<<8:
            return struct.pack('<B', value)
        if max_value < 1<<16:
            return struct.pack('<H', value)
        if max_value < 1<<32:
            return struct.pack('<L', value)
        return struct.pack('<Q', value)

    def encode_sint(value):
        if value >= -(1 << 7) and value <= (1 << 7) - 1:
            return 0, struct.pack('<b', value)
        if value >= -(1 << 15) and value <= (1 << 15) - 1:
            return 1, struct.pack('<h', value)
        if value >= -(1 << 31) and value <= (1 << 31) - 1:
            return 2, struct.pack('<l', value)
        return 3, struct.pack('<Q', value)

    primitive_cache = {
        'null': None,
        'true': None,
        'false': None,
    }

    def append_null():
        if primitive_cache['null'] is None:
            value_offset = len(value_data)
            value_data.append(tagbyte(0b0000, 0, 0))
            primitive_cache['null'] = value_offset
        return primitive_cache['null']

    def append_false():
        if primitive_cache['false'] is None:
            value_offset = len(value_data)
            value_data.append(tagbyte(0b0000, 0, 1))
            primitive_cache['false'] = value_offset
        return primitive_cache['false']

    def append_true():
        if primitive_cache['true'] is None:
            value_offset = len(value_data)
            value_data.append(tagbyte(0b0000, 0, 2))
            primitive_cache['true'] = value_offset
        return primitive_cache['true']

    def append_string(value):
        try:
            return string_value_offset[value]
        except KeyError:
            pass
        
        data = value.encode('utf-8')
        string_offset = string_data_offset.get(value)
        if string_offset is None:
            string_offset = len(string_data)
            string_data.extend(data)
            string_data_offset[value] = string_offset
        string_length = len(data)
        
        value_offset = len(value_data)
        value_data.append(tagbyte(0b0011, sizetag(string_offset), sizetag(string_length)))
        value_data.extend(encode_uint(string_offset))
        value_data.extend(encode_uint(string_length))

        string_value_offset[value] = value_offset
        return value_offset

    def append_int(value):
        try:
            return int_value_offset[value]
        except KeyError:
            pass
        
        tag, encoded = encode_sint(value)
        int_offset = len(value_data)
        value_data.append(tagbyte(0b0001, 0, tag))
        value_data.extend(encoded)
        int_value_offset[value] = int_offset
        return int_offset

    def append_float(value):
        try:
            return float_value_offset[value]
        except KeyError:
            pass
        
        float_offset = len(value_data)
        if value == ctypes.c_float(value).value:
            value_data.append(tagbyte(0b0010, 2, 0))
            value_data.extend(struct.pack('<f', value))
        else:
            value_data.append(tagbyte(0b0010, 3, 0))
            value_data.extend(struct.pack('<d', value))

        float_value_offset[value] = float_offset
        return float_offset

    def append_list(value):
        # 0100 kknn                     - short array, k element references of n bytes each
        # 0101 kknn [k bytes]           - long array, length follows tag, then that many elements of n bytes

        if len(value) == 0:
            value_offset = len(value_data)
            value_data.append(tagbyte(0b0100, 0, 0))
            return value_offset

        value_indices = map(append_value, value)
        max_value_index = max(value_indices)

        value_offset = len(value_data)
        if len(value) < 4:
            value_data.append(tagbyte(0b0100, len(value), sizetag(max_value_index)))
            for v in value_indices:
                value_data.extend(encode_uint(v, max_value_index))
        else:
            value_data.append(tagbyte(0b0101, sizetag(len(value)), sizetag(max_value_index)))
            value_data.extend(encode_uint(len(value)))
            for v in value_indices:
                value_data.extend(encode_uint(v, max_value_index))
                
        return value_offset

    def encode_object_shape(object_shape):
        object_shape = sorted(object_shape)
        
        object_length = len(object_shape)
        string_indices = map(append_string, object_shape)
        max_string_offset = max(string_indices) if string_indices else 0
        rv = tagbyte(0, sizetag(object_length), sizetag(max_string_offset))
        rv += encode_uint(object_length)
        for string_index, key in zip(string_indices, object_shape):
            rv += encode_uint(string_index, max_string_offset)
        return rv

    def append_dict(value):
        # 1000 kknn [k bytes]           - object, object index offset follows, then <length> values of width n

        # TODO: assert that keys are all strings
        object_shape = frozenset(value.keys())
        object_shape_offset = object_data_offset.get(object_shape)
        if object_shape_offset is None:
            object_shape_offset = len(object_data)
            object_data.extend(encode_object_shape(object_shape))
            object_data_offset[object_shape] = object_shape_offset

        value_indices = [append_value(value[k]) for k in sorted(object_shape)]
        
        dict_index = len(value_data)
        max_value_index = max(value_indices) if value_indices else 0

        value_data.append(tagbyte(0b1000, sizetag(object_shape_offset), sizetag(max_value_index)))
        value_data.extend(encode_uint(object_shape_offset))
        for value_index in value_indices:
            value_data.extend(encode_uint(value_index, max_value_index))
        return dict_index

    def append_value(value):
        if value is None:
            return append_null()
        elif value is False:
            return append_false()
        elif value is True:
            return append_true()
        elif isinstance(value, unicode):
            return append_string(value)
        elif isinstance(value, int):
            return append_int(value)
        elif isinstance(value, float):
            if value.is_integer():
                return append_int(int(value))
            else:
                return append_float(value)
        elif isinstance(value, list):
            return append_list(value)
        elif isinstance(value, dict):
            return append_dict(value)
        else:
            raise TypeError("what is this thing {!r}".format(type(value)))

    root_index = append_value(value)

    tags = tagbyte(sizetag(root_index), sizetag(len(string_data)), sizetag(len(object_data)))
    return 'jnd\0' + tags + encode_uint(len(object_data)) + encode_uint(len(string_data)) + encode_uint(root_index) + ''.join(object_data) + ''.join(string_data) + ''.join(value_data)

class FormatError(Exception):
    pass

def decode(bytes):
    def tag_to_bytes(k):
        assert k < 4
        return [1, 2, 4, 8][k]

    def decode_uint(data, length_in_bytes):
        if length_in_bytes == 1:
            return ord(data[0])
        elif length_in_bytes == 2:
            return (ord(data[1]) << 8) + ord(data[0])
        elif length_in_bytes == 4:
            return (ord(data[3]) << 24) + (ord(data[2]) << 16) + (ord(data[1]) << 8) + ord(data[0])
        elif length_in_bytes == 8:
            return (decode_uint(data[4:], 4) << 32) + decode_uint(data, 4)
        else:
            raise AssertionError('incorrect length_in_bytes: {}'.format(length_in_bytes))
    
    def decode_sint(data, length_in_bytes):
        fmt = {
            1: '<b',
            2: '<h',
            4: '<i',
            8: '<q',
        }[length_in_bytes]
        return struct.unpack(fmt, data[:length_in_bytes])[0]

    if bytes[0:4] != 'jnd\0':
        raise FormatError('invalid header')

    tags = ord(bytes[4])
    object_data_length_bytes = tag_to_bytes(tags & 3)
    string_data_length_bytes = tag_to_bytes((tags >> 2) & 3)
    root_index_length_bytes = tag_to_bytes((tags >> 4) & 3)

    object_data_length = decode_uint(bytes[5:], object_data_length_bytes)
    string_data_length = decode_uint(bytes[5+object_data_length_bytes:], string_data_length_bytes)
    root_index = decode_uint(bytes[5+object_data_length_bytes+string_data_length_bytes:], root_index_length_bytes)

    object_data_offset = 5 + object_data_length_bytes + string_data_length_bytes + root_index_length_bytes
    string_data_offset = object_data_offset + object_data_length

    value_data_offset = string_data_offset + string_data_length

    object_data = bytes[object_data_offset:][:object_data_length]
    string_data = bytes[string_data_offset:][:string_data_length]
    value_data = bytes[value_data_offset:]
    
    def _decode_value(index):
        assert index < len(value_data), index
        b = ord(value_data[index])
        nibble = (b >> 4) & 15
        k = (b >> 2) & 3
        n = b & 3
        if nibble == 0b0000:
            # primitive
            if b == 0: return None
            elif b == 1: return False
            elif b == 2: return True
            else: raise FormatError("unknown primitive {} at {}".format(b, index))
        elif nibble == 0b0001:
            # int
            return decode_sint(value_data[index+1:], tag_to_bytes(n))
        elif nibble == 0b0010:
            if k == 3:
                # double
                assert 0 == n
                return struct.unpack('<d', value_data[index+1:index+9])[0]
            elif k == 2:
                # float
                assert 0 == n
                return struct.unpack('<f', value_data[index+1:index+5])[0]
            else:
                raise FormatError("unknown float size")
        elif nibble == 0b0011:
            # string
            offset_width = tag_to_bytes(k)
            length_width = tag_to_bytes(n)
            offset = decode_uint(value_data[index+1:], offset_width)
            length = decode_uint(value_data[index+1+offset_width:], length_width)
            return string_data[offset:offset+length].decode('utf-8')
        elif nibble == 0b0100:
            # short array
            element_count = k
            element_width = tag_to_bytes(n)
            rv = []
            for i in range(element_count):
                element_value_index = decode_uint(value_data[index+1+i*element_width:], element_width)
                rv.append(decode_value(element_value_index))
            return rv
        elif nibble == 0b0101:
            # long array
            length_width = tag_to_bytes(k)
            element_width = tag_to_bytes(n)
            element_count = decode_uint(value_data[index+1:], length_width)
            rv = []
            for i in range(element_count):
                element_value_index = decode_uint(value_data[index+1+length_width+i*element_width:], element_width)
                rv.append(decode_value(element_value_index))
            return rv
        elif nibble == 0b1000:
            # object
            object_id = decode_uint(value_data[index+1:], tag_to_bytes(k))
            element_width = tag_to_bytes(n)
            
            object_tag = ord(object_data[object_id])
            object_k = (object_tag >> 2) & 3
            object_n = object_tag & 3
            object_length = decode_uint(object_data[object_id + 1:], tag_to_bytes(object_k))
            string_offset_width = tag_to_bytes(object_n)

            rv = {}
            for i in range(object_length):
                key_index = decode_uint(
                    object_data[object_id+1+tag_to_bytes(object_k)+i*string_offset_width:],
                    string_offset_width)
                key = decode_value(key_index)
                assert isinstance(key, basestring)
                element_value_index = decode_uint(value_data[index+1+tag_to_bytes(k)+i*element_width:], element_width)
                rv[key] = decode_value(element_value_index)
            return rv
        else:
            raise FormatError("unknown nibble {}".format(bin(nibble)))

    value_cache = {} # int -> value
    def decode_value(index):
        if index in value_cache:
            return value_cache[index]
        else:
            v = _decode_value(index)
            value_cache[index] = v
            return v
        
    return decode_value(root_index)

if len(sys.argv) >= 2:
    json_bytes = file(sys.argv[1], 'rb').read()
else:
    json_bytes = sys.stdin.read()
json_data = json.loads(json_bytes)
encoded_bytes = encode(json_data)

def as_hex(byte):
    table = '0123456789abcdef'
    return table[((byte >> 4) & 15)] + table[byte & 15]
#print ''.join(as_hex(ord(c)) for c in encoded_bytes)
#print encoded_bytes

class Format(object):
    def __init__(self, name, encode, decode):
        self.name = name
        self.encode = encode
        self.decode = decode

def bson_encode(v):
    return bson.BSON.encode({'_':v})
def bson_decode(d):
    return bson.BSON.decode(d)['_']

formats = [
    Format(
        name='JND',
        encode=encode,
        decode=decode),
    Format(
        name='JSON',
        encode=functools.partial(json.dumps, separators=(',', ':')),
        decode=json.loads),
    Format(
        name='MsgPack',
        encode=msgpack.packb,
        decode=functools.partial(msgpack.unpackb, encoding='utf-8')),
    Format(
        name='CBOR',
        encode=cbor.dumps,
        decode=cbor.loads),
    Format(
        name='BSON',
        encode=bson_encode,
        decode=bson_decode),
    Format(
        name='UBJSON',
        encode=ubjson.dumpb,
        decode=ubjson.loadb),
]

print "Format,Uncompressed,Zlib,Brotli,Zstd"
for format in formats:
    encoded_bytes = format.encode(json_data)
    decoded_value = format.decode(encoded_bytes)
    assert json_data == decoded_value, (json_data, decoded_value)

    zlib_bytes = zlib.compress(encoded_bytes, 9)
    brotli_bytes = brotli.compress(encoded_bytes)
    zstd_bytes = zstd.compress(encoded_bytes)
    print "%s,%d,%d,%d,%d" % (format.name, len(encoded_bytes), len(zlib_bytes), len(brotli_bytes), len(zstd_bytes))

