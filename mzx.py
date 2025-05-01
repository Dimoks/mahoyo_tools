
from io import BufferedRandom, BufferedReader, BytesIO
from math import ceil
from typing import Literal, Optional, Union, cast, overload
import numpy as np

from .utils.io import BytesRW, BytesReader, BytesWriter

MZX_FILE_MAGIC = b"MZX0"

class MzxCmd :
    RLE = 0
    BACKREF = 1
    RINGBUF = 2
    LITERAL = 3

class RingBuffer :
    def __init__(self, size: int, baseValue: int = None) -> None:
        self._size = size
    
        if baseValue is not None:
            self._file = BytesIO(bytes([baseValue]) * size)
        else:
            self._file = BytesIO()

    def append(self, buffer: bytes) -> None :
        pos = self._file.tell()
        
        if pos + len(buffer) <= self._size :
            self._file.write(buffer)
        else :
            split_index = self._size - pos
            self._file.write(buffer[:split_index])
            self._file.seek(0)
            self._file.write(buffer[split_index:])
    
    def get(self, index: int, size: int) -> bytes :
        if self._file.tell() == 0:
            return b''

        pos = self._file.tell()
        self._file.seek(index)
        result = self._file.read(size)
        self._file.seek(pos)
        return result
    
    def peek_back(self, size: int) -> bytes :
        pos = self._file.tell()
        read_pos = pos - size
        if read_pos >= 0 :
            self._file.seek(read_pos)
            return self._file.read(size)
        else :
            self._file.seek(read_pos, 2)
            part1 = self._file.read(-read_pos)
            self._file.seek(0)
            return part1 + self._file.read(pos)
    
    def word_position(self, word: bytes, word_size: int) :
        pos = self._file.tell()
        self._file.seek(0)
        index = -1
        while self._file.tell() < self._size :
            stored = self._file.read(word_size)
            if not stored:
                break
            if stored == word :
                index = self._file.tell() - word_size
                break
        self._file.seek(pos)
        return index

def rle_compress_length(words: np.ndarray, cursor: int, clear_count: int,
                        invert_bytes: bool) :
    
    word = words[cursor]
    if clear_count <= 0 or clear_count == 0x1000:
        last = 0xFFFF if invert_bytes else 0x0000
    else :
        last = words[cursor-1]
    if word == last :
        max_chunk_size = min(64, words.size - cursor)
        max_chunk_size = min(max_chunk_size, clear_count)
        length = 1
        while length < max_chunk_size and words[cursor+length] == last :
            length += 1
        return length
    else :
        return 0

def backref_compress_length(words: np.ndarray, cursor: int, clear_count: int) :
    start = max(cursor - 255, 0)
    occurrences = np.where(words[start:cursor] == words[cursor])[0]
    max_length = min(64, clear_count)
    max_dist = 0x1000 - clear_count
    best_index = -1
    best_len = 0
    for o in occurrences :
        distance = cursor - (start + o)
        if distance > max_dist or distance < 0 :
            continue
        length = 1
        while (length < max_length and
               cursor + length < words.size and
               words[start + o + length] == words[cursor + length]) :
            length += 1
        if (length > best_len or (length == best_len and
                distance < (cursor - (start + best_index)))) :
            best_index = o
            best_len = length
    if best_len > 0 :
        distance = cursor - (start + best_index)
        return (distance, best_len)
    else :
        return (0, 0)

def literal_compress(output: BytesWriter, words: np.ndarray, start: int,
                     length: int, invert: bool) :

    assert length <= 64
    cmd = (MzxCmd.LITERAL | ((length - 1) << 2))
    output.write(cmd.to_bytes(1))
    if invert :
        chunk = words[start:start+length] ^ 0xFFFF
    else :
        chunk = words[start:start+length]
    output.write(chunk.tobytes())

@overload
def mzx_compress(src: BytesReader, dest: BytesWriter,
                 invert: bool = False, level: int = 0) -> None : ...
@overload
def mzx_compress(src: BytesReader, dest: None = None,
                 invert: bool = False, level: int = 0) -> BytesIO : ...
def mzx_compress(src: BytesReader, dest: BytesWriter | None = None,
                 invert: bool = False, level: int = 0) :
    src = open(src, 'rb') if isinstance(src, str) else src
    start = src.tell()
    end = src.seek(0, 2)
    src.seek(start)
    match dest :
        case None : output = BytesIO()
        case str() : output = open(dest, "wb+")
        case _ : output = dest
    header = MZX_FILE_MAGIC + (end - start).to_bytes(4, 'little', signed=False)
    output.write(header)
    words = np.frombuffer(src.read(), dtype=np.uint8)
    if words.size % 2 == 1 :
        words = np.append(words, 0x00)
    words = words.view(dtype="<u2")
    end = words.size
    inversion_xor = 0xFFFF if invert else 0x0000
    if level == 0 :
        cursor = 0
        while cursor < end :
            # Len field is 6 bits, each word is 2 bytes,
            # we can write 64 words (128 bytes) per literal record
            chunk_size = min(end - cursor, 64)

            cmd: int = (MzxCmd.LITERAL | ((chunk_size - 1) << 2))
            output.write(cmd.to_bytes(1, 'little'))
            output.write(words[cursor:cursor+chunk_size] ^ inversion_xor)
            cursor += chunk_size
    else :
    
        clear_count = 0x1000
        ring_buf = RingBuffer(128)
        lit_start = 0
        lit_len = 0
        best_len = 0
        best_type = 0 # 0: LITERAL, 1: RLE, 2: BACKREF, 3: RINGBUF
        cursor = 0
        while cursor < end :
            best_len = 0
            best_type = 0 # LITERAL
            current_word = (words[cursor] ^ inversion_xor).tobytes()
            rle_len = rle_compress_length(words, cursor, clear_count, invert)
            if rle_len > 0 :
                best_len = rle_len
                best_type = 1 # RLE
            #print(f"{cursor}/{words.size}", end="\r")
            
            if cursor > 0 :
                if best_len < 64 and level >= 2:
                    # back-ref
                    br_dist, br_len = backref_compress_length(words, cursor, 
                                                                clear_count)
                    if br_len > best_len + 1 and not (best_len > 0 and
                            best_len * 2 + 1 >= br_len and
                            0x1000 - clear_count + best_len >= br_dist) :
                        best_len = br_len
                        best_type = 2 # BACKREF

                if best_len == 0 and level >= 2:
                    rb_index = ring_buf.word_position(current_word, 2)
                    if rb_index >= 0 :
                        best_len = rb_len = 1
                        best_type = 3 # RINGBUF
                        rb_index //= 2 # convert index to 2-bytes word

            if best_type == 0 :
                # best is literal
                match lit_len :
                    case 0 :
                        lit_start = cursor
                        lit_len = 1
                    case 63 :
                        literal_compress(output, words, lit_start, 64, invert)
                        if clear_count <= 0:
                            clear_count = 0x1000
                        lit_len = 0
                    case _ :
                        lit_len += 1
                if level >= 2 :
                    ring_buf.append(current_word)
                cursor += 1
                clear_count -= 1
                if clear_count <= 0 and lit_len > 0 :
                    clear_count = 0x1000
                    literal_compress(output, words, lit_start,
                                     lit_len, invert)
                    lit_len = 0
            else :
                if lit_len > 0 :
                    literal_compress(output, words, lit_start,
                                     lit_len, invert)
                    if clear_count <= 0:
                        clear_count = 0x1000
                    lit_len = 0
                if best_type == 1 : # RLE
                    cmd = (MzxCmd.RLE | ((rle_len - 1) << 2))
                    output.write(cmd.to_bytes(1))
                    clear_count -= rle_len
                elif best_type == 2 : # BACKREF
                    cmd = (MzxCmd.BACKREF | (br_len - 1) << 2)
                    output.write(cmd.to_bytes(1))
                    output.write(int(br_dist-1).to_bytes(1))
                    clear_count -= br_len
                elif best_type == 3 : # RINGBUF
                    cmd = (MzxCmd.RINGBUF | (rb_index << 2))
                    output.write(cmd.to_bytes(1))
                    clear_count -= rb_len
                else :
                    assert False, "Should never reach this point"
                if clear_count <= 0:
                    clear_count = 0x1000
                cursor += best_len
        
        if lit_len > 0 :
            literal_compress(output, words, lit_start,
                             lit_len, invert)

    src.seek(start)
    if dest is None :
        output.seek(0)
        return output

@overload
def mzx_decompress(src: BytesReader | str, dest: BytesRW | str,
                   invert_bytes: bool = False) -> None : ...

@overload
def mzx_decompress(src: Union[BytesReader, str],
                   dest: None = None,
                   invert_bytes: bool = False) -> BytesIO : ...

def mzx_decompress(src: BytesReader | str,
                   dest: BytesRW | str | None = None,
                   invert_bytes: bool = False) :
    input_file = open(src, 'rb') if isinstance(src, str) else src
    match dest :
        case str() : output_file = open(dest, "wb+")
        case None : output_file = BytesIO()
        case _ : output_file = dest; output_file.seek(0)
    start = input_file.tell()
    end = input_file.seek(0, 2)
    input_file.seek(start)

    len_header = len(MZX_FILE_MAGIC) + 4
    header = input_file.read(len_header)
    assert len(header) == len_header
    assert header[:len(MZX_FILE_MAGIC)] == MZX_FILE_MAGIC
    
    decompressed_size = int.from_bytes(header[-4:], 'little', signed = False)

    output_file.truncate(decompressed_size)

    filler_2bytes = b'\xFF\xFF' if invert_bytes else b'\0\0'

    ring_buf = RingBuffer(128, 0xFF if invert_bytes else 0)

    clear_count = 0

    while output_file.tell() < decompressed_size and input_file.tell() < end:
        # Get type / arg from next byte in input
        flags = input_file.read(1)[0]
        cmd = flags & 0x03
        arg = flags >> 2

        if clear_count <= 0:
            clear_count = 0x1000

        match cmd :
            case MzxCmd.RLE :
                # Repeat last two bytes arg + 1 times
                if clear_count == 0x1000 :
                    last = filler_2bytes
                else :
                    output_file.seek(-2, 1)
                    last = output_file.read(2)
                output_file.write(last * (arg + 1))

            case MzxCmd.BACKREF :
                # Next byte from input is lookback distance - 1, in 2-byte words
                pos = output_file.tell()
                k = 2 * (input_file.read(1)[0] + 1)
                length = 2 * (arg + 1)
                output_file.seek(pos-k)
                if k < length :
                    buffer = (output_file.read(k) * ceil(length/k))[:length]
                else :
                    buffer = output_file.read(length)

                output_file.seek(pos)
                output_file.write(buffer)

            case MzxCmd.RINGBUF : 
                # Write ring buffer data at position arg
                output_file.write(ring_buf.get(arg*2, 2))
            
            case _ : # 3: LITERAL
                buffer = input_file.read((arg+1)*2)
                if invert_bytes :
                    buffer = bytes([b ^ 0xFF for b in buffer])
                output_file.write(buffer)
                ring_buf.append(buffer)
        
        clear_count -= 1 if cmd == MzxCmd.RINGBUF else arg + 1

    output_file.truncate(decompressed_size)  # Resize stream to decompress size
    output_file.seek(0)
    input_file.seek(start)

    if isinstance(src, str) : cast(BufferedReader, input_file).close()
    if isinstance(dest, str) : cast(BufferedRandom, output_file).close()
    elif dest is None : return output_file