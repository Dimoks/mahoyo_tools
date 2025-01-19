
from io import BufferedWriter, BytesIO
from PIL import Image
from math import ceil
import struct
from typing import Any, cast, overload, TYPE_CHECKING
import numpy as np

from .utils.io import BytesReader, BytesWriter, BytesRW
from .utils.huffman import ByteHuffmanTable
from .utils.bitstream import BitStreamReader, BitStreamWriter

__all__ = [
    "CompressedBG"
]

CBG_MAGIC = b'CompressedBG_MT\0'

def read_variable(src: BytesReader) :
    value = 0
    shift = 0
    byte = 0xFF
    while byte & 0x80 != 0 :
        byte = src.read(1)[0]
        value |= (byte & 0x7F) << shift
        shift += 7
    return value

def write_variable(dest: BytesWriter, value: int) :
    
    while value > 0x7F :
        byte = (value & 0x7F) | 0x80
        dest.write(byte.to_bytes(1, 'little'))
        value >>= 7
    dest.write(value.to_bytes(1, 'little'))
    

class CompressedBG :
    
    def __init__(self, src: str | bytes | BytesRW) -> None:
        match src :
            case str() : self._file = open(src, "rb+")
            case bytes() : self._file = BytesIO(src)
            case _ : self._file = cast(BytesRW, src)
        
        assert self._file.read(len(CBG_MAGIC)) == CBG_MAGIC
        
        width, height, stripe_h, bpp = struct.unpack('<4I', self._file.read(4*4))
        self._width = width
        self._height = height
        self._stripe_h = stripe_h
        self._bpp = bpp
        self._nb_stripes = ceil(height / stripe_h)
        
        # skip empty bytes, check if there is unexpected data
        assert self._file.read(4*4) == b'\0'*16

        # get list of section offsets
        offsets = np.frombuffer(self._file.read(self._nb_stripes*4), dtype="<u4")
        sizes = np.diff(offsets, append=self._file.seek(0, 2))
        self._stripes: list[tuple[int, int]] = list(map(tuple, np.column_stack((offsets, sizes))))
    
    def _stripe_height(self, index: int) :
        if index < self._nb_stripes -1 :
            return self._stripe_h
        else :
            h = self._height % self._stripe_h
            if h == 0 :
                return self._stripe_h
            else :
                return h
    
    def _decompress_stripe(self, index: int) :
        assert index < self._nb_stripes
        
        offset, size = self._stripes[index]
        width = self._width
        height = self._stripe_height(index)
        self._file.seek(offset)

        # 1. Huffman decompression
        # 1.1. read decompressed size
        huffOutputSize = int.from_bytes(self._file.read(4), 'little')
        # 1.2. extract Huffman table
        huffTable = ByteHuffmanTable()
        for i in range(0, 256) :
            huffTable.getNode(i).weight = read_variable(self._file)
        huffTable.buildTree(511)

        # 1.3. decompress content
        bitStream = BitStreamReader(self._file)
        huffOutput = BytesIO()
        huffOutput.truncate(huffOutputSize)
        huffOutput.seek(0)
        for _ in range(0, huffOutputSize) :
            value = huffTable.decodeSequence(bitStream)
            huffOutput.write(value.to_bytes(1))
        huffOutput.seek(0)
        assert self._file.tell() == offset + size, \
            f"{index}: {self._file.tell()} != {offset} + {size}"

        # 2. Alternate between filling 0 and copying data from huffman output
        output = BytesIO()
        zeros = False
        while huffOutput.tell() < huffOutputSize :
            val = read_variable(huffOutput)
            if zeros :
                output.write(b'\0' * val)
            else :
                output.write(huffOutput.read(val))
            zeros = not zeros
        output.seek(0)
        
        # 3. resolve the diff: px[x,y] = px[x,y] + avg(px[x-1,y] + px[x,y-1]) (approx.)
        pixels = np.frombuffer(output.getbuffer(), dtype=np.uint8)
        pixels.shape = (height, width, self._bpp >> 3)
        for y in range(1, height) : pixels[y, 0, :] += pixels[y-1, 0, :]
        for x in range(1, width ) : pixels[0, x, :] += pixels[0, x-1, :]
        for y in range(1, height) :
            for x in range(1, width) :
                pixels[y,x,:] += ((
                    pixels[y-1,x,:].astype(np.uint16) +
                    pixels[y,x-1,:].astype(np.uint16)
                ) >> 1).astype(np.uint8)
                
        # 4. switch BGR to RGB encoding
        match self._bpp :
            case  8 : pass # grayscale image
            case 24 : pixels[:, :, :]  = pixels[:, :, ::-1] # BGR -> RGB
            case 32 : pixels[:, :, :3] = pixels[:, :, 2::-1] # BGRA -> RGBA
        # TODO check if output file is updated
        return output
    
    def _compress_stripe(self, buffer: np.ndarray[Any, np.dtype[np.uint8]], index: int) :
        width = self._width
        height = self._stripe_height(index)
        row_start = index * self._stripe_h
        row_end = row_start + height
        pixels = buffer[row_start:row_end, :].copy()
        
        # 1. RGB -> BGR
        match self._bpp :
            case  8 : pass # grayscale image
            case 24 : pixels[:, :, :] = pixels[:, :, ::-1] # RGB -> BGR
            case 32 : pixels[:, :, :3] = pixels[:, :, 2::-1] # RGBA -> BGRA

        # 2. create diff
        for y in reversed(range(1, height)) :
            for x in reversed(range(1, width)) :
                pixels[y,x] -= ((
                    pixels[y-1,x].astype(np.uint16) +
                    pixels[y,x-1].astype(np.uint16)
                ) >> 1).astype(np.uint8)
        for y in reversed(range(1, height)) : pixels[y, 0] -= pixels[y-1, 0]
        for x in reversed(range(1, width )) : pixels[0, x] -= pixels[0, x-1]
        
        # 3. compression 1 : compress to [n1, n1*\x00, n2, read(n2), ...]
        comp1 = BytesIO()
        pixels = pixels.flatten()
        cursor = 0
        if pixels[cursor] == 0 :
            write_variable(comp1, 0)
        while cursor < len(pixels) :
            i = cursor
            if pixels[i] != 0 :
                while i < len(pixels) and pixels[i] > 0 :
                    i += 1
                write_variable(comp1, i - cursor)
                comp1.write(pixels[cursor:i].tobytes())
            else :
                while i < len(pixels) and pixels[i] == 0 :
                    i += 1
                write_variable(comp1, i - cursor)
            cursor = i
        
        output = BytesIO()
        decomp_size = comp1.tell()
        output.write(decomp_size.to_bytes(4, 'little'))
        comp1.seek(0)

        huffTable = ByteHuffmanTable()
        while byte := comp1.read(1) :
            huffTable.getNode(byte[0]).weight += 1
        comp1.seek(0)
        for node in huffTable :
            write_variable(output, node.weight)

        huffTable.buildTree(511)
        
        bitStream = BitStreamWriter(output)
        while comp1.tell() < decomp_size :
            huffTable.encodeValue(bitStream, comp1.read(1)[0])
        bitStream.flush()
        output.seek(0)
        return output

    @overload
    def img_write(self, dest: str) -> None : ...
    @overload
    def img_write(self, dest: BytesWriter, format: str = "png") -> None : ...
    @overload
    def img_write(self, dest: None = None, format: str = "png") -> BytesIO : ...
    def img_write(self, dest: str | BytesWriter | None = None,
                  format: str = "png") :

        raw_pixels_file = BytesIO()
        for index in range(0, self._nb_stripes) :
            stripe_out = self._decompress_stripe(index)
            raw_pixels_file.write(stripe_out.getbuffer())
        
        match self._bpp :
            case  8 : mode = "L"
            case 24 : mode = "RGB"
            case 32 : mode = "RGBA"
        image = Image.frombytes(mode, (self._width, self._height), raw_pixels_file.getbuffer())

        if dest is None :
            dest = BytesIO()
            image.save(dest, format)
            return dest
        else :
            image.save(dest, format)
    
    def img_read(self, src: str | BytesReader | bytes | Image.Image) :
        if not isinstance(src, Image.Image) :
            src = Image.open(src)
        assert src.width == self._width
        assert src.height == self._height
        match self._bpp :
            case  8 : src = src.convert("L")
            case 24 : src = src.convert("RGB")
            case 32 : src = src.convert("RGBA")
        pixels = np.array(src)

        offsets = list(map(lambda os: os[0], self._stripes))
        for i, offset in enumerate(offsets) :
            data = self._compress_stripe(pixels, i)
            size = data.seek(0,2)
            data.seek(0)
            self._file.seek(offset)
            self._file.write(data.getbuffer())
            self._stripes[i] = (offset, size)
            if i < len(offsets)-1 :
                offsets[i+1] = offset + size
            else :
                total_size = offset + size
        self._file.seek(48)
        self._file.write(struct.pack(f"<{len(offsets)}I", *tuple(offsets)))
        self._file.truncate(total_size)

    @overload
    def cbg_write(self, dest: str | BytesWriter) -> None : ...
    @overload
    def cbg_write(self, dest: None = None) -> BytesIO : ...
    def cbg_write(self, dest: str | BytesWriter | None = None) :
        self._file.seek(0)
        match dest :
            case None :
                return BytesIO(self._file.read())
            case str() :
                output = open(dest, "wb")
                output.write(self._file.read())
                output.close()
            case _ :
                dest.write(self._file.read())
