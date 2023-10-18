from __future__ import print_function

import itertools
import math
import re
import operator
import struct
import sys
import warnings
import zlib
from array import array
from functools import reduce

try:
    import cpngfilters as pngfilters
except ImportError:
    pass
__version__="0.1"

__all__ = ['Image', 'Reader', 'Writer', 'write_chunks', 'from_array']

_signature = struct.pack('8B', 137, 80, 78, 71, 13, 10, 26, 10)

_adam7 = ((0, 0, 8, 8),
          (4, 0, 8, 8),
          (0, 4, 4, 8),
          (2, 0, 4, 4),
          (0, 2, 2, 4),
          (1, 0, 2, 2),
          (0, 1, 1, 2))

def group(s, n):
    return list(zip(*[iter(s)]*n))

def isarray(x):
    return isinstance(x, array)

def tostring(row):
    return row.tobytes()

def interleave_planes(ipixels, apixels, ipsize, apsize):
    itotal = len(ipixels)
    atotal = len(apixels)
    newtotal = itotal + atotal
    newpsize = ipsize + apsize
    out = array(ipixels.typecode)
    out.extend(ipixels)
    out.extend(apixels)
    for i in range(ipsize):
        out[i:newtotal:newpsize] = ipixels[i:itotal:ipsize]
    for i in range(apsize):
        out[i+ipsize:newtotal:newpsize] = apixels[i:atotal:apsize]
    return out

def check_palette(palette):
    if palette is None:
        return None

    p = list(palette)
    if not (0 < len(p) <= 256):
        raise ValueError("a palette must have between 1 and 256 entries")
    seen_triple = False
    for i,t in enumerate(p):
        if len(t) not in (3,4):
            raise ValueError(
              "palette entry %d: entries must be 3- or 4-tuples." % i)
        if len(t) == 3:
            seen_triple = True
        if seen_triple and len(t) == 4:
            raise ValueError(
              "palette entry %d: all 4-tuples must precede all 3-tuples" % i)
        for x in t:
            if int(x) != x or not(0 <= x <= 255):
                raise ValueError(
                  "palette entry %d: values must be integer: 0 <= x <= 255" % i)
    return p

def check_sizes(size, width, height):
    if not size:
        return width, height

    if len(size) != 2:
        raise ValueError(
          "size argument should be a pair (width, height)")
    if width is not None and width != size[0]:
        raise ValueError(
          "size[0] (%r) and width (%r) should match when both are used."
            % (size[0], width))
    if height is not None and height != size[1]:
        raise ValueError(
          "size[1] (%r) and height (%r) should match when both are used."
            % (size[1], height))
    return size

def check_color(c, greyscale, which):
    if c is None:
        return c
    if greyscale:
        try:
            len(c)
        except TypeError:
            c = (c,)
        if len(c) != 1:
            raise ValueError("%s for greyscale must be 1-tuple" %
                which)
        if not isinteger(c[0]):
            raise ValueError(
                "%s colour for greyscale must be integer" % which)
    else:
        if not (len(c) == 3 and
                isinteger(c[0]) and
                isinteger(c[1]) and
                isinteger(c[2])):
            raise ValueError(
                "%s colour must be a triple of integers" % which)
    return c

class Error(Exception):
    def __str__(self):
        return self.__class__.__name__ + ': ' + ' '.join(self.args)

class FormatError(Error):
    """Problem
    """

class ChunkError(FormatError):
    pass


class Writer:
    def __init__(self, width=None, height=None,
                 size=None,
                 greyscale=False,
                 alpha=False,
                 bitdepth=8,
                 palette=None,
                 transparent=None,
                 background=None,
                 gamma=None,
                 compression=None,
                 interlace=False,
                 bytes_per_sample=None, 
                 planes=None,
                 colormap=None,
                 maxval=None,
                 chunk_limit=2**20,
                 x_pixels_per_unit = None,
                 y_pixels_per_unit = None,
                 unit_is_meter = False):

        width, height = check_sizes(size, width, height)
        del size

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be greater than zero")
        if not isinteger(width) or not isinteger(height):
            raise ValueError("width and height must be integers")
        if width > 2**32-1 or height > 2**32-1:
            raise ValueError("width and height cannot exceed 2**32-1")

        if alpha and transparent is not None:
            raise ValueError(
                "transparent colour not allowed with alpha channel")

        if bytes_per_sample is not None:
            warnings.warn('please use bitdepth instead of bytes_per_sample',
                          DeprecationWarning)
            if bytes_per_sample not in (0.125, 0.25, 0.5, 1, 2):
                raise ValueError(
                    "bytes per sample must be .125, .25, .5, 1, or 2")
            bitdepth = int(8*bytes_per_sample)
        del bytes_per_sample
        if not isinteger(bitdepth) or bitdepth < 1 or 16 < bitdepth:
            raise ValueError("bitdepth (%r) must be a positive integer <= 16" %
              bitdepth)

        self.rescale = None
        palette = check_palette(palette)
        if palette:
            if bitdepth not in (1,2,4,8):
                raise ValueError("with palette, bitdepth must be 1, 2, 4, or 8")
            if transparent is not None:
                raise ValueError("transparent and palette not compatible")
            if alpha:
                raise ValueError("alpha and palette not compatible")
            if greyscale:
                raise ValueError("greyscale and palette not compatible")
        else:
            if alpha or not greyscale:
                if bitdepth not in (8,16):
                    targetbitdepth = (8,16)[bitdepth > 8]
                    self.rescale = (bitdepth, targetbitdepth)
                    bitdepth = targetbitdepth
                    del targetbitdepth
            else:
                assert greyscale
                assert not alpha
                if bitdepth not in (1,2,4,8,16):
                    if bitdepth > 8:
                        targetbitdepth = 16
                    elif bitdepth == 3:
                        targetbitdepth = 4
                    else:
                        assert bitdepth in (5,6,7)
                        targetbitdepth = 8
                    self.rescale = (bitdepth, targetbitdepth)
                    bitdepth = targetbitdepth
                    del targetbitdepth

        if bitdepth < 8 and (alpha or not greyscale and not palette):
            raise ValueError(
              "bitdepth < 8 only permitted with greyscale or palette")
        if bitdepth > 8 and palette:
            raise ValueError(
                "bit depth must be 8 or less for images with palette")

        transparent = check_color(transparent, greyscale, 'transparent')
        background = check_color(background, greyscale, 'background')
        self.width = width
        self.height = height
        self.transparent = transparent
        self.background = background
        self.gamma = gamma
        self.greyscale = bool(greyscale)
        self.alpha = bool(alpha)
        self.colormap = bool(palette)
        self.bitdepth = int(bitdepth)
        self.compression = compression
        self.chunk_limit = chunk_limit
        self.interlace = bool(interlace)
        self.palette = palette
        self.x_pixels_per_unit = x_pixels_per_unit
        self.y_pixels_per_unit = y_pixels_per_unit
        self.unit_is_meter = bool(unit_is_meter)

        self.color_type = 4*self.alpha + 2*(not greyscale) + 1*self.colormap
        assert self.color_type in (0,2,3,4,6)

        self.color_planes = (3,1)[self.greyscale or self.colormap]
        self.planes = self.color_planes + self.alpha
        self.psize = (self.bitdepth/8) * self.planes

    def make_palette(self):
        p = array('B')
        t = array('B')

        for x in self.palette:
            p.extend(x[0:3])
            if len(x) > 3:
                t.append(x[3])
        p = tostring(p)
        t = tostring(t)
        if t:
            return p,t
        return p,None

    def write(self, outfile, rows):
        if self.interlace:
            fmt = 'BH'[self.bitdepth > 8]
            a = array(fmt, itertools.chain(*rows))
            return self.write_array(outfile, a)

        nrows = self.write_passes(outfile, rows)
        if nrows != self.height:
            raise ValueError(
              "rows supplied (%d) does not match height (%d)" %
              (nrows, self.height))

    def write_passes(self, outfile, rows, packed=False):
        outfile.write(_signature)
        write_chunk(outfile, b'IHDR',
                    struct.pack("!2I5B", self.width, self.height,
                                self.bitdepth, self.color_type,
                                0, 0, self.interlace))
        if self.gamma is not None:
            write_chunk(outfile, b'gAMA',
                        struct.pack("!L", int(round(self.gamma*1e5))))
        if self.rescale:
            write_chunk(outfile, b'sBIT',
                struct.pack('%dB' % self.planes,
                            *[self.rescale[0]]*self.planes))
        if self.palette:
            p,t = self.make_palette()
            write_chunk(outfile, b'PLTE', p)
            if t:
                write_chunk(outfile, b'tRNS', t)
        if self.transparent is not None:
            if self.greyscale:
                write_chunk(outfile, b'tRNS',
                            struct.pack("!1H", *self.transparent))
            else:
                write_chunk(outfile, b'tRNS',
                            struct.pack("!3H", *self.transparent))
        if self.background is not None:
            if self.greyscale:
                write_chunk(outfile, b'bKGD',
                            struct.pack("!1H", *self.background))
            else:
                write_chunk(outfile, b'bKGD',
                            struct.pack("!3H", *self.background))
        if self.x_pixels_per_unit is not None and self.y_pixels_per_unit is not None:
            tup = (self.x_pixels_per_unit, self.y_pixels_per_unit, int(self.unit_is_meter))
            write_chunk(outfile, b'pHYs', struct.pack("!LLB",*tup))
        if self.compression is not None:
            compressor = zlib.compressobj(self.compression)
        else:
            compressor = zlib.compressobj()
        data = array('B')
        if self.bitdepth == 8 or packed:
            extend = data.extend
        elif self.bitdepth == 16:
            def extend(sl):
                fmt = '!%dH' % len(sl)
                data.extend(array('B', struct.pack(fmt, *sl)))
        else:
            assert self.bitdepth < 8
            spb = int(8/self.bitdepth)
            def extend(sl):
                a = array('B', sl)
                l = float(len(a))
                extra = math.ceil(l / float(spb))*spb - l
                a.extend([0]*int(extra))
                l = group(a, spb)
                l = [reduce(lambda x,y:
                                           (x << self.bitdepth) + y, e) for e in l]
                data.extend(l)
        if self.rescale:
            oldextend = extend
            factor = \
              float(2**self.rescale[1]-1) / float(2**self.rescale[0]-1)
            def extend(sl):
                oldextend([int(round(factor*x)) for x in sl])
        enumrows = enumerate(rows)
        del rows
        data.append(0)
        i,row = next(enumrows)
        try:
            extend(row)
        except:
            def wrapmapint(f):
                return lambda sl: f([int(x) for x in sl])
            extend = wrapmapint(extend)
            del wrapmapint
            extend(row)

        for i,row in enumrows:
            data.append(0)
            extend(row)
            if len(data) > self.chunk_limit:
                compressed = compressor.compress(tostring(data))
                if len(compressed):
                    write_chunk(outfile, b'IDAT', compressed)
                del data[:]
        if len(data):
            compressed = compressor.compress(tostring(data))
        else:
            compressed = b''
        flushed = compressor.flush()
        if len(compressed) or len(flushed):
            write_chunk(outfile, b'IDAT', compressed + flushed)
        write_chunk(outfile, b'IEND')
        return i+1

    def write_array(self, outfile, pixels):

        if self.interlace:
            self.write_passes(outfile, self.array_scanlines_interlace(pixels))
        else:
            self.write_passes(outfile, self.array_scanlines(pixels))

    def write_packed(self, outfile, rows):

        if self.rescale:
            raise Error("write_packed method not suitable for bit depth %d" %
              self.rescale[0])
        return self.write_passes(outfile, rows, packed=True)

    def convert_pnm(self, infile, outfile):

        if self.interlace:
            pixels = array('B')
            pixels.fromfile(infile,
                            (self.bitdepth/8) * self.color_planes *
                            self.width * self.height)
            self.write_passes(outfile, self.array_scanlines_interlace(pixels))
        else:
            self.write_passes(outfile, self.file_scanlines(infile))

    def convert_ppm_and_pgm(self, ppmfile, pgmfile, outfile):
        pixels = array('B')
        pixels.fromfile(ppmfile,
                        (self.bitdepth/8) * self.color_planes *
                        self.width * self.height)
        apixels = array('B')
        apixels.fromfile(pgmfile,
                         (self.bitdepth/8) *
                         self.width * self.height)
        pixels = interleave_planes(pixels, apixels,
                                   (self.bitdepth/8) * self.color_planes,
                                   (self.bitdepth/8))
        if self.interlace:
            self.write_passes(outfile, self.array_scanlines_interlace(pixels))
        else:
            self.write_passes(outfile, self.array_scanlines(pixels))

    def file_scanlines(self, infile):
        vpr = self.width * self.planes
        row_bytes = vpr
        if self.bitdepth > 8:
            assert self.bitdepth == 16
            row_bytes *= 2
            fmt = '>%dH' % vpr
            def line():
                return array('H', struct.unpack(fmt, infile.read(row_bytes)))
        else:
            def line():
                scanline = array('B', infile.read(row_bytes))
                return scanline
        for y in range(self.height):
            yield line()

    def array_scanlines(self, pixels):
        vpr = self.width * self.planes
        stop = 0
        for y in range(self.height):
            start = stop
            stop = start + vpr
            yield pixels[start:stop]

    def array_scanlines_interlace(self, pixels):
        fmt = 'BH'[self.bitdepth > 8]
        vpr = self.width * self.planes
        for xstart, ystart, xstep, ystep in _adam7:
            if xstart >= self.width:
                continue
            ppr = int(math.ceil((self.width-xstart)/float(xstep)))
            row_len = ppr*self.planes
            for y in range(ystart, self.height, ystep):
                if xstep == 1:
                    offset = y * vpr
                    yield pixels[offset:offset+vpr]
                else:
                    row = array(fmt)
                    row.extend(pixels[0:row_len])
                    offset = y * vpr + xstart * self.planes
                    end_offset = (y+1) * vpr
                    skip = self.planes * xstep
                    for i in range(self.planes):
                        row[i::self.planes] = \
                            pixels[offset+i:end_offset:skip]
                    yield row

def write_chunk(outfile, tag, data=b''):
    outfile.write(struct.pack("!I", len(data)))
    outfile.write(tag)
    outfile.write(data)
    checksum = zlib.crc32(tag)
    checksum = zlib.crc32(data, checksum)
    checksum &= 2**32-1
    outfile.write(struct.pack("!I", checksum))

def write_chunks(out, chunks):

    out.write(_signature)
    for chunk in chunks:
        write_chunk(out, *chunk)

def filter_scanline(type, line, fo, prev=None):
    assert 0 <= type < 5
    out = array('B', [type])

    def sub():
        ai = -fo
        for x in line:
            if ai >= 0:
                x = (x - line[ai]) & 0xff
            out.append(x)
            ai += 1
    def up():
        for i,x in enumerate(line):
            x = (x - prev[i]) & 0xff
            out.append(x)
    def average():
        ai = -fo
        for i,x in enumerate(line):
            if ai >= 0:
                x = (x - ((line[ai] + prev[i]) >> 1)) & 0xff
            else:
                x = (x - (prev[i] >> 1)) & 0xff
            out.append(x)
            ai += 1
    def paeth():
        ai = -fo 
        for i,x in enumerate(line):
            a = 0
            b = prev[i]
            c = 0

            if ai >= 0:
                a = line[ai]
                c = prev[ai]
            p = a + b - c
            pa = abs(p - a)
            pb = abs(p - b)
            pc = abs(p - c)
            if pa <= pb and pa <= pc:
                Pr = a
            elif pb <= pc:
                Pr = b
            else:
                Pr = c

            x = (x - Pr) & 0xff
            out.append(x)
            ai += 1

    if not prev:
        if type == 2: 
            type = 0
        elif type == 3:
            prev = [0]*len(line)
        elif type == 4: 
            type = 1
    if type == 0:
        out.extend(line)
    elif type == 1:
        sub()
    elif type == 2:
        up()
    elif type == 3:
        average()
    else: 
        paeth()
    return out

RegexModeDecode = re.compile("(LA?|RGBA?);?([0-9]*)", flags=re.IGNORECASE)

def from_array(a, mode=None, info={}):
    info = dict(info)
    match = RegexModeDecode.match(mode)
    if not match:
        raise Error("mode string should be 'RGB' or 'L;16' or similar.")

    mode, bitdepth = match.groups()
    alpha = 'A' in mode
    if bitdepth:
        bitdepth = int(bitdepth)
    if 'greyscale' in info:
        if bool(info['greyscale']) != ('L' in mode):
            raise Error("info['greyscale'] should match mode.")
    info['greyscale'] = 'L' in mode

    if 'alpha' in info:
        if bool(info['alpha']) != alpha:
            raise Error("info['alpha'] should match mode.")
    info['alpha'] = alpha
    if bitdepth:
        if info.get("bitdepth") and bitdepth != info['bitdepth']:
            raise Error("bitdepth (%d) should match bitdepth of info (%d)." %
              (bitdepth, info['bitdepth']))
        info['bitdepth'] = bitdepth
    if 'size' in info:
        assert len(info["size"]) == 2
        for dimension,axis in [('width', 0), ('height', 1)]:
            if dimension in info:
                if info[dimension] != info['size'][axis]:
                    raise Error(
                      "info[%r] should match info['size'][%r]." %
                      (dimension, axis))
        info['width'],info['height'] = info['size']

    if 'height' not in info:
        try:
            info['height'] = len(a)
        except TypeError:
            raise Error("len(a) does not work, supply info['height'] instead.")

    planes = len(mode)
    if 'planes' in info:
        if info['planes'] != planes:
            raise Error("info['planes'] should match mode.")
    a,t = itertools.tee(a)
    row = next(t)
    del t
    try:
        row[0][0]
        threed = True
        testelement = row[0]
    except (IndexError, TypeError):
        threed = False
        testelement = row
    if 'width' not in info:
        if threed:
            width = len(row)
        else:
            width = len(row) // planes
        info['width'] = width

    if threed:
        a = (itertools.chain.from_iterable(x) for x in a)

    if 'bitdepth' not in info:
        try:
            dtype = testelement.dtype
        except AttributeError:
            try:
                bitdepth = 8 * testelement.itemsize
            except AttributeError:
                bitdepth = 8
        else:
            if dtype.kind == 'b':
                bitdepth = 1
            else:
                bitdepth = 8 * dtype.itemsize
        info['bitdepth'] = bitdepth

    for thing in ["width", "height", "bitdepth", "greyscale", "alpha"]:
        assert thing in info

    return Image(a, info)

fromarray = from_array

class Image:
    def __init__(self, rows, info):  
        self.rows = rows
        self.info = info

    def save(self, file):

        w = Writer(**self.info)

        try:
            file.write
            def close(): pass
        except AttributeError:
            file = open(file, 'wb')
            def close(): file.close()

        try:
            w.write(file, self.rows)
        finally:
            close()

class _readable:
    def __init__(self, buf):
        self.buf = buf
        self.offset = 0

    def read(self, n):
        r = self.buf[self.offset:self.offset+n]
        if isarray(r):
            r = r.tostring()
        self.offset += n
        return r

try:
    str(b'dummy', 'ascii')
except TypeError:
    as_str = str
else:
    def as_str(x):
        return str(x, 'ascii')

class Reader:
    def __init__(self, _guess=None, **kw):
        if ((_guess is not None and len(kw) != 0) or
            (_guess is None and len(kw) != 1)):
            raise TypeError("Reader() takes exactly 1 argument")
        self.signature = None
        self.transparent = None
        self.atchunk = None

        if _guess is not None:
            if isarray(_guess):
                kw["bytes"] = _guess
            elif isinstance(_guess, str):
                kw["filename"] = _guess
            elif hasattr(_guess, 'read'):
                kw["file"] = _guess

        if "filename" in kw:
            self.file = open(kw["filename"], "rb")
        elif "file" in kw:
            self.file = kw["file"]
        elif "bytes" in kw:
            self.file = _readable(kw["bytes"])
        else:
            raise TypeError("expecting filename, file or bytes array")


    def chunk(self, seek=None, lenient=False):

        self.validate_signature()

        while True:
            if not self.atchunk:
                self.atchunk = self.chunklentype()
            length, type = self.atchunk
            self.atchunk = None
            data = self.file.read(length)
            if len(data) != length:
                raise ChunkError('Chunk %s too short for required %i octets.'
                  % (type, length))
            checksum = self.file.read(4)
            if len(checksum) != 4:
                raise ChunkError('Chunk %s too short for checksum.' % type)
            if seek and type != seek:
                continue
            verify = zlib.crc32(type)
            verify = zlib.crc32(data, verify)
            verify &= 2**32 - 1
            verify = struct.pack('!I', verify)
            if checksum != verify:
                (a, ) = struct.unpack('!I', checksum)
                (b, ) = struct.unpack('!I', verify)
                message = "Checksum error in %s chunk: 0x%08X != 0x%08X." % (type, a, b)
                if lenient:
                    warnings.warn(message, RuntimeWarning)
                else:
                    raise ChunkError(message)
            return type, data

    def chunks(self):

        while True:
            t,v = self.chunk()
            yield t,v
            if t == b'IEND':
                break

    def undo_filter(self, filter_type, scanline, previous):
        result = scanline

        if filter_type == 0:
            return result

        if filter_type not in (1,2,3,4):
            raise FormatError('Invalid PNG Filter Type.'
              '  See http://www.w3.org/TR/2003/REC-PNG-20031110/#9Filters .')
        fu = max(1, self.psize)
        if not previous:
            previous = array('B', [0]*len(scanline))

        def sub():

            ai = 0
            for i in range(fu, len(result)):
                x = scanline[i]
                a = result[ai]
                result[i] = (x + a) & 0xff
                ai += 1

        def up():

            for i in range(len(result)):
                x = scanline[i]
                b = previous[i]
                result[i] = (x + b) & 0xff

        def average():
            ai = -fu
            for i in range(len(result)):
                x = scanline[i]
                if ai < 0:
                    a = 0
                else:
                    a = result[ai]
                b = previous[i]
                result[i] = (x + ((a + b) >> 1)) & 0xff
                ai += 1

        def paeth():
            ai = -fu
            for i in range(len(result)):
                x = scanline[i]
                if ai < 0:
                    a = c = 0
                else:
                    a = result[ai]
                    c = previous[ai]
                b = previous[i]
                p = a + b - c
                pa = abs(p - a)
                pb = abs(p - b)
                pc = abs(p - c)
                if pa <= pb and pa <= pc:
                    pr = a
                elif pb <= pc:
                    pr = b
                else:
                    pr = c
                result[i] = (x + pr) & 0xff
                ai += 1

        (None,
         pngfilters.undo_filter_sub,
         pngfilters.undo_filter_up,
         pngfilters.undo_filter_average,
         pngfilters.undo_filter_paeth)[filter_type](fu, scanline, previous, result)
        return result

    def deinterlace(self, raw):

        vpr = self.width * self.planes
        fmt = 'BH'[self.bitdepth > 8]
        a = array(fmt, [0]*vpr*self.height)
        source_offset = 0

        for xstart, ystart, xstep, ystep in _adam7:
            if xstart >= self.width:
                continue
            recon = None
            ppr = int(math.ceil((self.width-xstart)/float(xstep)))
            row_size = int(math.ceil(self.psize * ppr))
            for y in range(ystart, self.height, ystep):
                filter_type = raw[source_offset]
                source_offset += 1
                scanline = raw[source_offset:source_offset+row_size]
                source_offset += row_size
                recon = self.undo_filter(filter_type, scanline, recon)
                flat = self.serialtoflat(recon, ppr)
                if xstep == 1:
                    assert xstart == 0
                    offset = y * vpr
                    a[offset:offset+vpr] = flat
                else:
                    offset = y * vpr + xstart * self.planes
                    end_offset = (y+1) * vpr
                    skip = self.planes * xstep
                    for i in range(self.planes):
                        a[offset+i:end_offset:skip] = \
                            flat[i::self.planes]
        return a

    def iterboxed(self, rows):

        def asvalues(raw):
            if self.bitdepth == 8:
                return array('B', raw)
            if self.bitdepth == 16:
                raw = tostring(raw)
                return array('H', struct.unpack('!%dH' % (len(raw)//2), raw))
            assert self.bitdepth < 8
            width = self.width
            spb = 8//self.bitdepth
            out = array('B')
            mask = 2**self.bitdepth - 1
            shifts = [self.bitdepth * i
                for i in reversed(list(range(spb)))]
            for o in raw:
                out.extend([mask&(o>>i) for i in shifts])
            return out[:width]

        return map(asvalues, rows)

    def serialtoflat(self, bytes, width=None):
        if self.bitdepth == 8:
            return bytes
        if self.bitdepth == 16:
            bytes = tostring(bytes)
            return array('H',
              struct.unpack('!%dH' % (len(bytes)//2), bytes))
        assert self.bitdepth < 8
        if width is None:
            width = self.width
        spb = 8//self.bitdepth
        out = array('B')
        mask = 2**self.bitdepth - 1
        shifts = list(map(self.bitdepth.__mul__, reversed(list(range(spb)))))
        l = width
        for o in bytes:
            out.extend([(mask&(o>>s)) for s in shifts][:l])
            l -= spb
            if l <= 0:
                l = width
        return out

    def iterstraight(self, raw):
        rb = self.row_bytes
        a = array('B')
        recon = None
        for some in raw:
            a.extend(some)
            while len(a) >= rb + 1:
                filter_type = a[0]
                scanline = a[1:rb+1]
                del a[:rb+1]
                recon = self.undo_filter(filter_type, scanline, recon)
                yield recon
        if len(a) != 0:
            raise FormatError(
              'Wrong size for decompressed IDAT chunk.')
        assert len(a) == 0

    def validate_signature(self):
        if self.signature:
            return
        self.signature = self.file.read(8)
        if self.signature != _signature:
            raise FormatError("PNG file has invalid signature.")

    def preamble(self, lenient=False):
        self.validate_signature()

        while True:
            if not self.atchunk:
                self.atchunk = self.chunklentype()
                if self.atchunk is None:
                    raise FormatError(
                      'This PNG file has no IDAT chunks.')
            if self.atchunk[1] == b'IDAT':
                return
            self.process_chunk(lenient=lenient)

    def chunklentype(self):
        x = self.file.read(8)
        if not x:
            return None
        if len(x) != 8:
            raise FormatError(
              'End of file whilst reading chunk length and type.')
        length,type = struct.unpack('!I4s', x)
        if length > 2**31-1:
            raise FormatError('Chunk %s is too large: %d.' % (type,length))
        return length,type

    def process_chunk(self, lenient=False):
        type, data = self.chunk(lenient=lenient)
        method = '_process_' + as_str(type)
        m = getattr(self, method, None)
        if m:
            m(data)

    def _process_IHDR(self, data):
        if len(data) != 13:
            raise FormatError('IHDR chunk has incorrect length.')
        (self.width, self.height, self.bitdepth, self.color_type,
         self.compression, self.filter,
         self.interlace) = struct.unpack("!2I5B", data)

        check_bitdepth_colortype(self.bitdepth, self.color_type)

        if self.compression != 0:
            raise Error("unknown compression method %d" % self.compression)
        if self.filter != 0:
            raise FormatError("Unknown filter method %d,"
              " see http://www.w3.org/TR/2003/REC-PNG-20031110/#9Filters ."
              % self.filter)
        if self.interlace not in (0,1):
            raise FormatError("Unknown interlace method %d,"
              " see http://www.w3.org/TR/2003/REC-PNG-20031110/#8InterlaceMethods ."
              % self.interlace)

        colormap =  bool(self.color_type & 1)
        greyscale = not (self.color_type & 2)
        alpha = bool(self.color_type & 4)
        color_planes = (3,1)[greyscale or colormap]
        planes = color_planes + alpha

        self.colormap = colormap
        self.greyscale = greyscale
        self.alpha = alpha
        self.color_planes = color_planes
        self.planes = planes
        self.psize = float(self.bitdepth)/float(8) * planes
        if int(self.psize) == self.psize:
            self.psize = int(self.psize)
        self.row_bytes = int(math.ceil(self.width * self.psize))
        self.plte = None
        self.trns = None
        self.sbit = None

    def _process_PLTE(self, data):
        if self.plte:
            warnings.warn("Multiple PLTE chunks present.")
        self.plte = data
        if len(data) % 3 != 0:
            raise FormatError(
              "PLTE chunk's length should be a multiple of 3.")
        if len(data) > (2**self.bitdepth)*3:
            raise FormatError("PLTE chunk is too long.")
        if len(data) == 0:
            raise FormatError("Empty PLTE is not allowed.")

    def _process_bKGD(self, data):
        try:
            if self.colormap:
                if not self.plte:
                    warnings.warn(
                      "PLTE chunk is required before bKGD chunk.")
                self.background = struct.unpack('B', data)
            else:
                self.background = struct.unpack("!%dH" % self.color_planes,
                  data)
        except struct.error:
            raise FormatError("bKGD chunk has incorrect length.")

    def _process_tRNS(self, data):
        self.trns = data
        if self.colormap:
            if not self.plte:
                warnings.warn("PLTE chunk is required before tRNS chunk.")
            else:
                if len(data) > len(self.plte)/3:
                    raise FormatError("tRNS chunk is too long.")
        else:
            if self.alpha:
                raise FormatError(
                  "tRNS chunk is not valid with colour type %d." %
                  self.color_type)
            try:
                self.transparent = \
                    struct.unpack("!%dH" % self.color_planes, data)
            except struct.error:
                raise FormatError("tRNS chunk has incorrect length.")

    def _process_gAMA(self, data):
        try:
            self.gamma = struct.unpack("!L", data)[0] / 100000.0
        except struct.error:
            raise FormatError("gAMA chunk has incorrect length.")

    def _process_sBIT(self, data):
        self.sbit = data
        if (self.colormap and len(data) != 3 or
            not self.colormap and len(data) != self.planes):
            raise FormatError("sBIT chunk has incorrect length.")

    def _process_pHYs(self, data):
        self.phys = data
        fmt = "!LLB"
        if len(data) != struct.calcsize(fmt):
            raise FormatError("pHYs chunk has incorrect length.")
        self.x_pixels_per_unit, self.y_pixels_per_unit, unit = struct.unpack(fmt,data)
        self.unit_is_meter = bool(unit)

    def read(self, lenient=False):
        def iteridat():
            while True:
                try:
                    type, data = self.chunk(lenient=lenient)
                except ValueError as e:
                    raise ChunkError(e.args[0])
                if type == b'IEND':

                    break
                if type != b'IDAT':
                    continue
                if self.colormap and not self.plte:
                    warnings.warn("PLTE chunk is required before IDAT chunk")
                yield data

        def iterdecomp(idat):
            d = zlib.decompressobj()
            for data in idat:
                yield array('B', d.decompress(data))
            yield array('B', d.flush())

        self.preamble(lenient=lenient)
        raw = iterdecomp(iteridat())

        if self.interlace:
            raw = array('B', itertools.chain(*raw))
            arraycode = 'BH'[self.bitdepth>8]
            pixels = map(lambda *row: array(arraycode, row),
                       *[iter(self.deinterlace(raw))]*self.width*self.planes)
        else:
            pixels = self.iterboxed(self.iterstraight(raw))
        meta = dict()
        for attr in 'greyscale alpha planes bitdepth interlace'.split():
            meta[attr] = getattr(self, attr)
        meta['size'] = (self.width, self.height)
        for attr in 'gamma transparent background'.split():
            a = getattr(self, attr, None)
            if a is not None:
                meta[attr] = a
        if self.plte:
            meta['palette'] = self.palette()
        return self.width, self.height, pixels, meta


    def read_flat(self):

        x, y, pixel, meta = self.read()
        arraycode = 'BH'[meta['bitdepth']>8]
        pixel = array(arraycode, itertools.chain(*pixel))
        return x, y, pixel, meta

    def palette(self, alpha='natural'):
        if not self.plte:
            raise FormatError(
                "Required PLTE chunk is missing in colour type 3 image.")
        plte = group(array('B', self.plte), 3)
        if self.trns or alpha == 'force':
            trns = array('B', self.trns or [])
            trns.extend([255]*(len(plte)-len(trns)))
            plte = list(map(operator.add, plte, group(trns, 1)))
        return plte

    def asDirect(self):
        self.preamble()

        if not self.colormap and not self.trns and not self.sbit:
            return self.read()

        x,y,pixels,meta = self.read()

        if self.colormap:
            meta['colormap'] = False
            meta['alpha'] = bool(self.trns)
            meta['bitdepth'] = 8
            meta['planes'] = 3 + bool(self.trns)
            plte = self.palette()
            def iterpal(pixels):
                for row in pixels:
                    row = [plte[x] for x in row]
                    yield array('B', itertools.chain(*row))
            pixels = iterpal(pixels)
        elif self.trns:
            it = self.transparent
            maxval = 2**meta['bitdepth']-1
            planes = meta['planes']
            meta['alpha'] = True
            meta['planes'] += 1
            typecode = 'BH'[meta['bitdepth']>8]
            def itertrns(pixels):
                for row in pixels:
                    row = group(row, planes)
                    opa = map(it.__ne__, row)
                    opa = map(maxval.__mul__, opa)
                    opa = list(zip(opa))
                    yield array(typecode,
                      itertools.chain(*map(operator.add, row, opa)))
            pixels = itertrns(pixels)
        targetbitdepth = None
        if self.sbit:
            sbit = struct.unpack('%dB' % len(self.sbit), self.sbit)
            targetbitdepth = max(sbit)
            if targetbitdepth > meta['bitdepth']:
                raise Error('sBIT chunk %r exceeds bitdepth %d' %
                    (sbit,self.bitdepth))
            if min(sbit) <= 0:
                raise Error('sBIT chunk %r has a 0-entry' % sbit)
            if targetbitdepth == meta['bitdepth']:
                targetbitdepth = None
        if targetbitdepth:
            shift = meta['bitdepth'] - targetbitdepth
            meta['bitdepth'] = targetbitdepth
            def itershift(pixels):
                for row in pixels:
                    yield [p >> shift for p in row]
            pixels = itershift(pixels)
        return x,y,pixels,meta

    def asFloat(self, maxval=1.0):

        x,y,pixels,info = self.asDirect()
        sourcemaxval = 2**info['bitdepth']-1
        del info['bitdepth']
        info['maxval'] = float(maxval)
        factor = float(maxval)/float(sourcemaxval)
        def iterfloat():
            for row in pixels:
                yield [factor * p for p in row]
        return x,y,iterfloat(),info

    def _as_rescale(self, get, targetbitdepth):

        width,height,pixels,meta = get()
        maxval = 2**meta['bitdepth'] - 1
        targetmaxval = 2**targetbitdepth - 1
        factor = float(targetmaxval) / float(maxval)
        meta['bitdepth'] = targetbitdepth
        def iterscale():
            for row in pixels:
                yield [int(round(x*factor)) for x in row]
        if maxval == targetmaxval:
            return width, height, pixels, meta
        else:
            return width, height, iterscale(), meta

    def asRGB8(self):

        return self._as_rescale(self.asRGB, 8)

    def asRGBA8(self):

        return self._as_rescale(self.asRGBA, 8)

    def asRGB(self):
        width,height,pixels,meta = self.asDirect()
        if meta['alpha']:
            raise Error("will not convert image with alpha channel to RGB")
        if not meta['greyscale']:
            return width,height,pixels,meta
        meta['greyscale'] = False
        typecode = 'BH'[meta['bitdepth'] > 8]
        def iterrgb():
            for row in pixels:
                a = array(typecode, [0]) * 3 * width
                for i in range(3):
                    a[i::3] = row
                yield a
        return width,height,iterrgb(),meta

    def asRGBA(self):
        width,height,pixels,meta = self.asDirect()
        if meta['alpha'] and not meta['greyscale']:
            return width,height,pixels,meta
        typecode = 'BH'[meta['bitdepth'] > 8]
        maxval = 2**meta['bitdepth'] - 1
        maxbuffer = struct.pack('=' + typecode, maxval) * 4 * width
        def newarray():
            return array(typecode, maxbuffer)

        if meta['alpha'] and meta['greyscale']:
            def convert():
                for row in pixels:
                    a = newarray()
                    pngfilters.convert_la_to_rgba(row, a)
                    yield a
        elif meta['greyscale']:
            def convert():
                for row in pixels:
                    a = newarray()
                    pngfilters.convert_l_to_rgba(row, a)
                    yield a
        else:
            assert not meta['alpha'] and not meta['greyscale']
            def convert():
                for row in pixels:
                    a = newarray()
                    pngfilters.convert_rgb_to_rgba(row, a)
                    yield a
        meta['alpha'] = True
        meta['greyscale'] = False
        return width,height,convert(),meta

def check_bitdepth_colortype(bitdepth, colortype):
    if bitdepth not in (1,2,4,8,16):
        raise FormatError("invalid bit depth %d" % bitdepth)
    if colortype not in (0,2,3,4,6):
        raise FormatError("invalid colour type %d" % colortype)
    if colortype & 1 and bitdepth > 8:
        raise FormatError(
          "Indexed images (colour type %d) cannot"
          " have bitdepth > 8 (bit depth %d)."
          " See http://www.w3.org/TR/2003/REC-PNG-20031110/#table111 ."
          % (bitdepth, colortype))
    if bitdepth < 8 and colortype not in (0,3):
        raise FormatError("Illegal combination of bit depth (%d)"
          " and colour type (%d)."
          " See http://www.w3.org/TR/2003/REC-PNG-20031110/#table111 ."
          % (bitdepth, colortype))

def isinteger(x):
    try:
        return int(x) == x
    except (TypeError, ValueError):
        return False

try:
    pngfilters
except NameError:
    class pngfilters(object):
        def undo_filter_sub(filter_unit, scanline, previous, result):

            ai = 0
            for i in range(filter_unit, len(result)):
                x = scanline[i]
                a = result[ai]
                result[i] = (x + a) & 0xff
                ai += 1
        undo_filter_sub = staticmethod(undo_filter_sub)

        def undo_filter_up(filter_unit, scanline, previous, result):


            for i in range(len(result)):
                x = scanline[i]
                b = previous[i]
                result[i] = (x + b) & 0xff
        undo_filter_up = staticmethod(undo_filter_up)

        def undo_filter_average(filter_unit, scanline, previous, result):


            ai = -filter_unit
            for i in range(len(result)):
                x = scanline[i]
                if ai < 0:
                    a = 0
                else:
                    a = result[ai]
                b = previous[i]
                result[i] = (x + ((a + b) >> 1)) & 0xff
                ai += 1
        undo_filter_average = staticmethod(undo_filter_average)

        def undo_filter_paeth(filter_unit, scanline, previous, result):

            ai = -filter_unit
            for i in range(len(result)):
                x = scanline[i]
                if ai < 0:
                    a = c = 0
                else:
                    a = result[ai]
                    c = previous[ai]
                b = previous[i]
                p = a + b - c
                pa = abs(p - a)
                pb = abs(p - b)
                pc = abs(p - c)
                if pa <= pb and pa <= pc:
                    pr = a
                elif pb <= pc:
                    pr = b
                else:
                    pr = c
                result[i] = (x + pr) & 0xff
                ai += 1
        undo_filter_paeth = staticmethod(undo_filter_paeth)

        def convert_la_to_rgba(row, result):
            for i in range(3):
                result[i::4] = row[0::2]
            result[3::4] = row[1::2]
        convert_la_to_rgba = staticmethod(convert_la_to_rgba)

        def convert_l_to_rgba(row, result):
            for i in range(3):
                result[i::4] = row
        convert_l_to_rgba = staticmethod(convert_l_to_rgba)

        def convert_rgb_to_rgba(row, result):
            for i in range(3):
                result[i::4] = row[i::3]
        convert_rgb_to_rgba = staticmethod(convert_rgb_to_rgba)


def read_pam_header(infile):
    header = dict()
    while True:
        l = infile.readline().strip()
        if l == b'ENDHDR':
            break
        if not l:
            raise EOFError('PAM ended prematurely')
        if l[0] == b'#':
            continue
        l = l.split(None, 1)
        if l[0] not in header:
            header[l[0]] = l[1]
        else:
            header[l[0]] += b' ' + l[1]

    required = [b'WIDTH', b'HEIGHT', b'DEPTH', b'MAXVAL']
    WIDTH,HEIGHT,DEPTH,MAXVAL = required
    present = [x for x in required if x in header]
    if len(present) != len(required):
        raise Error('PAM file must specify WIDTH, HEIGHT, DEPTH, and MAXVAL')
    width = int(header[WIDTH])
    height = int(header[HEIGHT])
    depth = int(header[DEPTH])
    maxval = int(header[MAXVAL])
    if (width <= 0 or
        height <= 0 or
        depth <= 0 or
        maxval <= 0):
        raise Error(
          'WIDTH, HEIGHT, DEPTH, MAXVAL must all be positive integers')
    return 'P7', width, height, depth, maxval

def read_pnm_header(infile, supported=(b'P5', b'P6')):
    type = infile.read(3).rstrip()
    if type not in supported:
        raise NotImplementedError('file format %s not supported' % type)
    if type == b'P7':
        return read_pam_header(infile)
    expected = 4
    pbm = (b'P1', b'P4')
    if type in pbm:
        expected = 3
    header = [type]
    def getc():
        c = infile.read(1)
        if not c:
            raise Error('premature EOF reading PNM header')
        return c

    c = getc()
    while True:
        while c.isspace():
            c = getc()
        while c == '#':
            while c not in b'\n\r':
                c = getc()
        if not c.isdigit():
            raise Error('unexpected character %s found in header' % c)
        token = b''
        while c.isdigit():
            token += c
            c = getc()
        header.append(int(token))
        if len(header) == expected:
            break
    while c == '#':
        while c not in '\n\r':
            c = getc()
    if not c.isspace():
        raise Error('expected header to end with whitespace, not %s' % c)

    if type in pbm:

        header.append(1)
    depth = (1,3)[type == b'P6']
    return header[0], header[1], header[2], depth, header[3]

def write_pnm(file, width, height, pixels, meta):

    bitdepth = meta['bitdepth']
    maxval = 2**bitdepth - 1
    planes = meta['planes']
    assert planes in (1,2,3,4)
    if planes in (1,3):
        if 1 == planes:
            fmt = 'P5'
        else:
            fmt = 'P6'
        header = '%s %d %d %d\n' % (fmt, width, height, maxval)
    if planes in (2,4):
        if 2 == planes:
            tupltype = 'GRAYSCALE_ALPHA'
        else:
            tupltype = 'RGB_ALPHA'
        header = ('P7\nWIDTH %d\nHEIGHT %d\nDEPTH %d\nMAXVAL %d\n'
                  'TUPLTYPE %s\nENDHDR\n' %
                  (width, height, planes, maxval, tupltype))
    file.write(header.encode('ascii'))
    vpr = planes * width
    fmt = '>%d' % vpr
    if maxval > 0xff:
        fmt = fmt + 'H'
    else:
        fmt = fmt + 'B'
    for row in pixels:
        file.write(struct.pack(fmt, *row))
    file.flush()

def color_triple(color):
    if color.startswith('#') and len(color) == 4:
        return (int(color[1], 16),
                int(color[2], 16),
                int(color[3], 16))
    if color.startswith('#') and len(color) == 7:
        return (int(color[1:3], 16),
                int(color[3:5], 16),
                int(color[5:7], 16))
    elif color.startswith('#') and len(color) == 13:
        return (int(color[1:5], 16),
                int(color[5:9], 16),
                int(color[9:13], 16))

def _add_common_options(parser):
    parser.add_option("-i", "--interlace",
                      default=False, action="store_true",
                      help="create an interlaced PNG file (Adam7)")
    parser.add_option("-t", "--transparent",
                      action="store", type="string", metavar="#RRGGBB",
                      help="mark the specified colour as transparent")
    parser.add_option("-b", "--background",
                      action="store", type="string", metavar="#RRGGBB",
                      help="save the specified background colour")
    parser.add_option("-g", "--gamma",
                      action="store", type="float", metavar="value",
                      help="save the specified gamma value")
    parser.add_option("-c", "--compression",
                      action="store", type="int", metavar="level",
                      help="zlib compression level (0-9)")
    return parser

def _main(argv):
    from optparse import OptionParser
    version = '%prog ' + __version__
    parser = OptionParser(version=version)
    parser.set_usage("%prog [options] [imagefile]")
    parser.add_option('-r', '--read-png', default=False,
                      action='store_true',
                      help='Read PNG, write PNM')
    parser.add_option("-a", "--alpha",
                      action="store", type="string", metavar="pgmfile",
                      help="alpha channel transparency (RGBA)")
    _add_common_options(parser)

    (options, args) = parser.parse_args(args=argv[1:])
    if options.transparent is not None:
        options.transparent = color_triple(options.transparent)
    if options.background is not None:
        options.background = color_triple(options.background)
    if len(args) == 0:
        infilename = '-'
        infile = sys.stdin
    elif len(args) == 1:
        infilename = args[0]
        infile = open(infilename, 'rb')
    else:
        parser.error("more than one input file")
    outfile = sys.stdout
    if sys.platform == "win32":
        import msvcrt, os
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    if options.read_png:
        png = Reader(file=infile)
        width,height,pixels,meta = png.asDirect()
        write_pnm(outfile, width, height, pixels, meta) 
    else:
        format, width, height, depth, maxval = \
          read_pnm_header(infile, (b'P5',b'P6',b'P7'))
        greyscale = depth <= 2
        pamalpha = depth in (2,4)
        supported = [2**x-1 for x in range(1,17)]
        try:
            mi = supported.index(maxval)
        except ValueError:
            raise NotImplementedError(
              'your maxval (%s) not in supported list %s' %
              (maxval, str(supported)))
        bitdepth = mi+1
        writer = Writer(width, height,
                        greyscale=greyscale,
                        bitdepth=bitdepth,
                        interlace=options.interlace,
                        transparent=options.transparent,
                        background=options.background,
                        alpha=bool(pamalpha or options.alpha),
                        gamma=options.gamma,
                        compression=options.compression)
        if options.alpha:
            pgmfile = open(options.alpha, 'rb')
            format, awidth, aheight, adepth, amaxval = \
              read_pnm_header(pgmfile, 'P5')
            if amaxval != '255':
                raise NotImplementedError(
                  'maxval %s not supported for alpha channel' % amaxval)
            if (awidth, aheight) != (width, height):
                raise ValueError("alpha channel image size mismatch"
                                 " (%s has %sx%s but %s has %sx%s)"
                                 % (infilename, width, height,
                                    options.alpha, awidth, aheight))
            writer.convert_ppm_and_pgm(infile, pgmfile, outfile)
        else:
            writer.convert_pnm(infile, outfile)


if __name__ == '__main__':
    try:
        _main(sys.argv)
    except Error as e:
        print(e, file=sys.stderr)
