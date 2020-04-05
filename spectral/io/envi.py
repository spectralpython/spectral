'''
ENVI [#envi-trademark]_ is a popular commercial software package for processing
and analyzing geospatial imagery.  SPy supports reading imagery with associated
ENVI header files and reading & writing spectral libraries with ENVI headers.
ENVI files are opened automatically by the SPy :func:`~spectral.image` function
but can also be called explicitly.  It may be necessary to open an ENVI file
explicitly if the data file is in a separate directory from the header or if
the data file has an unusual file extension that SPy can not identify.

    >>> import spectral.io.envi as envi
    >>> img = envi.open('cup95eff.int.hdr', '/Users/thomas/spectral_data/cup95eff.int')

.. [#envi-trademark] ENVI is a registered trademark of Exelis, Inc.
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import os
import sys
import warnings

import spectral as spy
from ..spectral import BandInfo
from ..utilities.python23 import IS_PYTHON3, is_string
from ..utilities.errors import SpyException
from .bilfile import BilFile
from .bipfile import BipFile
from .bsqfile import BsqFile
from .spyfile import (FileNotFoundError, find_file_path, interleave_transpose,
                      InvalidFileError, SpyFile)



if IS_PYTHON3:
    import builtins
else:
    import __builtin__ as builtins

logger = logging.getLogger('spectral')

# Known ENVI data file extensions. Upper and lower case versions will be
# recognized, as well as interleaves ('bil', 'bip', 'bsq'), and no extension.
KNOWN_EXTS = ['img', 'dat', 'sli', 'hyspex', 'raw']

dtype_map = [('1', np.uint8),                   # unsigned byte
             ('2', np.int16),                   # 16-bit int
             ('3', np.int32),                   # 32-bit int
             ('4', np.float32),                 # 32-bit float
             ('5', np.float64),                 # 64-bit float
             ('6', np.complex64),               # 2x32-bit complex
             ('9', np.complex128),              # 2x64-bit complex
             ('12', np.uint16),                 # 16-bit unsigned int
             ('13', np.uint32),                 # 32-bit unsigned int
             ('14', np.int64),                  # 64-bit int
             ('15', np.uint64)]                 # 64-bit unsigned int
envi_to_dtype = dict((k, np.dtype(v).char) for (k, v) in dtype_map)
dtype_to_envi = dict(tuple(reversed(item)) for item in list(envi_to_dtype.items()))

class EnviException(SpyException):
    '''Base class for ENVI file-related exceptions.'''
    pass

class EnviDataTypeError(EnviException, TypeError):
    '''Raised when saving invalid image data type to ENVI format.
    '''
    def __init__(self, dtype):
        msg = 'Image data type "{0}" can not be saved to ENVI data file. ' \
          'Call spectral.envi.get_supported_dtypes for a list of supported ' \
          'data type names.'.format(np.dtype(dtype).name)
        super(EnviDataTypeError, self).__init__(msg)

class EnviFeatureNotSupported(EnviException, NotImplementedError):
    '''A specified ENVI capability is not supported by the spectral module.'''
    pass

class FileNotAnEnviHeader(EnviException, InvalidFileError):
    '''Raised when "ENVI" does not appear on the first line of the file.'''
    def __init__(self, msg):
        super(FileNotAnEnviHeader, self).__init__(msg)

class MissingEnviHeaderParameter(EnviException):
    '''Raised when a mandatory header parameter is missing.'''
    def __init__(self, param):
        msg = 'Mandatory parameter "%s" missing from header file.' % param
        super(MissingEnviHeaderParameter, self).__init__(msg)

class EnviHeaderParsingError(EnviException, InvalidFileError):
    '''Raised upon failure to parse parameter/value pairs from a file.'''
    def __init__(self):
        msg = 'Failed to parse ENVI header file.'
        super(EnviHeaderParsingError, self).__init__(msg)

class EnviDataFileNotFoundError(EnviException, FileNotFoundError):
    '''Raised when data file associated with a header is not found.'''
    pass

def _validate_dtype(dtype):
    '''Raises EnviDataTypeError if dtype can not be written to ENVI file.'''
    typename = np.dtype(dtype).name
    if typename not in [np.dtype(t).name for t in list(dtype_to_envi.keys())]:
        raise EnviDataTypeError(dtype)

def get_supported_dtypes():
    '''Returns list of names of image data types supported by ENVI format.'''
    return [np.dtype(t).name for t in list(dtype_to_envi.keys())]

def read_envi_header(file):
    '''
    USAGE: hdr = read_envi_header(file)

    Reads an ENVI ".hdr" file header and returns the parameters in a
    dictionary as strings.  Header field names are treated as case
    insensitive and all keys in the dictionary are lowercase.
    '''
    f = builtins.open(file, 'r')

    try:
        starts_with_ENVI = f.readline().strip().startswith('ENVI')
    except UnicodeDecodeError:
        msg = 'File does not appear to be an ENVI header (appears to be a ' \
          'binary file).'
        f.close()
        raise FileNotAnEnviHeader(msg)
    else:
        if not starts_with_ENVI:
            msg = 'File does not appear to be an ENVI header (missing "ENVI" \
              at beginning of first line).'
            f.close()
            raise FileNotAnEnviHeader(msg)

    lines = f.readlines()
    f.close()

    dict = {}
    have_nonlowercase_param = False
    support_nonlowercase_params = spy.settings.envi_support_nonlowercase_params
    try:
        while lines:
            line = lines.pop(0)
            if line.find('=') == -1: continue
            if line[0] == ';': continue

            (key, sep, val) = line.partition('=')
            key = key.strip()
            if not key.islower():
                have_nonlowercase_param = True
                if not support_nonlowercase_params:
                    key = key.lower()
            val = val.strip()
            if val and val[0] == '{':
                str = val.strip()
                while str[-1] != '}':
                    line = lines.pop(0)
                    if line[0] == ';': continue

                    str += '\n' + line.strip()
                if key == 'description':
                    dict[key] = str.strip('{}').strip()
                else:
                    vals = str[1:-1].split(',')
                    for j in range(len(vals)):
                        vals[j] = vals[j].strip()
                    dict[key] = vals
            else:
                dict[key] = val

        if have_nonlowercase_param and not support_nonlowercase_params:
            msg = 'Parameters with non-lowercase names encountered ' \
                  'and converted to lowercase. To retain source file ' \
                  'parameter name capitalization, set ' \
                  'spectral.settings.envi_support_nonlowercase_params to ' \
                  'True.'
            warnings.warn(msg)
            logger.debug('ENVI header parameter names converted to lower case.')
        return dict
    except:
        raise EnviHeaderParsingError()


def gen_params(envi_header):
    '''
    Parse an envi_header to a `Params` object.

    Arguments:

    `envi_header` (dict or file_name):

        A dict or an `.hdr` file name
    '''
    if not isinstance(envi_header, dict):
        headerPath = find_file_path(envi_header)
        h = read_envi_header(headerPath)
    else:
        h = envi_header

    class Params:
        pass
    p = Params()
    p.nbands = int(h["bands"])
    p.nrows = int(h["lines"])
    p.ncols = int(h["samples"])
    p.offset = int(h["header offset"]) if "header offset" in h else int(0)
    p.byte_order = int(h["byte order"])
    p.dtype = np.dtype(envi_to_dtype[str(h["data type"])]).str
    if p.byte_order != spy.byte_order:
        p.dtype = np.dtype(p.dtype).newbyteorder().str
    p.filename = None
    return p

def _has_frame_offset(params):
    '''
    Returns True if header params indicate non-zero frame offsets.

    Arguments:

        `params` (dict):

            Dictionary of header parameters assocaited with hdr file.

    Returns:

        bool

    This function returns True when either "major frame offsets" or
    "minor frame offsets" is specified and contains a non-zero value.
    '''
    for param in ['major frame offsets', 'minor frame offsets']:
        if param in params:
            val = params[param]
            if np.iterable(val):
                offsets = [int(x) for x in val]
            else:
                offsets = [int(val)] * 2
            if not np.all(np.equal(offsets, 0)):
                return True
    return False

def check_compatibility(header):
    '''
    Verifies that all features of an ENVI header are supported.
    '''
    if is_string(header):
        header = read_envi_header(find_file_path(header))

    mandatory_params = ['lines', 'samples', 'bands', 'data type',
                        'interleave', 'byte order']
    for p in mandatory_params:
        if p not in header:
            raise MissingEnviHeaderParameter(p)

    if _has_frame_offset(header):
        raise EnviFeatureNotSupported(
            'ENVI image frame offsets are not supported.')

def open(file, image=None):
    '''
    Opens an image or spectral library with an associated ENVI HDR header file.

    Arguments:

        `file` (str):

            Name of the header file for the image.

        `image` (str):

            Optional name of the associated image data file.

    Returns:

        :class:`spectral.SpyFile` or :class:`spectral.io.envi.SpectralLibrary`
        object.

    Raises:

        TypeError, EnviDataFileNotFoundError

    If the specified file is not found in the current directory, all
    directories listed in the SPECTRAL_DATA environment variable will be
    searched until the file is found.  Based on the name of the header file,
    this function will search for the image file in the same directory as the
    header, looking for a file with the same name as the header but different
    extension. Extensions recognized are .img, .dat, .sli, and no extension.
    Capitalized versions of the file extensions are also searched.
    '''

    header_path = find_file_path(file)
    h = read_envi_header(header_path)
    check_compatibility(h)
    p = gen_params(h)

    inter = h["interleave"]

    #  Validate image file name
    if not image:
        #  Try to determine the name of the image file
        (header_path_title, header_ext) = os.path.splitext(header_path)
        if header_ext.lower() == '.hdr':
            exts = [ext.lower() for ext in KNOWN_EXTS] + [inter.lower()]
            exts = [''] + exts + [ext.upper() for ext in exts]
            for ext in exts:
                if len(ext) == 0:
                    testname = header_path_title
                else:
                    testname = header_path_title + '.' + ext
                if os.path.isfile(testname):
                    image = testname
                    break
        if not image:
            msg = 'Unable to determine the ENVI data file name for the ' \
              'given header file. You can specify the data file by passing ' \
              'its name as the optional `image` argument to envi.open.'
            raise EnviDataFileNotFoundError(msg)
    else:
        image = find_file_path(image)

    p.filename = image

    if h.get('file type') == 'ENVI Spectral Library':
        # File is a spectral library
        data = np.fromfile(p.filename, p.dtype, p.ncols * p.nrows)
        data.shape = (p.nrows, p.ncols)
        return SpectralLibrary(data, h, p)

    #  Create the appropriate object type for the interleave format.
    inter = h["interleave"]
    if inter == 'bil' or inter == 'BIL':
        img = BilFile(p, h)
    elif inter == 'bip' or inter == 'BIP':
        img = BipFile(p, h)
    else:
        img = BsqFile(p, h)

    img.scale_factor = float(h.get('reflectance scale factor', 1.0))

    # Add band info

    if 'wavelength' in h:
        try:
            img.bands.centers = [float(b) for b in h['wavelength']]
        except:
            pass
    if 'fwhm' in h:
        try:
            img.bands.bandwidths = [float(f) for f in h['fwhm']]
        except:
            pass
    img.bands.band_unit = h.get('wavelength units', None)

    if 'bbl' in h:
        try:
            h['bbl'] = [int(float(b)) for b in h['bbl']]
        except:
            logger.warning('Unable to parse bad band list (bbl) in ENVI ' \
                           'header as integers.')
    return img


def check_new_filename(hdr_file, img_ext, force):
    '''Raises an exception if the associated header or image file names exist.
    '''
    if img_ext is None:
        img_ext = ''
    elif len(img_ext) > 0 and img_ext[0] != '.':
        img_ext = '.' + img_ext
    hdr_file = os.path.realpath(hdr_file)
    (base, ext) = os.path.splitext(hdr_file)
    if ext.lower() != '.hdr':
        raise EnviException('Header file name must end in ".hdr" or ".HDR".')
    image_file = base + img_ext
    if not force:
        if os.path.isfile(hdr_file):
            raise EnviException('Header file %s already exists. Use `force` '
                                'keyword to force overwrite.' % hdr_file)
        if os.path.isfile(image_file):
            raise EnviException('Image file %s already exists. Use `force` '
                                'keyword to force overwrite.' % image_file)
    return (hdr_file, image_file)


def save_image(hdr_file, image, **kwargs):
    '''
    Saves an image to disk.

    Arguments:

        `hdr_file` (str):

            Header file (with ".hdr" extension) name with path.

        `image` (SpyFile object or numpy.ndarray):

            The image to save.

    Keyword Arguments:

        `dtype` (numpy dtype or type string):

            The numpy data type with which to store the image.  For example,
            to store the image in 16-bit unsigned integer format, the argument
            could be any of `numpy.uint16`, "u2", "uint16", or "H".

        `force` (bool):

            If the associated image file or header already exist and `force` is
            True, the files will be overwritten; otherwise, if either of the
            files exist, an exception will be raised.

        `ext` (str or None):

            The extension to use for the image file.  If not specified, the
            default extension ".img" will be used.  If `ext` is an empty
            string or is None, the image file will have the same name as the
            header but without the ".hdr" extension.

        `interleave` (str):

            The band interleave format to use in the file.  This argument
            should be one of "bil", "bip", or "bsq".  If not specified, the
            image will be written in BIP interleave.

        `byteorder` (int or string):

            Specifies the byte order (endian-ness) of the data as
            written to disk. For little endian, this value should be
            either 0 or "little".  For big endian, it should be
            either 1 or "big". If not specified, native byte order
            will be used.

        `metadata` (dict):

            A dict containing ENVI header parameters (e.g., parameters
            extracted from a source image).

    Example::

        >>> # Save the first 10 principal components of an image
        >>> data = open_image('92AV3C.lan').load()
        >>> pc = principal_components(data)
        >>> pcdata = pc.reduce(num=10).transform(data)
        >>> envi.save_image('pcimage.hdr', pcdata, dtype=np.float32)

    If the source image being saved was already in ENVI format, then the
    SpyFile object for that image will contain a `metadata` dict that can be
    passed as the `metadata` keyword. However, care should be taken to ensure
    that all the metadata fields from the source image are still accurate
    (e.g., band names or wavelengths will no longer be correct if the data
    being saved are from a principal components transformation).

    '''
    data, metadata = _prepared_data_and_metadata(hdr_file, image, **kwargs)
    metadata['file type'] = "ENVI Standard"
    _write_image(hdr_file, data, metadata, **kwargs)


def save_classification(hdr_file, image, **kwargs):
    '''Saves a classification image to disk.

    Arguments:

        `hdr_file` (str):

            Header file (with ".hdr" extension) name with path.

        `image` (SpyFile object or numpy.ndarray):

            The image to save.

    Keyword Arguments:

        `dtype` (numpy dtype or type string):

            The numpy data type with which to store the image.  For example,
            to store the image in 16-bit unsigned integer format, the argument
            could be any of `numpy.uint16`, "u2", "uint16", or "H".

        `force` (bool):

            If the associated image file or header already exist and `force` is
            True, the files will be overwritten; otherwise, if either of the
            files exist, an exception will be raised.

        `ext` (str):

            The extension to use for the image file.  If not specified, the
            default extension ".img" will be used.  If `ext` is an empty
            string, the image file will have the same name as the header but
            without the ".hdr" extension.

        `interleave` (str):

            The band interleave format to use in the file.  This argument
            should be one of "bil", "bip", or "bsq".  If not specified, the
            image will be written in BIP interleave.

        `byteorder` (int or string):

            Specifies the byte order (endian-ness) of the data as
            written to disk. For little endian, this value should be
            either 0 or "little".  For big endian, it should be
            either 1 or "big". If not specified, native byte order
            will be used.

        `metadata` (dict):

            A dict containing ENVI header parameters (e.g., parameters
            extracted from a source image).

        `class_names` (array of strings):

            For classification results, specifies the names to assign each
            integer in the class map being written.  If not given, default
            class names are created. 

        `class_colors` (array of RGB-tuples):

            For classification results, specifies colors to assign each
            integer in the class map being written.  If not given, default
            colors are automatically generated.  

    If the source image being saved was already in ENVI format, then the
    SpyFile object for that image will contain a `metadata` dict that can be
    passed as the `metadata` keyword. However, care should be taken to ensure
    that all the metadata fields from the source image are still accurate
    (e.g., wavelengths do not apply to classification results).

    '''
    data, metadata = _prepared_data_and_metadata(hdr_file, image, **kwargs)
    metadata['file type'] = "ENVI Classification"

    class_names = kwargs.get('class_names', metadata.get('class_names', None))
    class_colors = kwargs.get('class_colors', metadata.get('class_colors', None))
    if class_names is None:
        # guess the number of classes and create default class names
        n_classes = int(np.max(data) + 1)
        metadata['classes'] = str(n_classes)
        metadata['class names'] = (['Unclassified'] + 
                                   ['Class ' + str(i) for i in range(1, n_classes)])
        # if keyword is given, override whatever is in the metadata dict
    else:
        n_classes = int(max(np.max(data) + 1, len(class_names)))
        metadata['class names'] = class_names
        metadata['classes'] = str(n_classes)
        
    # the resulting value for 'class lookup' needs to be a flattened array.
    colors = []
    if class_colors is not None:
        try:
            for color in class_colors:
                # call list() in case color is a numpy array
                colors += list(color)
        except:
            # list was already flattened
            colors = list(class_colors)
    if len(colors) < n_classes * 3:
        colors = []
        for i in range(n_classes):
            colors += list(spy.spy_colors[i % len(spy.spy_colors)])
    metadata['class lookup'] = colors

    _write_image(hdr_file, data, metadata, **kwargs)

def _prepared_data_and_metadata(hdr_file, image, **kwargs):
    '''
    Return data array and metadata dict representing `image`.
    '''
    endian_out = str(kwargs.get('byteorder', sys.byteorder)).lower()
    if endian_out in ('0', 'little'):
        endian_out = 'little'
    elif endian_out in ('1', 'big'):
        endian_out = 'big'
    else:
        raise ValueError('Invalid byte order: "%s".' % endian_out)

    if isinstance(image, np.ndarray):
        data = image
        src_interleave = 'bip'
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
        swap = False
        metadata = {}    
    elif isinstance(image, SpyFile):
        if image.using_memmap is True:
            data = image._memmap
            src_interleave = {spy.BSQ: 'bsq', spy.BIL: 'bil',
                              spy.BIP: 'bip'}[image.interleave]
            swap = image.swap
        else:
            data = image.load(dtype=image.dtype, scale=False)
            src_interleave = 'bip'
            swap = False
        metadata = image.metadata.copy()
    else:
        data = image.load()
        src_interleave = 'bip'
        swap = False
        if hasattr(image, 'metadata'):
            metadata = image.metadata.copy()
        else:
            metadata = {}

    metadata.update(kwargs.get('metadata', {}))
    add_image_info_to_metadata(image, metadata)
    if hasattr(image, 'bands'):
        add_band_info_to_metadata(image.bands, metadata)

    dtype = np.dtype(kwargs.get('dtype', data.dtype)).char
    _validate_dtype(dtype)
    if dtype != data.dtype.char:
        data = data.astype(dtype)
    metadata['data type'] = dtype_to_envi[dtype]

    interleave = kwargs.get('interleave', 'bip').lower()
    if interleave not in ['bil', 'bip', 'bsq']:
        raise ValueError('Invalid interleave: %s'
                         % str(kwargs['interleave']))
    if interleave != src_interleave:
        data = data.transpose(interleave_transpose(src_interleave, interleave))
    metadata['interleave'] = interleave
    metadata['byte order'] = 1 if endian_out == 'big' else 0
    if (endian_out == sys.byteorder and not data.dtype.isnative) or \
      (endian_out != sys.byteorder and data.dtype.isnative):
        data = data.byteswap()

    return data, metadata


# A few header parameters need to be set no matter what is provided in the
# supplied metadata.
def add_image_info_to_metadata(image, metadata):
    '''
    Set keys in metadata dict to values appropriate for image.
    '''
    if isinstance(image, SpyFile) and image.scale_factor != 1:
        metadata['reflectance scale factor'] = image.scale_factor

    # Always write data from start of file, regardless of what was in
    # the provided metadata.
    offset = int(metadata.get('header offset', 0))
    if offset != 0:
        logger.debug('Ignoring non-zero header offset in provided metadata.')
    metadata['header offset'] = 0

    metadata['lines'] = image.shape[0]
    metadata['samples'] = image.shape[1]
    if len(image.shape) == 3:
        metadata['bands'] = image.shape[2]
    else:
        metadata['bands'] = 1


def add_band_info_to_metadata(bands, metadata, overwrite=False):
    '''Adds BandInfo data to the metadata dict.

    Data is only added if not already present, unless `overwrite` is True.
    '''
    if bands.centers is not None and (overwrite is True or
                                      'wavelength' not in metadata):
        metadata['wavelength'] = bands.centers
    if bands.bandwidths is not None and (overwrite is True or
                                      'fwhm' not in metadata):
        metadata['fwhm'] = bands.bandwidths
    if bands.band_unit is not None and (overwrite is True or
                                        'wavelength units' not in metadata):
        metadata['wavelength units'] = bands.band_unit
        

def _write_image(hdr_file, data, header, **kwargs):
    '''
    Write `data` as an ENVI file using the metadata in `header`.
    '''
    check_compatibility(header)
    force = kwargs.get('force', False)
    img_ext = kwargs.get('ext', '.img')
    
    (hdr_file, img_file) = check_new_filename(hdr_file, img_ext, force)
    write_envi_header(hdr_file, header, is_library=False)
    logger.debug('Saving', img_file)
    # bufsize = data.shape[0] * data.shape[1] * np.dtype(dtype).itemsize
    bufsize = data.shape[0] * data.shape[1] * data.dtype.itemsize
    fout = builtins.open(img_file, 'wb', bufsize)
    fout.write(data.tostring())
    fout.close()


def create_image(hdr_file, metadata=None, **kwargs):
    '''
    Creates an image file and ENVI header with a memmep array for write access.

    Arguments:

        `hdr_file` (str):

            Header file (with ".hdr" extension) name with path.

        `metadata` (dict):

            Metadata to specify the image file format. The following parameters
            (in ENVI header format) are required, if not specified via
            corresponding keyword arguments: "bands", "lines", "samples",
            and "data type".

    Keyword Arguments:

        `dtype` (numpy dtype or type string):

            The numpy data type with which to store the image.  For example,
            to store the image in 16-bit unsigned integer format, the argument
            could be any of `numpy.uint16`, "u2", "uint16", or "H". If this
            keyword is given, it will override the "data type" parameter in
            the `metadata` argument.

        `force` (bool, False by default):

            If the associated image file or header already exist and `force` is
            True, the files will be overwritten; otherwise, if either of the
            files exist, an exception will be raised.

        `ext` (str):

            The extension to use for the image file.  If not specified, the
            default extension ".img" will be used.  If `ext` is an empty
            string, the image file will have the same name as the header but
            without the ".hdr" extension.

        `interleave` (str):

            Must be one of "bil", "bip", or "bsq". This keyword supercedes the
            value of "interleave" in the metadata argument, if given. If no
            interleave is specified (via keyword or `metadata`), "bip" is
            assumed.

        `shape` (tuple of integers):

            Specifies the number of rows, columns, and bands in the image.
            This keyword should be either of the form (R, C, B) or (R, C),
            where R, C, and B specify the number or rows, columns, and bands,
            respectively. If B is omitted, the number of bands is assumed to
            be one. If this keyword is given, its values supercede the values
            of "bands", "lines", and "samples" if they are present in the
            `metadata` argument.

        `offset` (integer, default 0):

            The offset (in bytes) of image data from the beginning of the file.
            This value supercedes the value of "header offset" in the metadata
            argument (if given).

    Returns:

        `SpyFile` object:

            To access a `numpy.memmap` for the returned `SpyFile` object, call
            the `open_memmap` method of the returned object.

    Examples:

        Creating a new image from metadata::

            >>> md = {'lines': 30,
                      'samples': 40,
                      'bands': 50,
                      'data type': 12}
            >>> img = envi.create_image('new_image.hdr', md)

        Creating a new image via keywords::

            >>> img = envi.create_image('new_image2.hdr',
                                        shape=(30, 40, 50),
                                        dtype=np.uint16)

        Writing to the new image using a memmap interface::

            >>> # Set all band values for a single pixel to 100.
            >>> mm = img.open_memmap(writable=True)
            >>> mm[30, 30] = 100

    '''
    force = kwargs.get('force', False)
    img_ext = kwargs.get('ext', '.img')
    memmap_mode = kwargs.get('memmap_mode', 'w+')
    (hdr_file, img_file) = check_new_filename(hdr_file, img_ext, force)

    default_metadata = {'header offset': 0, 'interleave': 'bip'}
    
    if metadata is None:
        metadata = default_metadata
    else:
        default_metadata.update(metadata)
        metadata = default_metadata

    # Keyword args supercede metadata dict
    if 'shape' in kwargs:
        shape = kwargs['shape']
        metadata['lines'] = shape[0]
        metadata['samples'] = shape[1]
        if len(shape) == 3:
            metadata['bands'] = shape[2]
        else:
            metadata['bands'] = 1
    if 'offset' in kwargs:
        metadata['header offset'] = kwargs['offset']
    if 'dtype' in kwargs:
        metadata['data type'] = dtype_to_envi[np.dtype(kwargs['dtype']).char]
    if 'interleave' in kwargs:
        metadata['interleave'] = kwargs['interleave']

    metadata['byte order'] = spy.byte_order

    # Verify minimal set of parameters have been provided
    if 'lines' not in metadata:
        raise EnviException('Number of image rows is not defined.')
    elif 'samples' not in metadata:
        raise EnviException('Number of image columns is not defined.')
    elif 'bands' not in metadata:
        raise EnviException('Number of image bands is not defined.')
    elif 'samples' not in metadata:
        raise EnviException('Number of image columns is not defined.')
    elif 'data type' not in metadata:
        raise EnviException('Image data type is not defined.')

    params = gen_params(metadata)
    dt = np.dtype(params.dtype).char
    _validate_dtype(dt)
    params.filename = img_file
        
    is_library = False
    if metadata.get('file type') == 'ENVI Spectral Library':
        is_library = True
        raise NotImplementedError('ENVI Spectral Library cannot be created ')

    # Create the appropriate object type -> the memmap (=image) will be
    # created on disk
    inter = metadata["interleave"]
    (R, C, B) = (params.nrows, params.ncols, params.nbands)
    if inter.lower() not in ['bil', 'bip', 'bsq']:
        raise ValueError('Invalid interleave specified: %s.' % str(inter))
    if inter.lower() == 'bil':
        memmap = np.memmap(img_file, dtype=dt, mode=memmap_mode,
                           offset=params.offset, shape=(R, B, C))
        img = BilFile(params, metadata)
        img._memmap = memmap
    elif inter.lower() == 'bip':
        memmap = np.memmap(img_file, dtype=dt, mode=memmap_mode,
                           offset=params.offset, shape=(R, C, B))
        img = BipFile(params, metadata)
        img._memmap = memmap
    else:
        memmap = np.memmap(img_file, dtype=dt, mode=memmap_mode,
                           offset=params.offset, shape=(B, R, C))
        img = BsqFile(params, metadata)
        img._memmap = memmap

    # Write the header file after the image to assure write success
    write_envi_header(hdr_file, metadata, is_library=is_library)
    return img


class SpectralLibrary:
    '''
    The envi.SpectralLibrary class holds data contained in an ENVI-formatted
    spectral library file (.sli files), which stores data as specified by a
    corresponding .hdr header file.  The primary members of an
    Envi.SpectralLibrary object are:

        `spectra` (:class:`numpy.ndarray`):

            A subscriptable array of all spectra in the library. `spectra` will
            have shape `CxB`, where `C` is the number of spectra in the library
            and `B` is the number of bands for each spectrum.

        `names` (list of str):

            A length-`C` list of names corresponding to the spectra.

        `bands` (:class:`spectral.BandInfo`):

            Spectral bands associated with the library spectra.

    '''

    def __init__(self, data, header=None, params=None):
        '''Creates a new spectral library array

        Arguments:

            `data` (array-like):

                Array with shape `CxB`, where `C` is the number of spectra in
                the library and `B` is the number of bands for each spectrum.

            `header` (dict):

                Optional dict of ENVI header parameters.

            `params` (Params):

                Optional SpyFile Params object
        '''
        self.spectra = data
        (n_spectra, n_bands) = data.shape

        if header is None:
            header = {}
        header = header.copy()

        self.bands = BandInfo()
        centers = header.pop('wavelength', None)
        if centers is not None:
            if len(centers) != n_bands:
                raise ValueError('Number of band centers does not match data')
            self.bands.centers = [float(c) for c in centers]
        fwhm = header.pop('fwhm', None)
        if fwhm is not None:
            if len(fwhm) != n_bands:
                raise ValueError('Number of fwhm values does not match data')
            self.bands.bandwidths = [float(f) for f in fwhm]
        names = header.pop('spectra names', None)
        if names is not None:
            if len(names) != n_spectra:
                raise ValueError('Number of spectrum names does not match data')
            self.names = names
        else:
            self.names = [str(i + 1) for i in range(n_spectra)]
        self.bands.band_unit = header.get('wavelength units', "<unspecified>")
        self.bands.band_quantity = "Wavelength"
        self.params = params
        self.metadata = header.copy()
        self.metadata['data ignore value'] = 'NaN'

    def save(self, file_basename, description=None):
        '''
        Saves the spectral library to a library file.

        Arguments:

            `file_basename` (str):

                Name of the file (without extension) to save.

            `description` (str):

                Optional text description of the library.

        This method creates two files: `file_basename`.hdr and
        `file_basename`.sli.
        '''
        meta = self.metadata.copy()
        meta['samples'] = self.spectra.shape[1]
        meta['lines'] = self.spectra.shape[0]
        meta['bands'] = 1
        meta['header offset'] = 0
        meta['data type'] = 4           # 32-bit float
        meta['interleave'] = 'bsq'
        meta['byte order'] = spy.byte_order
        meta['wavelength units'] = self.bands.band_unit
        meta['spectra names'] = [str(n) for n in self.names]
        if self.bands.centers is not None:
            meta['wavelength'] = self.bands.centers
        if self.bands.bandwidths is not None:
            meta['fwhm'] = self.bands.bandwidths
        if (description):
            meta['description'] = description
        write_envi_header(file_basename + '.hdr', meta, True)
        fout = builtins.open(file_basename + '.sli', 'wb')
        self.spectra.astype('f').tofile(fout)
        fout.close()

def _write_header_param(fout, paramName, paramVal):
    if paramName.lower() == 'description':
        valStr = '{\n%s}' % '\n'.join(['  ' + line for line
                                       in paramVal.split('\n')])
    elif not is_string(paramVal) and hasattr(paramVal, '__len__'):
        valStr = '{ %s }' % (
            ' , '.join([str(v).replace(',', '-') for v in paramVal]),)
    else:
        valStr = str(paramVal)
    fout.write('%s = %s\n' % (paramName, valStr))


def write_envi_header(fileName, header_dict, is_library=False):
    fout = builtins.open(fileName, 'w')
    d = {}
    d.update(header_dict)
    if is_library:
        d['file type'] = 'ENVI Spectral Library'
    elif 'file type' not in d:
        d['file type'] = 'ENVI Standard'
    fout.write('ENVI\n')
    # Write the standard parameters at the top of the file
    std_params = ['description', 'samples', 'lines', 'bands', 'header offset',
                  'file type', 'data type', 'interleave', 'sensor type',
                  'byte order', 'reflectance scale factor', 'map info']
    for k in std_params:
        if k in d:
            _write_header_param(fout, k, d[k])
    for k in d:
        if k not in std_params:
            _write_header_param(fout, k, d[k])
    fout.close()

