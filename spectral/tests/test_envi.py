'''Tests of functions associated with the ENVI file format.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_envi.py
'''

import os

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

import spectral as spy
from spectral.io.envi import SpectralLibrary

MIXED_CASE_HEADER = '''ENVI
samples = 145
lines = 145
bands = 220
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bip
byte order = 0
some Param = 0
'''


class TestENVIWrite:
    '''Tests that SpyFile memmap interfaces read and write properly.'''

    def test_save_image_ndarray(self, testdir):
        '''Test saving an ENVI formatted image from a numpy.ndarray.'''
        (R, B, C) = (10, 20, 30)
        (r, b, c) = (3, 8, 23)
        datum = 33
        data = np.zeros((R, B, C), dtype=np.uint16)
        data[r, b, c] = datum
        fname = os.path.join(testdir, 'test_save_image_ndarray.hdr')
        spy.envi.save_image(fname, data, interleave='bil')
        img = spy.open_image(fname)
        assert_almost_equal(img[r, b, c], datum)

    def test_save_image_ndarray_no_ext(self, testdir):
        '''Test saving an ENVI formatted image with no image file extension.'''
        data = np.arange(1000, dtype=np.int16).reshape(10, 10, 10)
        base = os.path.join(testdir, 'test_save_image_ndarray_noext')
        hdr_file = base + '.hdr'
        spy.envi.save_image(hdr_file, data, ext='')
        rdata = spy.open_image(hdr_file).load()
        assert (np.all(data == rdata))

    def test_save_image_ndarray_alt_ext(self, testdir):
        '''Test saving an ENVI formatted image with alternate extension.'''
        data = np.arange(1000, dtype=np.int16).reshape(10, 10, 10)
        base = os.path.join(testdir, 'test_save_image_ndarray_alt_ext')
        hdr_file = base + '.hdr'
        ext = '.foo'
        img_file = base + ext
        spy.envi.save_image(hdr_file, data, ext=ext)
        rdata = spy.envi.open(hdr_file, img_file).load()
        assert (np.all(data == rdata))

    def test_save_image_spyfile(self, testdir, av3c_image):
        '''Test saving an ENVI formatted image from a SpyFile object.'''
        (r, b, c) = (3, 8, 23)
        fname = os.path.join(testdir, 'test_save_image_spyfile.hdr')
        spy.envi.save_image(fname, av3c_image)
        img = spy.open_image(fname)
        assert_almost_equal(av3c_image[r, b, c], img[r, b, c])

    def test_create_image_metadata(self, testdir):
        '''Test calling `envi.create_image` using a metadata dict.'''
        (R, B, C) = (10, 20, 30)
        (r, b, c) = (3, 8, 23)
        offset = 1024
        datum = 33
        md = {'lines': R,
              'samples': B,
              'bands': C,
              'interleave': 'bsq',
              'header offset': offset,
              'data type': 12,
              'USER DEFINED': 'test case insensitivity'}
        fname = os.path.join(testdir, 'test_create_image_metadata.hdr')
        img = spy.envi.create_image(fname, md)
        mm = img.open_memmap(writable=True)
        mm.fill(0)
        mm[r, b, c] = datum
        mm.flush()
        img = spy.open_image(fname)
        img._disable_memmap()
        assert_almost_equal(img[r, b, c], datum)
        assert (img.offset == offset)
        for key in md:
            assert key.lower() in img.metadata
            assert str(md[key]) == img.metadata[key.lower()]

    def test_create_image_keywords(self, testdir):
        '''Test calling `envi.create_image` using keyword args.'''
        (R, B, C) = (10, 20, 30)
        (r, b, c) = (3, 8, 23)
        offset = 1024
        datum = 33
        fname = os.path.join(testdir, 'test_create_image_keywords.hdr')
        img = spy.envi.create_image(fname, shape=(R, B, C),
                                    interleave='bsq',
                                    dtype=np.uint16,
                                    offset=offset)
        mm = img.open_memmap(writable=True)
        mm.fill(0)
        mm[r, b, c] = datum
        mm.flush()
        img = spy.open_image(fname)
        img._disable_memmap()
        assert_almost_equal(img[r, b, c], datum)
        assert (img.offset == offset)

    def test_save_invalid_dtype_fails(self, testdir):
        '''Should not be able to write unsupported data type to file.'''
        from spectral.io.envi import EnviDataTypeError
        a = np.random.randint(0, 200, 900).reshape((30, 30)).astype(np.int8)
        fname = os.path.join(testdir, 'invalid.hdr')
        with pytest.raises(EnviDataTypeError):
            spy.envi.save_image(fname, a)

    def test_save_load_classes(self, testdir, gt):
        '''Verify that `envi.save_classification` saves data correctly.'''
        fname = os.path.join(testdir, 'test_save_load_classes.hdr')
        spy.envi.save_classification(fname, gt, dtype=np.uint8)
        gt2 = spy.open_image(fname).read_band(0)
        assert (np.all(gt == gt2))

    def test_open_nonzero_frame_offset_fails(self, testdir, av3c_image):
        '''Opening files with nonzero frame offsets should fail.'''
        fname = os.path.join(testdir, 'test_open_nonzero_frame_offset_fails.hdr')
        spy.envi.save_image(fname, av3c_image)
        with open(fname, 'a') as fout:
            fout.write('major frame offsets = 128\n')
        with pytest.raises(spy.envi.EnviFeatureNotSupported):
            spy.envi.open(fname)

    def test_open_zero_frame_offset_passes(self, testdir, av3c_image):
        '''Files with frame offsets set to zero should open.'''
        fname = os.path.join(testdir, 'test_open_zero_frame_offset_passes.hdr')
        spy.envi.save_image(fname, av3c_image)
        with open(fname, 'a') as fout:
            fout.write('major frame offsets = 0\n')
            fout.write('minor frame offsets = {0, 0}\n')
        spy.envi.open(fname)

    def test_save_nonzero_frame_offset_fails(self, testdir, av3c_image):
        '''Opening files with nonzero frame offsets should fail.'''
        fname = os.path.join(testdir, 'test_save_nonzero_frame_offset_fails.hdr')
        meta = {'major frame offsets': [128, 0]}
        with pytest.raises(spy.envi.EnviFeatureNotSupported):
            spy.envi.save_image(fname, av3c_image, metadata=meta)

    def test_save_zero_frame_offset_passes(self, testdir, av3c_image):
        '''Opening files with nonzero frame offsets should fail.'''
        fname = os.path.join(testdir, 'test_save_zero_frame_offset_passes.hdr')
        meta = {'major frame offsets': 0}
        spy.envi.save_image(fname, av3c_image, metadata=meta)

    def test_catch_parse_error(self, testdir, av3c_image):
        '''Failure to parse parameters should raise EnviHeaderParsingError.'''
        fname = os.path.join(testdir, 'test_catch_parse_error.hdr')
        spy.envi.save_image(fname, av3c_image)
        with open(fname, 'a') as fout:
            fout.write('foo = {{\n')
        with pytest.raises(spy.envi.EnviHeaderParsingError):
            spy.envi.open(fname)

    def test_header_missing_mandatory_parameter_fails(self, testdir, av3c_image):
        '''Missing mandatory parameter should raise EnviMissingHeaderParameter.'''
        fname = os.path.join(testdir, 'test_missing_param_fails.hdr')
        spy.envi.save_image(fname, av3c_image)
        lines = [line for line in open(fname).readlines()
                 if 'bands' not in line]
        with open(fname, 'w') as fout:
            for line in lines:
                fout.write(line)
        with pytest.raises(spy.envi.MissingEnviHeaderParameter):
            spy.envi.open(fname)

    def test_param_name_converted_to_lower_case(self, testdir):
        '''By default, parameter names are converted to lower case.'''
        header = os.path.join(testdir, 'mixed_case_header1.hdr')
        open(header, 'w').write(MIXED_CASE_HEADER)
        h = spy.envi.read_envi_header(header)
        assert ('some param' in h)

    def test_support_nonlowercase_params(self, testdir):
        '''By default, parameter names are converted to lower case.'''
        header = os.path.join(testdir, 'mixed_case_header2.hdr')
        open(header, 'w').write(MIXED_CASE_HEADER)
        orig = spy.settings.envi_support_nonlowercase_params
        try:
            spy.settings.envi_support_nonlowercase_params = True
            h = spy.envi.read_envi_header(header)
        finally:
            spy.settings.envi_support_nonlowercase_params = orig
        assert ('some Param' in h)

    def test_missing_ENVI_in_header_fails(self, testdir, av3c_image):
        '''FileNotAnEnviHeader should be raised if "ENVI" not on first line.'''
        fname = os.path.join(testdir, 'test_header_missing_ENVI_fails.hdr')
        spy.envi.save_image(fname, av3c_image)
        lines = open(fname).readlines()
        with open(fname, 'w') as fout:
            for line in lines[1:]:
                fout.write(line)
        with pytest.raises(spy.envi.FileNotAnEnviHeader):
            spy.envi.open(fname)

    def test_open_missing_data_raises_envidatafilenotfounderror(self, testdir, av3c_image):
        '''EnviDataFileNotFound should be raise if data file is not found.'''
        fname = os.path.join(testdir, 'header_without_data.hdr')
        spy.envi.save_image(fname, av3c_image, ext='.img')
        os.unlink(os.path.splitext(fname)[0] + '.img')
        with pytest.raises(spy.envi.EnviDataFileNotFoundError):
            spy.envi.open(fname)

    def test_create_spectral_lib_with_header(self, testdir, av3c_image):
        '''Can create ENVI spectral library from numpy array with bands.'''
        (nrows, ncols, nbands) = av3c_image.shape
        header = {'wavelength': np.arange(nbands).astype(np.float32)}
        slib = SpectralLibrary(av3c_image[0, :20, :].squeeze(), header)
        basename = os.path.join(testdir, 'slib_with_header')
        slib.save(basename)
        slib = spy.envi.open(basename + '.hdr')
        assert (slib.spectra.shape == (20, nbands))

    def test_create_spectral_lib_without_header(self, testdir, av3c_image):
        '''Can create ENVI spectral library from numpy array without bands.'''
        (nrows, ncols, nbands) = av3c_image.shape
        slib = SpectralLibrary(av3c_image[0, :20, :].squeeze())
        basename = os.path.join(testdir, 'slib_without_header')
        slib.save(basename)
        slib = spy.envi.open(basename + '.hdr')
        assert (slib.spectra.shape == (20, nbands))
