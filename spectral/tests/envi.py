'''
Runs unit tests of functions associated with the ENVI file format.

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.envi
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy.testing import assert_almost_equal
import os

import spectral as spy
from spectral.io.envi import SpectralLibrary
from spectral.tests import testdir
from spectral.tests.spytest import SpyTest

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


class ENVIWriteTest(SpyTest):
    '''Tests that SpyFile memmap interfaces read and write properly.'''
    def __init__(self):
        pass

    def setup(self):
        if not os.path.isdir(testdir):
            os.makedirs(testdir)

    def test_save_image_ndarray(self):
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

    def test_save_image_ndarray_no_ext(self):
        '''Test saving an ENVI formatted image with no image file extension.'''
        data = np.arange(1000, dtype=np.int16).reshape(10, 10, 10)
        base = os.path.join(testdir, 'test_save_image_ndarray_noext')
        hdr_file = base + '.hdr'
        spy.envi.save_image(hdr_file, data, ext='')
        rdata = spy.open_image(hdr_file).load()
        assert (np.all(data == rdata))

    def test_save_image_ndarray_alt_ext(self):
        '''Test saving an ENVI formatted image with alternate extension.'''
        data = np.arange(1000, dtype=np.int16).reshape(10, 10, 10)
        base = os.path.join(testdir, 'test_save_image_ndarray_alt_ext')
        hdr_file = base + '.hdr'
        ext = '.foo'
        img_file = base + ext
        spy.envi.save_image(hdr_file, data, ext=ext)
        rdata = spy.envi.open(hdr_file, img_file).load()
        assert (np.all(data == rdata))

    def test_save_image_spyfile(self):
        '''Test saving an ENVI formatted image from a SpyFile object.'''
        (r, b, c) = (3, 8, 23)
        fname = os.path.join(testdir, 'test_save_image_spyfile.hdr')
        src = spy.open_image('92AV3C.lan')
        spy.envi.save_image(fname, src)
        img = spy.open_image(fname)
        assert_almost_equal(src[r, b, c], img[r, b, c])

    def test_create_image_metadata(self):
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

    def test_create_image_keywords(self):
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

    def test_save_invalid_dtype_fails(self):
        '''Should not be able to write unsupported data type to file.'''
        from spectral.io.envi import EnviDataTypeError
        a = np.random.randint(0, 200, 900).reshape((30, 30)).astype(np.int8)
        try:
            spy.envi.save_image('invalid.hdr', a)
        except EnviDataTypeError:
            pass
        else:
            raise Exception('Expected EnviDataTypeError to be raised.')

    def test_save_load_classes(self):
        '''Verify that `envi.save_classification` saves data correctly.'''
        fname = os.path.join(testdir, 'test_save_load_classes.hdr')
        gt = spy.open_image('92AV3GT.GIS').read_band(0)
        spy.envi.save_classification(fname, gt, dtype=np.uint8)
        gt2 = spy.open_image(fname).read_band(0)
        assert (np.all(gt == gt2))

    def test_open_nonzero_frame_offset_fails(self):
        '''Opening files with nonzero frame offsets should fail.'''
        img = spy.open_image('92AV3C.lan')
        fname = os.path.join(testdir, 'test_open_nonzero_frame_offset_fails.hdr')
        spy.envi.save_image(fname, img)
        fout = open(fname, 'a')
        fout.write('major frame offsets = 128\n')
        fout.close()
        try:
            spy.envi.open(fname)
        except spy.envi.EnviFeatureNotSupported:
            pass
        else:
            raise Exception('File erroneously opened.')

    def test_open_zero_frame_offset_passes(self):
        '''Files with frame offsets set to zero should open.'''
        img = spy.open_image('92AV3C.lan')
        fname = os.path.join(testdir, 'test_open_zero_frame_offset_passes.hdr')
        spy.envi.save_image(fname, img)
        fout = open(fname, 'a')
        fout.write('major frame offsets = 0\n')
        fout.write('minor frame offsets = {0, 0}\n')
        fout.close()
        spy.envi.open(fname)

    def test_save_nonzero_frame_offset_fails(self):
        '''Opening files with nonzero frame offsets should fail.'''
        img = spy.open_image('92AV3C.lan')
        fname = os.path.join(testdir, 'test_save_nonzero_frame_offset_fails.hdr')
        meta = {'major frame offsets': [128, 0]}
        try:
            spy.envi.save_image(fname, img, metadata=meta)
        except spy.envi.EnviFeatureNotSupported:
            pass
        else:
            raise Exception('File erroneously saved.')

    def test_save_zero_frame_offset_passes(self):
        '''Opening files with nonzero frame offsets should fail.'''
        img = spy.open_image('92AV3C.lan')
        fname = os.path.join(testdir, 'test_save_zero_frame_offset_passes.hdr')
        meta = {'major frame offsets': 0}
        spy.envi.save_image(fname, img, metadata=meta)

    def test_catch_parse_error(self):
        '''Failure to parse parameters should raise EnviHeaderParsingError.'''
        img = spy.open_image('92AV3C.lan')
        fname = os.path.join(testdir, 'test_catch_parse_error.hdr')
        spy.envi.save_image(fname, img)
        fout = open(fname, 'a')
        fout.write('foo = {{\n')
        fout.close()
        try:
            spy.envi.open(fname)
        except spy.envi.EnviHeaderParsingError:
            pass
        else:
            raise Exception('Failed to raise EnviHeaderParsingError')

    def test_header_missing_mandatory_parameter_fails(self):
        '''Missing mandatory parameter should raise EnviMissingHeaderParameter.'''
        img = spy.open_image('92AV3C.lan')
        fname = os.path.join(testdir, 'test_missing_param_fails.hdr')
        spy.envi.save_image(fname, img)
        lines = [line for line in open(fname).readlines()
                 if 'bands' not in line]
        fout = open(fname, 'w')
        for line in lines:
            fout.write(line)
        fout.close()
        try:
            spy.envi.open(fname)
        except spy.envi.MissingEnviHeaderParameter:
            pass
        else:
            raise Exception('Failed to raise EnviMissingHeaderParameter')

    def test_param_name_converted_to_lower_case(self):
        '''By default, parameter names are converted to lower case.'''
        header = 'mixed_case_header.hdr'
        open(header, 'w').write(MIXED_CASE_HEADER)
        h = spy.envi.read_envi_header(header)
        assert ('some param' in h)

    def test_support_nonlowercase_params(self):
        '''By default, parameter names are converted to lower case.'''
        header = 'mixed_case_header.hdr'
        open(header, 'w').write(MIXED_CASE_HEADER)
        orig = spy.settings.envi_support_nonlowercase_params
        try:
            spy.settings.envi_support_nonlowercase_params = True
            h = spy.envi.read_envi_header(header)
        finally:
            spy.settings.envi_support_nonlowercase_params = orig
        assert ('some Param' in h)

    def test_missing_ENVI_in_header_fails(self):
        '''FileNotAnEnviHeader should be raised if "ENVI" not on first line.'''
        img = spy.open_image('92AV3C.lan')
        fname = os.path.join(testdir, 'test_header_missing_ENVI_fails.hdr')
        spy.envi.save_image(fname, img)
        lines = open(fname).readlines()
        fout = open(fname, 'w')
        for line in lines[1:]:
            fout.write(line)
        fout.close()
        try:
            spy.envi.open(fname)
        except spy.envi.FileNotAnEnviHeader:
            pass
        else:
            raise Exception('Failed to raise EnviMissingHeaderParameter')

    def test_open_missing_data_raises_envidatafilenotfounderror(self):
        '''EnviDataFileNotFound should be raise if data file is not found.'''
        img = spy.open_image('92AV3C.lan')
        fname = os.path.join(testdir, 'header_without_data.hdr')
        spy.envi.save_image(fname, img, ext='.img')
        os.unlink(os.path.splitext(fname)[0] + '.img')
        try:
            spy.envi.open(fname)
        except spy.envi.EnviDataFileNotFoundError:
            pass
        else:
            raise Exception('Expected EnviDataFileNotFoundError')

    def test_create_spectral_lib_with_header(self):
        '''Can create ENVI spectral library from numpy array with bands.'''
        img = spy.open_image('92AV3C.lan')
        (nrows, ncols, nbands) = img.shape
        header = {'wavelength': np.arange(nbands).astype(np.float32)}
        slib = SpectralLibrary(img[0, :20, :].squeeze(), header)
        basename = os.path.join(testdir, 'slib')
        slib.save(basename)
        slib = spy.envi.open(basename + '.hdr')
        assert (slib.spectra.shape == (20, nbands))

    def test_create_spectral_lib_without_header(self):
        '''Can create ENVI spectral library from numpy array without bands.'''
        img = spy.open_image('92AV3C.lan')
        (nrows, ncols, nbands) = img.shape
        slib = SpectralLibrary(img[0, :20, :].squeeze())
        basename = os.path.join(testdir, 'slib')
        slib.save(basename)
        slib = spy.envi.open(basename + '.hdr')
        assert (slib.spectra.shape == (20, nbands))


def run():
    print('\n' + '-' * 72)
    print('Running ENVI tests.')
    print('-' * 72)
    write_test = ENVIWriteTest()
    write_test.run()


if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
