#########################################################################
#
#   envi.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2013 Thomas Boggs
#
#   Spectral Python is free software; you can redistribute it and/
#   or modify it under the terms of the GNU General Public License
#   as published by the Free Software Foundation; either version 2
#   of the License, or (at your option) any later version.
#
#   Spectral Python is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this software; if not, write to
#
#               Free Software Foundation, Inc.
#               59 Temple Place, Suite 330
#               Boston, MA 02111-1307
#               USA
#
#########################################################################
#
# Send comments to:
# Thomas Boggs, tboggs@users.sourceforge.net
#
# spyfile.py
'''Runs unit tests of functions associated with the ENVI file format.

To run the unit tests, type the following from the system command line:

    # python -m spectral.tests.envi
'''

import numpy as np
from numpy.testing import assert_almost_equal
from spytest import SpyTest, test_method
from spectral.tests import testdir

class ENVIWriteTest(SpyTest):
    '''Tests that SpyFile memmap interfaces read and write properly.'''
    def __init__(self):
        pass

    def setup(self):
        import os
        if not os.path.isdir(testdir):
            os.makedirs(testdir)
        
    @test_method
    def test_save_image_ndarray(self):
        '''Test saving an ENVI formated image from a numpy.ndarray.'''
        import os
        import spectral
        (R, B, C) = (10, 20, 30)
        (r, b, c) = (3, 8, 23)
        datum = 33
        data = np.zeros((R, B, C), dtype=np.uint16)
        data[r, b, c] = datum
        fname = os.path.join(testdir, 'test_save_image_ndarray.hdr')
        spectral.envi.save_image(fname, data, interleave='bil')
        img = spectral.open_image(fname)
        assert_almost_equal(img[r, b, c], datum)

    @test_method
    def test_save_image_spyfile(self):
        '''Test saving an ENVI formatted image from a SpyFile object.'''
        import os
        import spectral
        (r, b, c) = (3, 8, 23)
        fname = os.path.join(testdir, 'test_save_image_spyfile.hdr')
        src = spectral.open_image('92AV3C.lan')
        spectral.envi.save_image(fname, src)
        img = spectral.open_image(fname)
        assert_almost_equal(src[r, b, c], img[r, b, c])

    @test_method
    def test_create_image_metadata(self):
        '''Test calling `envi.create_image` using a metadata dict.'''
        import os
        import spectral
        (R, B, C) = (10, 20, 30)
        (r, b, c) = (3, 8, 23)
        datum = 33
        md = {'lines': R,
              'samples': B,
              'bands': C,
              'data type': 12}
        fname = os.path.join(testdir, 'test_create_image_metadata.hdr')
        img = spectral.envi.create_image(fname, md)
        mm = img.open_memmap(writable=True)
        mm.fill(0)
        mm[r, b, c] = datum
        mm.flush()
        img = spectral.open_image(fname)
        assert_almost_equal(img[r, b, c], datum)

    @test_method
    def test_create_image_keywords(self):
        '''Test calling `envi.create_image` using keyword args.'''
        import os
        import spectral
        (R, B, C) = (10, 20, 30)
        (r, b, c) = (3, 8, 23)
        datum = 33
        fname = os.path.join(testdir, 'test_create_image_keywords.hdr')
        img = spectral.envi.create_image(fname, shape=(R,B,C),
                                         dtype=np.uint16,
                                         offset=120)
        mm = img.open_memmap(writable=True)
        mm.fill(0)
        mm[r, b, c] = datum
        mm.flush()
        img = spectral.open_image(fname)
        assert_almost_equal(img[r, b, c], datum)

    def run(self):
        '''Executes the test case.'''
        self.setup()
        self.test_save_image_ndarray()
        self.test_save_image_spyfile()
        self.test_create_image_metadata()
        self.test_create_image_keywords()
        self.finish()


def run():
    print '\n' + '-' * 72
    print 'Running ENVI tests.'
    print '-' * 72
    write_test = ENVIWriteTest()
    write_test.run()

if __name__ == '__main__':
    from spectral.tests.run import parse_args, reset_stats, print_summary
    parse_args()
    reset_stats()
    run()
    print_summary()
