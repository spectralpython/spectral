#########################################################################
#
#   ecostress.py - This file is part of the Spectral Python (SPy) package.
#
#   Copyright (C) 2010 Thomas Boggs
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
'''Code for reading and managing ECOSTRESS spectral library data.'''

from __future__ import division, print_function, unicode_literals

import itertools

from spectral.utilities.python23 import IS_PYTHON3
from .aster import AsterDatabase, Signature

if IS_PYTHON3:
    readline = lambda fin: fin.readline()
    open_file = lambda filename: open(filename, encoding='iso-8859-1')
else:
    readline = lambda fin: fin.readline().decode('iso-8859-1')
    open_file = lambda filename: open(filename)


def read_ecostress_file(filename):
    '''Reads an ECOSTRESS v1 spectrum file.'''

    lines = open_file(filename).readlines()
    if not IS_PYTHON3:
        lines = [line.decode('iso-8859-1') for line in lines]

    metaline_to_pair = lambda line: [x.strip() for x in line.split(':', 1)]

    s = Signature()

    # Read sample metadata
    for i in itertools.count():
        if lines[i].strip().startswith('Measurement'):
            break
        pair = metaline_to_pair(lines[i])
        try:
            s.sample[pair[0].lower()] = pair[1]
        except:
            print('line {}: {}'.format(i, lines[i]))
            raise

    # Read measurment metadata
    for j in itertools.count(i):
        if len(lines[j].strip()) == 0:
            break
        pair = metaline_to_pair(lines[j])
        s.measurement[pair[0].lower()] = pair[1]

    # Read signature spectrum
    pairs = []
    for line in lines[j:]:
        line = line.strip()
        if len(line) == 0:
            continue
        pair = line.split()
        nItems = len(pair)

        # Try to handle invalid values on signature lines
        if nItems == 1:
            print('single item (%s) on signature line, %s' \
                  %  (pair[0], filename))
            continue
        elif nItems > 2:
            print('more than 2 values on signature line,', filename)
            continue
        try:
            x = float(pair[0])
        except:
            print('corrupt signature line,', filename)
        if x == 0:
#           print 'Zero wavelength value', filename
            continue
        elif x < 0:
            print('Negative wavelength value,', filename)
            continue

        pairs.append(pair)

    [x, y] = [list(v) for v in zip(*pairs)]

    # Make sure wavelengths are ascending
    if float(x[0]) > float(x[-1]):
        x.reverse()
        y.reverse()
    s.x = [float(val) for val in x]
    s.y = [float(val) for val in y]
    s.measurement['first x value'] = x[0]
    s.measurement['last x value'] = x[-1]
    s.measurement['number of x values'] = len(x)

    return s

class EcostressDatabase(AsterDatabase):
    '''A relational database to manage ECOSTRESS spectral library data.'''

    @classmethod
    def create(cls, filename, data_dir=None):
        '''Creates an ECOSTRESS relational database by parsing ECOSTRESS data files.

        Arguments:

            `filename` (str):

                Name of the new sqlite database file to create.

            `data_dir` (str):

                Path to the directory containing ECOSTRESS library data files. If
                this argument is not provided, no data will be imported.

        Returns:

            An :class:`~spectral.database.EcostressDatabase` object.

        Example::

            >>> EcostressDatabase.create("ecostress.db", "./eco_data_ver1/")

        This is a class method (it does not require instantiating an
        EcostressDatabase object) that creates a new database by parsing all of the
        files in the ECOSTRESS library data directory.  Normally, this should only
        need to be called once.  Subsequently, a corresponding database object
        can be created by instantiating a new EcostressDatabase object with the
        path the database file as its argument.  For example::

            >>> from spectral.database.ecostress import EcostressDatabase
            >>> db = EcostressDatabase("~/ecostress.db")
        '''
        import os
        if os.path.isfile(filename):
            raise Exception('Error: Specified file already exists.')
        db = cls()
        db._connect(filename)
        for schema in cls.schemas:
            db.cursor.execute(schema)
        if data_dir:
            db._import_files(data_dir)
        return db

    def read_file(self, filename):
        return read_ecostress_file(filename)

    def _import_files(self, data_dir, ignore=None):
        '''Import each file from the ECOSTRESS library into the database.'''
        from glob import glob
        import numpy
        import os

        if not os.path.isdir(data_dir):
            raise Exception('Error: Invalid directory name specified.')
        if ignore is not None:
            filesToIgnore = [data_dir + '/' + f for f in ignore]
        else:
            filesToIgnore = []

        numFiles = 0
        numIgnored = 0

        sigID = 1

        class Sig:
            pass
        sigs = []

        for f in glob(data_dir + '/*spectrum.txt'):
            if f in filesToIgnore:
                numIgnored += 1
                continue
            print('Importing %s.' % f)
            numFiles += 1
            sig = self.read_file(f)
            s = sig.sample
            if 'particle size' in s:
                if s['particle size'].lower == 'liquid':
                    phase = 'liquid'
                else:
                    phase = 'solid'
            else:
                phase = 'unknown'
                s['particle size'] = 'none'
            if 'sample no.' in s:
                sampleNum = s['sample no.']
            else:
                sampleNum = ''
            subclass = s.get('subclass', 'none')
            if subclass is 'none' and 'genus' in s:
                subclass = s['genus']
            id = self._add_sample(s['name'], s['type'], s['class'], subclass,
                                  s['particle size'], sampleNum, s['owner'],
                                  s['origin'], phase, s['description'])

            instrument = os.path.basename(f).split('.')[1]
            environment = 'lab'
            m = sig.measurement

            # Correct numerous mispellings of "reflectance" and "transmittance"
            yUnit = m['y units']
            if yUnit.find('reflectence') > -1:
                yUnit = 'reflectance (percent)'
            elif yUnit.find('trans') == 0:
                yUnit = 'transmittance (percent)'
            measurement = m['measurement']
            if measurement[0] == 't':
                measurement = 'transmittance'
            self._add_signature(id, -1, instrument, environment, measurement,
                                m['x units'], yUnit, m['first x value'],
                                m['last x value'], sig.x, sig.y)
        if numFiles == 0:
            print('No data files were found in directory "%s".' \
                  % data_dir)
        else:
            print('Processed %d files.' % numFiles)
        if numIgnored > 0:
            print('Ignored the following %d bad files:' % (numIgnored))
            for f in filesToIgnore:
                print('\t' + f)

        return sigs
