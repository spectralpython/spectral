

class SpectralDatabase:
    def _connect(self, sqlite_filename):
        '''Establishes a connection to the Specbase sqlite database.'''
        import sqlite3
        self.db = sqlite3.connect(sqlite_filename)
        self.cursor = self.db.cursor()

    def query(self, sql, args=None):
        '''Returns the result of an arbitrary SQL statement.

        Arguments:

            `sql` (str):

                An SQL statement to be passed to the database. Use "?" for
                variables passed into the statement.

            `args` (tuple):

                Optional arguments which will replace the "?" placeholders in
                the `sql` argument.

        Returns:

            An :class:`sqlite3.Cursor` object with the query results.

        Example::

            >>> sql = r'SELECT SpectrumID, Name FROM Samples, Spectra ' +
            ...        'WHERE Spectra.SampleID = Samples.SampleID ' +
            ...        'AND Name LIKE "%grass%" AND MinWavelength < ?'
            >>> args = (0.5,)
            >>> cur = db.query(sql, args)
            >>> for row in cur:
            ...     print row
            ...
            (356, u'dry grass')
            (357, u'grass')
        '''
        if args:
            return self.cursor.execute(sql, args)
        else:
            return self.cursor.execute(sql)

    def print_query(self, sql, args=None):
        '''Prints the text result of an arbitrary SQL statement.

        Arguments:

            `sql` (str):

                An SQL statement to be passed to the database. Use "?" for
                variables passed into the statement.

            `args` (tuple):

                Optional arguments which will replace the "?" placeholders in
                the `sql` argument.

        This function performs the same query function as
        :meth:`spectral.database.SpectralDatabase.query` except query results are
        printed to **stdout** instead of returning a cursor object.

        Example:

            >>> sql = r'SELECT SpectrumID, Name FROM Samples, Spectra ' +
            ...        'WHERE Spectra.SampleID = Samples.SampleID ' +
            ...        'AND Name LIKE "%grass%" AND MinWavelength < ?'
            >>> args = (0.5,)
            >>> db.print_query(sql, args)
            356|dry grass
            357|grass
        '''
        ret = self.query(sql, args)
        for row in ret:
            print("|".join([str(x) for x in row]))
