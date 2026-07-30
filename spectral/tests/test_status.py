'''Tests of `StatusDisplay`, used to show progress of long-running
iterative algorithms (e.g., `kmeans`, `ppi`).

Every `StatusDisplay` method is a no-op unless running in an interactive
interpreter (`hasattr(sys, 'ps1')`) with `spectral.settings.show_progress`
enabled, so most tests here monkeypatch `sys.ps1` to force the
interactive branch and capture stdout via `capsys`.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_status.py
'''

import sys

import pytest

from spectral import settings
from spectral.utilities.status import StatusDisplay


@pytest.fixture
def status():
    return StatusDisplay()


@pytest.fixture
def interactive(monkeypatch):
    '''Forces the interactive/`show_progress` branch that all
    `StatusDisplay` methods otherwise skip.'''
    monkeypatch.setattr(sys, 'ps1', '>>>', raising=False)
    monkeypatch.setattr(settings, 'show_progress', True)


class TestNonInteractive:
    '''Outside an interactive interpreter, every method should be a no-op,
    regardless of `show_progress`.'''

    def test_no_output_without_ps1(self, status, capsys, monkeypatch):
        monkeypatch.delattr(sys, 'ps1', raising=False)
        monkeypatch.setattr(settings, 'show_progress', True)
        status.display_percentage('Working...')
        status.update_percentage(50.0)
        status.end_percentage()
        status.write('a message\n')
        assert capsys.readouterr().out == ''

    def test_no_output_when_show_progress_disabled(self, status, capsys,
                                                    monkeypatch):
        monkeypatch.setattr(sys, 'ps1', '>>>', raising=False)
        monkeypatch.setattr(settings, 'show_progress', False)
        status.display_percentage('Working...')
        status.update_percentage(50.0)
        status.end_percentage()
        status.write('a message\n')
        assert capsys.readouterr().out == ''


class TestInteractive:
    '''With `sys.ps1` present and `show_progress` enabled, methods should
    write to stdout.'''

    def test_display_percentage_writes_pretext_and_percent(self, status,
                                                            interactive,
                                                            capsys):
        status.display_percentage('Working...', 0.0)
        out = capsys.readouterr().out
        assert out == 'Working...' + '% 5.1f' % 0.0 + '%'

    def test_display_percentage_custom_format(self, status, interactive,
                                              capsys):
        status.display_percentage('Working...', 5.0, format='%d')
        out = capsys.readouterr().out
        assert out == 'Working...5%'

    def test_update_percentage_backspaces_previous_text(self, status,
                                                        interactive, capsys):
        status.display_percentage('Working...', 0.0)
        capsys.readouterr()  # discard display_percentage's output

        status.update_percentage(50.0)
        out = capsys.readouterr().out
        prev_len = len('Working...' + '% 5.1f' % 0.0 + '%')
        expected_text = 'Working...' + '% 5.1f' % 50.0 + '%'
        assert out == '\b' * prev_len + expected_text

    def test_end_percentage_writes_final_text_and_resets_overwrite(
            self, status, interactive, capsys):
        status.display_percentage('Working...', 0.0)
        prev_len = len(capsys.readouterr().out)

        status.end_percentage('done.')
        out = capsys.readouterr().out
        expected_text = 'Working...done.'
        fmt = '%%-%ds\n' % prev_len
        assert out == '\b' * prev_len + fmt % expected_text
        assert status._overwrite is False

    def test_end_percentage_default_text(self, status, interactive, capsys):
        status.display_percentage('Working...', 0.0)
        capsys.readouterr()

        status.end_percentage()
        out = capsys.readouterr().out
        assert 'Working...done' in out

    def test_write_inserts_newline_when_overwriting_progress(
            self, status, interactive, capsys):
        status.display_percentage('Working...', 0.0)
        capsys.readouterr()

        status.write('a log message')
        out = capsys.readouterr().out
        assert out == '\na log message'

    def test_write_does_not_insert_newline_when_not_overwriting(
            self, status, interactive, capsys):
        status.write('a log message')
        out = capsys.readouterr().out
        assert out == 'a log message'

    def test_write_newline_does_not_double_up(self, status, interactive,
                                              capsys):
        status.display_percentage('Working...', 0.0)
        capsys.readouterr()

        status.write('\n')
        out = capsys.readouterr().out
        assert out == '\n'

    def test_write_does_not_reset_overwrite(self, status, interactive,
                                            capsys):
        status.display_percentage('Working...', 0.0)
        status.write('a log message')
        assert status._overwrite is True
