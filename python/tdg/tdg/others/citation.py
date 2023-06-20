import logging
logger = logging.getLogger(__name__)
from collections import deque

from pathlib import Path
bibtex_file=Path(f'{__file__}').parents[4] / 'master.bib'
logger.info(bibtex_file)

from pathlib import Path
from pybtex.database.input import bibtex
bibtex_file=Path(f'{__file__}').parents[4] / 'master.bib'
_parser = bibtex.Parser()
_bibdata = _parser.parse_file(bibtex_file)

class Citation:
    r'''
    A Citation is a collection bunch of bibliographic information.

    On each call log the short name and a message if the message hasn't been logged before; the first time it is called log the bibtex entry.

    Parameters
    ----------
        short: string
            A plain-text rendering of the citation.
        bibtex: string
            A bibtex key or complete bibtex entry that could be used to cite the reference.
            The key will be looked up in tdg's master bibtex file.
    '''

    def __init__(self, short, bibtex):
        self.short = short
        try:
            self.bibtex_key = ' ' + bibtex + ' '
            self.bibtex = _bibdata.entries[bibtex].to_string('bibtex')
        except KeyError:
            self.bibtex_key = ''
            self.bibtex = bibtex
        self.cited = False
        self.messages = deque()

    def __call__(self, message=''):
        if not self.cited:
            logger.info(f'{self.bibtex_key}\n{self.short}\n{message}\n')
            logger.info(self.bibtex)
            self.cited = True
        elif message not in self.messages:
            logger.info(f'{self.bibtex_key} {message}')
        else:
            return
        self.messages.append(message)

