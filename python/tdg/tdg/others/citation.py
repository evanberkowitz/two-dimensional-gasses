import logging
logger = logging.getLogger(__name__)

class Citation:
    r'''
    A Citation is a collection bunch of bibliographic information.

    The first time it is called, log the short name at the info level and the bibtex information at the debug level.

    Parameters
    ----------
        short: string
            A plain-text rendering of the citation.
        bibtex: string
            A bibtex item that could be used to cite the reference.
    '''

    def __init__(self, short, bibtex):
        self.short = short
        self.bibtex = bibtex
        self.cited = False

    def __call__(self):
        if not self.cited:
            logger.info(f'Using a result from {self.short}')
            logger.debug(self.bibtex)
            self.cited = True

