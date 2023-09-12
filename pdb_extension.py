import pdb,sys
import os

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def post_mmortem(t=None):
    # handling the default
    if t is None:
        # sys.exc_info() returns (type, value, traceback) if an exception is
        # being handled, otherwise it returns None
        t = sys.exc_info()[2]
    if t is None:
        raise ValueError("A valid traceback must be passed if no "
                         "exception is being handled")
    p = ForkedPdb()
    p.reset()
    p.interaction(None, t)

pdb.set_mtrace = ForkedPdb().set_trace
# pdb.set_trace=lambda:0
#  pdb.set_trace=pdb.set_ttrace
pdb.set_ttrace = pdb.set_trace
pdb.post_mmortem = post_mmortem