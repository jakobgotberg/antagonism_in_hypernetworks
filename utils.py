import signal

class TimeoutExpired(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutExpired

def start_timer(seconds: float) -> None:
    signal.setitimer(signal.ITIMER_REAL, seconds)

def cancel_timer() -> None:
    signal.setitimer(signal.ITIMER_REAL, 0)

def set_signal(): 
    signal.signal(signal.SIGALRM, _timeout_handler)

def feedback(msg, verbose):
    if not verbose:
        return
    print(msg)

def comb(n, k):
    '''
    Binom(n,k).
    Python 3.6 does have math.comb.
    '''
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    return result


