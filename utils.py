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
