from multiprocessing import Pool
import time
import signal


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_mp(function, args, ncpus):
    pool = Pool(ncpus, init_worker)
    try:
        rs = pool.map_async(function, args, chunksize=1)
        while (True):
            if rs.ready():
                break
            print rs._number_left * rs._chunksize, "tasks to complete..."
            time.sleep(1)
    except Exception, e:
        print e
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()

    return rs.get()
