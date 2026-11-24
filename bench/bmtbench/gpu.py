"""Shared-GPU etiquette: wait for head-room, stay out of an optional quiet window, cap VRAM."""
import datetime, os, subprocess, time, zoneinfo

def _blackout():
    """optional quiet window of the host, BMT_GPU_BLACKOUT="HH:MM-HH:MM" on weekdays in the BMT_GPU_TZ time zone (default UTC)."""
    w = os.environ.get("BMT_GPU_BLACKOUT", "")
    if not w:
        return None
    a, b = (tuple(int(x) for x in p.split(":")) for p in w.split("-"))
    return a, b


def in_blackout(now=None):
    w = _blackout()
    if not w:
        return False
    n = now or datetime.datetime.now(zoneinfo.ZoneInfo(os.environ.get("BMT_GPU_TZ", "UTC")))
    return n.weekday() < 5 and w[0] <= (n.hour, n.minute) < w[1]


def free_mb():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    t, u = (int(x) for x in o.strip().splitlines()[0].split(","))
    return t - u


def wait_for_gpu(need_mb=3000, poll=30):
    """block until the GPU has need_mb free and we are outside the blackout window."""
    while in_blackout() or free_mb() < need_mb:
        time.sleep(poll)


def cap_torch(gb=None):
    import torch
    if torch.cuda.is_available():
        gb = gb or float(os.environ.get("BMT_VRAM_GB", "1.0"))
        torch.cuda.set_per_process_memory_fraction(gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)


def pause_if_needed(min_free_mb=1500, poll=30):
    """called inside long GPU loops: yield (release the allocator cache and sleep) during the quiet window or when the card is nearly full.
    Returns the seconds spent paused."""
    t0 = time.time()
    while in_blackout() or free_mb() < min_free_mb:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        time.sleep(poll)
    return time.time() - t0


CPU_QUIET_ET = (((9, 0), (10, 0)), ((15, 20), (16, 15)))


def cpu_quiet(now=None):
    n = now or datetime.datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    return n.weekday() < 5 and any(a <= (n.hour, n.minute) < b for a, b in CPU_QUIET_ET)


def cpu_gate(max_load=None, poll=30):
    """CPU runs on a shared host: wait out the weekday ET quiet windows and a 1-minute load average above BMT_CPU_LOAD (default 45)."""
    max_load = max_load or float(os.environ.get("BMT_CPU_LOAD", "45"))
    t0 = time.time()
    while cpu_quiet() or os.getloadavg()[0] > max_load:
        time.sleep(poll)
    return time.time() - t0
