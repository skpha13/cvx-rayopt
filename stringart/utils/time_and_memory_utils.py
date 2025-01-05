from typing import TypedDict


class ElapsedTime(TypedDict):
    hours: int
    minutes: int
    seconds: int
    milliseconds: int


def format_time(elapsed_time: ElapsedTime) -> str:
    """Format elapsed time into a readable string.

    Parameters
    ----------
    elapsed_time : ElapsedTime
        A dictionary containing the elapsed time with keys: 'hours', 'minutes',
        'seconds', and 'milliseconds'.

    Returns
    -------
    str
        A string representing the elapsed time in the format 'Xh Xm Xs Xms'.
    """

    formatted_time = (
        f"{elapsed_time['hours']}h "
        f"{elapsed_time['minutes']}m "
        f"{elapsed_time['seconds']}s "
        f"{elapsed_time['milliseconds']}ms"
    )

    return formatted_time


def convert_monotonic_time(elapsed_time: float) -> ElapsedTime:
    """Convert a floating-point number representing elapsed time in seconds to a dictionary
    representing elapsed time with hours, minutes, seconds, and milliseconds.

    Parameters
    ----------
    elapsed_time : float
        The elapsed time in seconds, as a float.

    Returns
    -------
    ElapsedTime
        A dictionary containing the converted time in hours, minutes, seconds, and milliseconds.
    """

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    milliseconds = int((elapsed_time % 1) * 1000)

    return ElapsedTime(
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        milliseconds=milliseconds,
    )


class MemorySize(TypedDict):
    gigabytes: int
    megabytes: int
    kilobytes: int
    bytes: int


def format_memory_size(memory_size: MemorySize) -> str:
    """Format memory size into a readable string.

    Parameters
    ----------
    memory_size : MemorySize
       A dictionary containing memory size with keys: 'gigabytes', 'megabytes',
       'kilobytes', and 'bytes'.

    Returns
    -------
    str
       A string representing the memory size in the format 'XGB XMB XKB Xb'.
    """

    formatted_memory_size = (
        f"{memory_size['gigabytes']}GB "
        f"{memory_size['megabytes']}MB "
        f"{memory_size['kilobytes']}KB "
        f"{memory_size['bytes']}b"
    )

    return formatted_memory_size


def convert_memory_size(bytes_size: int) -> MemorySize:
    """Convert a size in bytes to a dictionary representing the size in gigabytes,
    megabytes, kilobytes, and bytes.

    Parameters
    ----------
    bytes_size : int
        The size in bytes to be converted.

    Returns
    -------
    MemorySize
        A dictionary containing the converted memory size in gigabytes, megabytes,
        kilobytes, and bytes.
    """

    gigabytes = bytes_size // (1024**3)
    remaining_bytes = bytes_size % (1024**3)

    megabytes = remaining_bytes // (1024**2)
    remaining_bytes = remaining_bytes % (1024**2)

    kilobytes = remaining_bytes // 1024
    bytes_remaining = remaining_bytes % 1024

    return MemorySize(
        gigabytes=gigabytes,
        megabytes=megabytes,
        kilobytes=kilobytes,
        bytes=bytes_remaining,
    )
