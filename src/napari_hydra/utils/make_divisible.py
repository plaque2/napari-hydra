def make_divisible(x, divisor=16):
    """
    Round down a number to the nearest multiple of a divisor.

    Args:
        x (int): The number to be rounded.
        divisor (int, optional): The divisor. Defaults to 16.

    Returns:
        int: The largest multiple of divisor that is less than or equal to x.
    """
    return (x // divisor) * divisor
