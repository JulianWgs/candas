_options = {
    "interpolation.on": None,
    "interpolation.how": None,
}


def _set_option(pat, val):
    _options[pat] = val


def _get_option(pat):
    return _options[pat]
