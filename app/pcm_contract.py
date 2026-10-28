"""Offline-testable, explicit 16-bit PCM wire contract (no model imports)."""


def explicit_l16_accepted(accept):
    """Only a positive, explicit audio/L16 Accept media range opts into BE PCM."""
    for item in accept.split(","):
        media, *parameters = item.split(";")
        if media.strip().lower() != "audio/l16":
            continue
        valid = True
        for parameter in parameters:
            key, sep, value = parameter.strip().partition("=")
            key, value = key.lower(), value.strip().strip('"')
            if key == "q":
                try:
                    quality = float(value)
                    valid = valid and 0 < quality <= 1
                except ValueError:
                    valid = False
            elif key == "rate":
                valid = valid and sep == "=" and value == "16000"
            elif key == "channels":
                valid = valid and sep == "=" and value == "1"
            else:
                # Unknown parameters are not an implicit format request.
                valid = False
        if valid:
            return True
    return False


def pcm_le_to_l16(raw):
    """Convert signed PCM16 little-endian bytes into big-endian L16 bytes."""
    if len(raw) % 2:
        raise ValueError("Invalid PCM sample width")
    result = bytearray(len(raw))
    result[::2] = raw[1::2]
    result[1::2] = raw[::2]
    return bytes(result)
