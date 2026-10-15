"""Fail-closed authentication/bind contract without socket or model imports.

The web server is accessible without a key only when it binds directly to a
numeric loopback IP. Do not rely on DNS names, proxy headers or host port
publishing to make an unauthenticated wildcard listener safe.
"""
import ipaddress


def validate_auth_bind(host, require_api_key):
    """Reject unauthenticated network-facing or ambiguous bind addresses.

    This is deliberately a configuration check, not a network probe. It must
    run before model initialization and before the WSGI listener is created.
    """
    if require_api_key:
        return
    try:
        address = ipaddress.ip_address(host)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Unauthenticated server requires an explicit numeric loopback bind"
        ) from exc
    # IPv4-mapped loopback (::ffff:127.0.0.1) is not an explicit numeric bind.
    if getattr(address, "ipv4_mapped", None) is not None or not address.is_loopback:
        raise ValueError(
            "Unauthenticated server requires an explicit numeric loopback bind"
        )
