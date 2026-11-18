"""Balanced synthetic truth-table tasks for Crystal caller transfer."""

from __future__ import annotations

from hashlib import sha256


_SYSTEM = "Infer the helper's Boolean outputs; do not patch, use tools, or retry."

# Sorting by a seeded digest gives a reproducible shuffle without relying on a
# process-global PRNG or Python's random implementation.
_ORDER_SEEDS = {"train": "truth-train-v1", "holdout": "truth-holdout-v1"}

_FAMILIES = {
    "train": (
        {
            "family": "train-amber",
            "layout": "separate",
            "helper_path": ("AmberSupport",),
            "caller_path": ("StreamPolicy",),
            "helper_name": "decode?",
            "caller_name": "route?",
            "helper_arg": "sample",
            "caller_arg": "requested",
        },
        {
            "family": "train-cobalt",
            "layout": "shared",
            "helper_path": ("CobaltLedger",),
            "caller_path": ("CobaltLedger",),
            "helper_name": "mark?",
            "caller_name": "record?",
            "helper_arg": "marker",
            "caller_arg": "entry",
        },
        {
            "family": "train-quartz",
            "layout": "nested",
            "helper_path": ("Quartz", "Vault"),
            "caller_path": ("Quartz", "Dispatch"),
            "helper_name": "stamp?",
            "caller_name": "apply?",
            "helper_arg": "token",
            "caller_arg": "candidate",
        },
        {
            "family": "train-cedar",
            "layout": "shared",
            "helper_path": ("CedarCycle",),
            "caller_path": ("CedarCycle",),
            "helper_name": "phase?",
            "caller_name": "join?",
            "helper_arg": "phase_code",
            "caller_arg": "join_code",
        },
    ),
    "holdout": (
        {
            "family": "holdout-mosaic",
            "layout": "separate",
            "helper_path": ("MosaicSupport",),
            "caller_path": ("KernelFlow",),
            "helper_name": "inspect?",
            "caller_name": "emit?",
            "helper_arg": "sampled",
            "caller_arg": "requested_value",
        },
        {
            "family": "holdout-silver",
            "layout": "shared",
            "helper_path": ("SilverVault",),
            "caller_path": ("SilverVault",),
            "helper_name": "fold?",
            "caller_name": "transfer?",
            "helper_arg": "numbered_mark",
            "caller_arg": "transfer_code",
        },
        {
            "family": "holdout-velvet",
            "layout": "nested",
            "helper_path": ("Velvet", "Index"),
            "caller_path": ("Velvet", "Ingress"),
            "helper_name": "resolve?",
            "caller_name": "accept?",
            "helper_arg": "index_code",
            "caller_arg": "incoming_code",
        },
        {
            "family": "holdout-pine",
            "layout": "shared",
            "helper_path": ("PineBridge",),
            "caller_path": ("PineBridge",),
            "helper_name": "unpack?",
            "caller_name": "connect?",
            "helper_arg": "decoded_value",
            "caller_arg": "applied_value",
        },
    ),
}

_QUERY_ORDERS = ((0, 1), (1, 0))


def _module_name(path: tuple[str, ...]) -> str:
    return "::".join(path)


def _method(
    name: str,
    argument: str,
    *,
    body: str,
    indent: int,
) -> list[str]:
    pad = " " * indent
    inner = " " * (indent + 2)
    return [
        f"{pad}def self.{name}({argument} : Int32) : Bool",
        f"{inner}{body}",
        f"{pad}end",
    ]


def _module_block(path: tuple[str, ...], methods: list[list[str]]) -> str:
    lines: list[str] = []
    for depth, name in enumerate(path):
        lines.append(f"{' ' * (depth * 2)}module {name}")
    for method in methods:
        lines.extend(method)
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    for depth in reversed(range(len(path))):
        lines.append(f"{' ' * (depth * 2)}end")
    return "\n".join(lines)


def _render_source(family: dict, inverted: bool) -> str:
    helper_path = family["helper_path"]
    caller_path = family["caller_path"]
    if helper_path == caller_path:
        helper_reference = f"self.{family['helper_name']}"
    else:
        helper_reference = (
            f"{_module_name(helper_path)}.{family['helper_name']}"
        )
    negation = "!" if inverted else ""
    caller_body = (
        f"return {negation}{helper_reference}({family['caller_arg']})"
    )
    caller_method = _method(
        family["caller_name"],
        family["caller_arg"],
        body=caller_body,
        indent=len(caller_path) * 2,
    )
    helper_method = _method(
        family["helper_name"],
        family["helper_arg"],
        body="<HOLE>",
        indent=len(helper_path) * 2,
    )

    if family["layout"] == "shared":
        # The caller precedes the helper in two families to vary ordinary source
        # layout without changing the visible transfer question.
        methods = [caller_method, helper_method]
        return _module_block(caller_path, methods) + "\n"

    return (
        _module_block(helper_path, [helper_method])
        + "\n\n"
        + _module_block(caller_path, [caller_method])
        + "\n"
    )


def _public_value(public_zero: bool, input_value: int) -> bool:
    return public_zero if input_value == 0 else not public_zero


def _truth_for_order(
    public_zero: bool, inverted: bool, order: tuple[int, int]
) -> list[bool]:
    values = []
    for input_value in order:
        public_value = _public_value(public_zero, input_value)
        values.append(not public_value if inverted else public_value)
    return values


def _identifiers(family: dict) -> list[str]:
    symbols = (
        *family["helper_path"],
        *family["caller_path"],
        family["helper_name"],
        family["caller_name"],
        family["helper_arg"],
        family["caller_arg"],
    )
    return list(dict.fromkeys(symbols))


def _messages(
    *,
    family: dict,
    source: str,
    public_zero: bool,
    contract_order: tuple[int, int],
    query_order: tuple[int, int],
) -> list[dict[str, str]]:
    call = f"{_module_name(family['caller_path'])}.{family['caller_name']}"
    contract = []
    for input_value in contract_order:
        result = str(_public_value(public_zero, input_value)).lower()
        contract.append(f"- `{call}({input_value})` returns `{result}`.")
    ordered_values = ", ".join(str(value) for value in query_order)
    helper_call = (
        f"{_module_name(family['helper_path'])}.{family['helper_name']}"
    )
    user = (
        "A Crystal caller must satisfy this public contract:\n"
        + "\n".join(contract)
        + "\n\nCaller and helper template:\n"
        + "```crystal\n"
        + source.rstrip()
        + "\n```\n\n"
        + f"What Boolean must `{helper_call}` return for inputs "
        + f"[{ordered_values}], in that order, so the caller satisfies the "
        + "contract? Answer with exactly two comma-separated lowercase "
        + "Boolean literals in that order, without brackets."
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def _example(
    *,
    split: str,
    family_index: int,
    family: dict,
    public_zero: bool,
    inverted: bool,
    order: tuple[int, int],
    contract_order: tuple[int, int],
) -> dict:
    source = _render_source(family, inverted)
    if source.count("<HOLE>") != 1:
        raise AssertionError("source must contain exactly one helper hole")
    polarity_id = "z1" if public_zero else "z0"
    sign_id = "inverse" if inverted else "direct"
    order_id = "01" if order == (0, 1) else "10"
    identifier = (
        f"{split}-{family_index}-{polarity_id}-{sign_id}-{order_id}"
    )
    return {
        "id": identifier,
        "family": family["family"],
        "public_zero": public_zero,
        "inverted": inverted,
        "order": list(order),
        "messages": _messages(
            family=family,
            source=source,
            public_zero=public_zero,
            contract_order=contract_order,
            query_order=order,
        ),
        "truth": _truth_for_order(public_zero, inverted, order),
        "source": source,
        "call": f"{_module_name(family['caller_path'])}.{family['caller_name']}",
        "helper_arg": family["helper_arg"],
        "identifiers": _identifiers(family),
    }


def examples(split: str) -> list[dict]:
    """Return the deterministic, shuffled 32-example train or holdout split.

    Each split has four authored families crossed with caller negation, query
    order, and public truth polarity. Contract statement order is independently
    balanced by family and polarity, and shared by both matched pair types.
    """
    if split not in _FAMILIES:
        raise ValueError("split must be 'train' or 'holdout'")

    families = _FAMILIES[split]
    split_offset = 0 if split == "train" else 1
    rows = []
    for family_index, family in enumerate(families):
        for public_zero in (False, True):
            polarity_index = int(public_zero)
            # This depends on family and public polarity, never caller sign or
            # query order. Across four families both statement orders occur
            # twice for each polarity.
            reverse_contract = (
                family_index + polarity_index + split_offset
            ) % 2 == 1
            contract_order = (1, 0) if reverse_contract else (0, 1)
            for inverted in (False, True):
                for order in _QUERY_ORDERS:
                    rows.append(
                        _example(
                            split=split,
                            family_index=family_index,
                            family=family,
                            public_zero=public_zero,
                            inverted=inverted,
                            order=order,
                            contract_order=contract_order,
                        )
                    )

    seed = _ORDER_SEEDS[split]
    rows.sort(
        key=lambda row: sha256(f"{seed}:{row['id']}".encode("utf-8")).digest()
    )
    return rows
