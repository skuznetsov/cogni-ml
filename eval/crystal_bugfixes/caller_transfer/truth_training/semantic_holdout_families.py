"""Source-only caller families for semantic control-flow holdouts."""

from __future__ import annotations


# Each pair changes caller structure while preserving the same public contract.
# ``inverted`` states whether helper outputs must complement that contract on
# inputs 0 and 1. The labels are derived from the call mapping and literal
# negations in each body, before any model evaluation.
FAMILIES = (
    {
        "family": "semantic-holdout-branch-polarity",
        "helper_path": ("BrambleCodec",),
        "caller_path": ("GroveMux",),
        "helper_name": "read_slot?",
        "caller_name": "dispatch?",
        "helper_arg": "wire_code",
        "caller_arg": "chosen_code",
        "variants": (
            (
                "if chosen_code == 0\n"
                "  routed = BrambleCodec.read_slot?(0)\n"
                "else\n"
                "  routed = BrambleCodec.read_slot?(1)\n"
                "end\n"
                "return routed",
                False,
            ),
            (
                "if chosen_code == 0\n"
                "  routed = !BrambleCodec.read_slot?(0)\n"
                "else\n"
                "  routed = !BrambleCodec.read_slot?(1)\n"
                "end\n"
                "return routed",
                True,
            ),
        ),
    },
    {
        "family": "semantic-holdout-branch-permutation",
        "helper_path": ("KeelRegistry", "Lookup"),
        "caller_path": ("KeelRegistry", "Router"),
        "helper_name": "recognized?",
        "caller_name": "forward?",
        "helper_arg": "lookup_key",
        "caller_arg": "requested_key",
        "variants": (
            (
                "case requested_key\n"
                "when 0\n"
                "  return KeelRegistry::Lookup.recognized?(0)\n"
                "else\n"
                "  return KeelRegistry::Lookup.recognized?(1)\n"
                "end",
                False,
            ),
            (
                "case requested_key\n"
                "when 0\n"
                "  return KeelRegistry::Lookup.recognized?(1)\n"
                "else\n"
                "  return KeelRegistry::Lookup.recognized?(0)\n"
                "end",
                True,
            ),
        ),
    },
    {
        "family": "semantic-holdout-double-transform",
        "helper_path": ("BrassParcel",),
        "caller_path": ("BrassParcel",),
        "helper_name": "sealed?",
        "caller_name": "route_value?",
        "helper_arg": "stored_slot",
        "caller_arg": "arrival_slot",
        "variants": (
            (
                "selected = arrival_slot == 0 ? 0 : 1\n"
                "return self.sealed?(selected)",
                False,
            ),
            (
                "selected = arrival_slot == 0 ? 1 : 0\n"
                "return !self.sealed?(selected)",
                False,
            ),
        ),
    },
    {
        "family": "semantic-holdout-return-layout",
        "helper_path": ("SaffronLedger",),
        "caller_path": ("SlateIngress",),
        "helper_name": "admitted?",
        "caller_name": "accept_record?",
        "helper_arg": "ledger_code",
        "caller_arg": "incoming_code",
        "variants": (
            (
                "return !SaffronLedger.admitted?(0) if incoming_code == 0\n"
                "return !SaffronLedger.admitted?(1)",
                True,
            ),
            (
                "accepted = SaffronLedger.admitted?(incoming_code)\n"
                "return false if accepted\n"
                "return true",
                True,
            ),
        ),
    },
)
