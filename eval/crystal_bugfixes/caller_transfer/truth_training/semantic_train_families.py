"""Authored caller transformations for a fresh semantic SFT control."""

FAMILIES = (
    {
        "family": "semantic-train-xor-flip",
        "helper_path": ("FlintWire",),
        "caller_path": ("FlintRoute",),
        "helper_name": "read_wire?",
        "caller_name": "forward_wire?",
        "helper_arg": "wire_value",
        "caller_arg": "incoming_wire",
        "variants": (
            ("return FlintWire.read_wire?(incoming_wire)", False),
            ("return FlintWire.read_wire?(incoming_wire ^ 1)", True),
        ),
    },
    {
        "family": "semantic-train-composed-flip",
        "helper_path": ("MarbleTag", "Codec"),
        "caller_path": ("MarbleTag", "Dispatch"),
        "helper_name": "valid?",
        "caller_name": "emit_tag?",
        "helper_arg": "tag_value",
        "caller_arg": "received_tag",
        "variants": (
            ("return MarbleTag::Codec.valid?(received_tag ^ 1) == true", True),
            ("return MarbleTag::Codec.valid?(received_tag ^ 1) == false", False),
        ),
    },
    {
        "family": "semantic-train-direct-invariance",
        "helper_path": ("LindenLedger",),
        "caller_path": ("LindenLedger",),
        "helper_name": "marked?",
        "caller_name": "note?",
        "helper_arg": "mark_value",
        "caller_arg": "posted_mark",
        "variants": (
            ("return self.marked?(posted_mark ^ 0)", False),
            ("return self.marked?(posted_mark + 0)", False),
        ),
    },
    {
        "family": "semantic-train-inverse-invariance",
        "helper_path": ("RuneArchive",),
        "caller_path": ("RuneIngress",),
        "helper_name": "filtered?",
        "caller_name": "ingest?",
        "helper_arg": "stored_code",
        "caller_arg": "submitted_code",
        "variants": (
            ("return RuneArchive.filtered?(submitted_code) == false", True),
            ("return RuneArchive.filtered?(submitted_code) != true", True),
        ),
    },
)
