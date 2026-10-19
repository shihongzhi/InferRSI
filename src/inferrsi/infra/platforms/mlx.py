"""Apple Silicon discovery for MLX trials; no Metal performance claims."""

# MLX addresses one unified-memory Metal device per SoC, so discrete device IDs
# do not apply; only the chip identity and an optional model check are verified.


def inspect_devices(*, device_ids, device_model, command):
    from . import tool_facts

    device = tool_facts(command(["sysctl", "-n", "machdep.cpu.brand_string"]))
    probe = device["probe"]
    if probe["returncode"] != 0:
        return device
    chip = probe["stdout"].strip()
    device["chip"] = chip
    if not chip.casefold().startswith("apple"):
        device["status"] = "unavailable"
        device["reason"] = "sysctl did not report an Apple Silicon chip; MLX requires one"
    elif device_model and device_model.casefold() not in chip.casefold():
        device["status"] = "device_model_mismatch"
    else:
        device["status"] = "available"
        device["selection_verified"] = True
    return device
