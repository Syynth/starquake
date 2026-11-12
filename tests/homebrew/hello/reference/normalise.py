#!/usr/bin/env python3
"""Normalises a reference emulator's trace of the hello fixture into `startup.trace`.

The reference emulator (Eden-NX2, see README.md in this folder) logs every system call twice,
with the raw registers:

    [   0.681243] Kernel.SVC <Info> svc 0x02 in  t84 [0]=0x80c05000 [1]=0x1000 [2]=0x1 [3]=0x3
    [   0.681249] Kernel.SVC <Info> svc 0x02 out t84 [0]=0x0 [1]=0x1000 [2]=0x1

plus one line per IPC request (`Service <Info> ipc sm:::0 (Initialize)`) and one per line the
guest logs (`Debug.Emulated <Info> ...`). Its addresses are randomised, its timestamps and
thread ids differ from run to run, and the registers a call does not take hold whatever was in
them. This script keeps what a second emulator can be compared with:

- one line per system call, with the call's name, its arguments and its outputs decoded with
  the kernel's register convention (`crates/kernel/src/svc/reference/svc_wrappers64.txt`), so
  registers that are not arguments are gone;
- no timestamps and no thread ids (a trace with more than one thread is refused);
- every address replaced by `anchor+offset`, where the anchor is the start of the region the
  address lies in: the module (`main`) or one of the regions `GetInfo` reported in the same
  trace (`aslr_region`, `stack_region`, `alias_region`, `heap_region`).

Which arguments and outputs are addresses is listed in `ADDRESSES` below; a call that is not
listed there is refused, so nothing is passed through unmasked by oversight.

Usage, from the repository root:

    tests/homebrew/hello/reference/normalise.py <raw startup.txt>

The module's base is not in the trace, so it is worked out from the fixture
(`tests/homebrew/hello/hello.nro`, or `--nro`): at start-up libnx asks `QueryMemory` about
`__argdata__`, the first address after its image (libnx/nx/source/runtime/argv.c), so the
highest module address the trace names is the module's base plus the NRO's image size (the
word at 0x18 of the file). The script checks that the base is then page aligned and that
every other module address lies inside the image.
"""

import argparse
import pathlib
import re
import struct
import sys

REPO = pathlib.Path(__file__).resolve().parents[4]
REFERENCE = REPO / "crates/kernel/src/svc/reference"
FIXTURE = REPO / "tests/homebrew/hello/hello.nro"
# Where an NRO's header keeps the size of its image (`NroHeader::size`).
NRO_IMAGE_SIZE_OFFSET = 0x18

# The arguments and outputs of each call that are addresses. Every call a trace contains must
# be listed, with an empty list if it has none.
ADDRESSES = {
    "SetHeapSize": ["out_address"],
    "SetMemoryPermission": ["address"],
    "SetMemoryAttribute": ["address"],
    "QueryMemory": ["out_memory_info", "address"],
    "ExitProcess": [],
    "CloseHandle": [],
    "ConnectToNamedPort": ["name"],
    "SendSyncRequest": [],
    "OutputDebugString": ["debug_str"],
    "GetInfo": [],  # Its output is an address for some ids: see INFO_REGIONS.
}

# `GetInfo` ids whose answer is where a region starts (`InfoType` in
# eden/src/core/hle/kernel/svc_types.h), and the id that reports the same region's size.
INFO_REGIONS = {
    2: ("alias_region", 3),
    4: ("heap_region", 5),
    12: ("aslr_region", 13),
    14: ("stack_region", 15),
}

SVC = re.compile(r"Kernel\.SVC <Info> svc (0x[0-9a-f]+) (in|out) +t(\d+) (.*)$")
IPC = re.compile(r"Service <Info> ipc (.*)$")
LOG = re.compile(r"Debug\.Emulated <Info> (.*)$")
OTHER = re.compile(r"Kernel\.SVC <Info> (Process \d+ exiting)$")
REGISTER = re.compile(r"\[(\d)\]=(0x[0-9a-f]+)")


def conventions():
    """Number -> (name, inputs, outputs); an input or output is (name, type, register)."""
    numbers = {}
    for line in (REFERENCE / "svc_ids.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, number = line.split(" = ")
            numbers[name.strip()] = int(number, 16)
    sizes = {"uint64_t": 8, "int64_t": 8, "uint32_t": 4, "int32_t": 4, "bool": 1}
    for line in (REFERENCE / "svc_type_sizes.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, size = line.split(" = ")
            sizes[name.strip()] = int(size)
    table = {}
    for line in (REFERENCE / "svc_wrappers64.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, inputs, outputs = (part.strip() for part in line.split("|"))

        def args(text):
            parsed = []
            for item in text.split():
                arg, kind, register = item.split(":")
                parsed.append((arg, sizes[kind], int(register[1:]), kind))
            return parsed

        table[numbers[name]] = (name, args(inputs), args(outputs))
    return table


def decode(args, registers):
    """The values of `args` in `registers`, each truncated to its size."""
    return [
        (name, registers[register] & ((1 << (8 * size)) - 1), kind)
        for name, size, register, kind in args
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=pathlib.Path)
    parser.add_argument("--nro", type=pathlib.Path, default=FIXTURE)
    options = parser.parse_args()
    (image_size,) = struct.unpack_from("<I", options.nro.read_bytes(), NRO_IMAGE_SIZE_OFFSET)
    table = conventions()

    # First pass: pair each call's two lines, in the order the calls returned, with the lines
    # logged in between before it.
    calls = []
    threads = set()
    pending = None
    for raw_line in options.raw.read_text().splitlines():
        line = re.sub(r"^\[ *[\d.]+\] ", "", raw_line)
        if match := SVC.search(line):
            number, direction, thread, rest = match.groups()
            threads.add(thread)
            registers = {int(n): int(value, 16) for n, value in REGISTER.findall(rest)}
            if direction == "in":
                if pending is not None:
                    sys.exit(f"nested call: {raw_line}")
                pending = (int(number, 16), registers)
            else:
                if pending is None or pending[0] != int(number, 16):
                    sys.exit(f"unpaired return: {raw_line}")
                calls.append(("svc", pending[0], pending[1], registers))
                pending = None
        elif match := IPC.search(line):
            calls.append(("ipc", match.group(1)))
        elif match := LOG.search(line):
            calls.append(("log", match.group(1)))
        elif not OTHER.search(line):
            sys.exit(f"unrecognised line: {raw_line}")
    if pending is not None:
        sys.exit("the trace ends inside a call")
    if len(threads) != 1:
        sys.exit(f"expected one thread, found {sorted(threads)}")

    # Second pass: the anchors. Regions come from `GetInfo`'s answers.
    anchors = {}
    info = {}
    for call in calls:
        if call[0] == "svc" and table[call[1]][0] == "GetInfo":
            _, number, registers, returned = call
            name, inputs, outputs = table[number]
            info_type = decode(inputs, registers)[0][1]
            info[info_type] = decode(outputs, returned)[1][1]
    for start_id, (anchor, size_id) in INFO_REGIONS.items():
        if start_id in info and size_id in info:
            anchors[anchor] = (info[start_id], info[start_id] + info[size_id])

    def address_values():
        for call in calls:
            if call[0] != "svc":
                continue
            _, number, registers, returned = call
            name, inputs, outputs = table[number]
            if name not in ADDRESSES:
                sys.exit(f"no address rule for {name}: add it to ADDRESSES")
            values = decode(inputs, registers) + decode(outputs, returned)
            for arg, value, _ in values:
                if arg in ADDRESSES[name]:
                    yield value

    # The module lies in the ASLR region, below every other region; the end of its image is
    # the highest address there that the trace names (see the module docs).
    others = [start for name, (start, _) in anchors.items() if name != "aslr_region"]
    module_addresses = [value for value in address_values() if value < min(others)]
    image_end = max(module_addresses)
    module = (image_end - image_size, image_end)
    if module[0] % 0x1000 or min(module_addresses) < module[0]:
        sys.exit(f"the image {module[0]:#x}..{module[1]:#x} does not hold every code address")

    def symbol(value):
        # The module first (its image's end included: libnx asks about it), then the regions,
        # innermost first; the ASLR region contains all the others, so it goes last.
        if module[0] <= value <= module[1]:
            return f"main+{value - module[0]:#x}"
        for name in ["stack_region", "alias_region", "heap_region", "aslr_region"]:
            if name in anchors and anchors[name][0] <= value < anchors[name][1]:
                return f"{name}+{value - anchors[name][0]:#x}"
        sys.exit(f"address {value:#x} lies in no known region")

    def render(name, values, info_type=None):
        parts = []
        for arg, value, kind in values:
            if kind == "Result":
                continue
            is_address = arg in ADDRESSES[name] or (
                name == "GetInfo" and arg == "out" and info_type in INFO_REGIONS
            )
            parts.append(f"{arg}={symbol(value) if is_address else hex(value)}")
        return parts

    for call in calls:
        if call[0] != "svc":
            print(f"{call[0]} {call[1]}")
            continue
        _, number, registers, returned = call
        name, inputs, outputs = table[number]
        arguments = decode(inputs, registers)
        results = decode(outputs, returned)
        info_type = arguments[0][1] if name == "GetInfo" else None
        line = f"svc {number:#04x} {name}({', '.join(render(name, arguments))})"
        if results and results[0][2] == "Result":
            line += f" -> {results[0][1]:#x}"
            # Outputs other than the result are only meaningful when the call succeeded.
            if results[0][1] == 0:
                for part in render(name, results, info_type):
                    line += f" {part}"
        print(line)


if __name__ == "__main__":
    main()
