#!/usr/bin/env python3
"""Normalises a reference emulator's trace of a homebrew fixture into its `startup.trace`.

The reference emulator (Eden-NX2, see docs/reference-emulator.md) logs every system call twice,
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
  address lies in: the module (`main`), a placement (below), or one of the regions `GetInfo`
  reported in the same trace (`aslr_region`, `stack_region`, `alias_region`, `heap_region`);
- the answers of `GetInfo` ids whose value is random by design (`RANDOM_INFO`) written as
  `random`.

Which arguments and outputs are addresses is listed in `ADDRESSES` below; a call that is not
listed there is refused, so nothing is passed through unmasked by oversight.

**Placements.** libnx picks where to map a shared memory at random: `virtmemFindAslr` draws a
page of the ASLR region from its random generator, asks `QueryMemory` about the page a guard
before it, and maps there if the range is free (libnx/nx/source/kernel/virtmem.c,
`_memregionFindRandom` and `_memregionIsMapped`; the guard is 0x1000 in
libnx/nx/source/kernel/shmem.c, `shmemMap`). The generator is seeded from `GetInfo`'s random
entropy, so the page differs from run to run. Each range a call in `PLACEMENTS` maps is a
placement, named `placement1`, `placement2`, ... in the order the trace maps them, and anchored
at the range's start less the guard (the address libnx's probe asks about). A placement must lie
where libnx may put one, or the trace is refused: in the ASLR region, outside the module, the
alias region and the heap region. `_memregionFindRandom` skips the alias and heap regions and
anything mapped, and the kernel lets a shared memory lie anywhere else in the ASLR region
(eden/src/core/hle/kernel/k_page_table_base.cpp, `KPageTableBase::CanContain`:
`is_in_region && !is_in_heap && !is_in_alias` for `MemoryState::Shared`), so a placement in the
stack region is accepted. `crates/core/tests/homebrew.rs` pairs placements by the same rule.

Usage, from the repository root:

    tools/oracle/normalise.py --nro <fixture.nro> <raw startup.txt>

The module's base is not in the trace, so it is worked out from the fixture. At start-up libnx
asks `QueryMemory` about `__argdata__` (libnx/nx/source/runtime/argv.c), which libnx's linker
script puts at the first page boundary after `.bss` (`. = ALIGN(0x1000)` before `__argdata__`
in libnx/nx/switch.ld), so the highest module address the trace names is `__argdata__`. The
script takes the module's extent, from its base to `__argdata__`, to be the NRO header's image
size (the word at 0x18 of the file) plus its `.bss` size (the word at 0x38), each rounded up to
a page. For both fixtures that ends exactly at `__argdata__` (checked against each ELF's symbol
table when the startup fixture was added); that it does for every libnx NRO is a guess.

This is not the extent Eden maps. Every libnx NRO has a `MOD0` header, and for those Eden's
`LoadNroImpl` (eden/src/core/loader/nro.cpp, the `has_mod_header` branch) takes the `.bss`
size from `MOD0` (`bss_end_offset - bss_start_offset`, rounded up to a page) and maps it after
the page-aligned image. That is one page more than this script's extent for both fixtures:
hello's `MOD0` `.bss` is 0xce0 bytes, which fit in the image's last page (its NRO header's
`.bss` size is 0), and Eden maps a page for them (0x7000 against 0x6000); startup's is 0x47b0
bytes, which Eden rounds to 0x5000 where the NRO header has 0x4000 (0x27000 against 0x26000).

The script checks that the base is then page aligned and that every other module address lies
inside the module.
"""

import argparse
import pathlib
import re
import struct
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
REFERENCE = REPO / "crates/kernel/src/svc/reference"
# Where an NRO's header keeps the size of its image and of its `.bss` (`NroHeader::file_size`
# and `NroHeader::bss_size` in eden/src/core/loader/nro.cpp).
NRO_IMAGE_SIZE_OFFSET = 0x18
NRO_BSS_SIZE_OFFSET = 0x38
PAGE = 0x1000

# The arguments and outputs of each call that are addresses. Every call a trace contains must
# be listed, with an empty list if it has none.
ADDRESSES = {
    "SetHeapSize": ["out_address"],
    "SetMemoryPermission": ["address"],
    "SetMemoryAttribute": ["address"],
    "QueryMemory": ["out_memory_info", "address"],
    "ExitProcess": [],
    "MapSharedMemory": ["address"],
    "UnmapSharedMemory": ["address"],
    "CreateTransferMemory": ["address"],
    "CloseHandle": [],
    "WaitSynchronization": ["handles"],
    "ConnectToNamedPort": ["name"],
    "SendSyncRequest": [],
    "OutputDebugString": ["debug_str"],
    "GetInfo": [],  # Its output is an address for some ids: see INFO_REGIONS.
}

# The calls that map a range libnx placed at random, with the arguments holding the range's
# start and size (see the module docs).
PLACEMENTS = {
    "MapSharedMemory": ("address", "size"),
}
# The guard libnx keeps free on each side of a placement (libnx/nx/source/kernel/shmem.c,
# `virtmemFindAslr(s->size, 0x1000)`).
PLACEMENT_GUARD = 0x1000

# `GetInfo` ids whose answer is where a region starts (`InfoType` in
# eden/src/core/hle/kernel/svc_types.h), and the id that reports the same region's size.
INFO_REGIONS = {
    2: ("alias_region", 3),
    4: ("heap_region", 5),
    12: ("aslr_region", 13),
    14: ("stack_region", 15),
}
# `GetInfo` ids whose answer is random by design: `RandomEntropy` (eden/src/core/hle/kernel/
# svc_types.h, `InfoType::RandomEntropy = 11`), which libnx seeds its generator from.
RANDOM_INFO = {11}

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
    parser.add_argument("--nro", type=pathlib.Path, required=True)
    options = parser.parse_args()
    nro = options.nro.read_bytes()
    (image_size,) = struct.unpack_from("<I", nro, NRO_IMAGE_SIZE_OFFSET)
    (bss_size,) = struct.unpack_from("<I", nro, NRO_BSS_SIZE_OFFSET)
    extent = -(-image_size // PAGE) * PAGE + -(-bss_size // PAGE) * PAGE
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

    def svc_calls():
        """Each system call as (name, decoded inputs, decoded outputs)."""
        for call in calls:
            if call[0] == "svc":
                _, number, registers, returned = call
                name, inputs, outputs = table[number]
                if name not in ADDRESSES:
                    sys.exit(f"no address rule for {name}: add it to ADDRESSES")
                yield name, decode(inputs, registers), decode(outputs, returned)

    # Second pass: the anchors. Regions come from `GetInfo`'s answers, placements from the
    # calls that map them.
    anchors = {}
    info = {}
    for name, inputs, outputs in svc_calls():
        if name == "GetInfo":
            info[inputs[0][1]] = outputs[1][1]
    for start_id, (anchor, size_id) in INFO_REGIONS.items():
        if start_id in info and size_id in info:
            anchors[anchor] = (info[start_id], info[start_id] + info[size_id])
    placements = []
    for name, inputs, _ in svc_calls():
        if name in PLACEMENTS:
            values = {arg: value for arg, value, _ in inputs}
            start_arg, size_arg = PLACEMENTS[name]
            start, size = values[start_arg], values[size_arg]
            placement = (start - PLACEMENT_GUARD, start + size + PLACEMENT_GUARD)
            if placement not in placements:
                placements.append(placement)

    def in_placement(value):
        return any(start <= value < end for start, end in placements)

    def address_values():
        for name, inputs, outputs in svc_calls():
            for arg, value, _ in inputs + outputs:
                if arg in ADDRESSES[name]:
                    yield value

    # The module lies in the ASLR region, below every other region; the end of its image is
    # the highest address there that the trace names (see the module docs).
    others = [start for name, (start, _) in anchors.items() if name != "aslr_region"]
    module_addresses = [
        value for value in address_values() if value < min(others) and not in_placement(value)
    ]
    image_end = max(module_addresses)
    module = (image_end - extent, image_end)
    if module[0] % PAGE or min(module_addresses) < module[0]:
        sys.exit(f"the image {module[0]:#x}..{module[1]:#x} does not hold every code address")

    inner = ["stack_region", "alias_region", "heap_region"]
    # The regions libnx keeps placements out of (see the module docs).
    avoided = ["alias_region", "heap_region"]
    for start, end in placements:
        outside = all(
            end <= anchors[name][0] or anchors[name][1] <= start
            for name in avoided
            if name in anchors
        )
        aslr = anchors.get("aslr_region")
        if (
            not outside
            or aslr is None
            or not (aslr[0] <= start and end <= aslr[1])
            or not (end <= module[0] or module[1] < start)
        ):
            sys.exit(f"the placement {start:#x}..{end:#x} is not in the ASLR region alone")

    def symbol(value):
        # The module first (its end included: libnx asks about it), then the
        # placements, then the regions, innermost first; the ASLR region contains all the
        # others, so it goes last.
        if module[0] <= value <= module[1]:
            return f"main+{value - module[0]:#x}"
        for index, (start, end) in enumerate(placements, start=1):
            if start <= value < end:
                return f"placement{index}+{value - start:#x}"
        for name in inner + ["aslr_region"]:
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
            if name == "GetInfo" and arg == "out" and info_type in RANDOM_INFO:
                text = "random"
            elif is_address:
                text = symbol(value)
            else:
                text = hex(value)
            parts.append(f"{arg}={text}")
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
