#!/usr/bin/env python3
"""summarize.py <run-dir> [stop-regex]: writes <run-dir>/startup.txt (svc/ipc/guest lines up to
the first line matching stop-regex, inclusive; default the first svc 0x07 out) and prints the
services opened via sm GetService in order plus SVC counts for that window."""
import gzip, re, sys, collections
run = sys.argv[1]
stop = re.compile(sys.argv[2] if len(sys.argv) > 2 else r'svc 0x07 out')
names = {int(l.split()[0], 16): l.split()[1] for l in open(__file__.rsplit('/', 1)[0] + '/svc_names.txt')}
keep = re.compile(r'svc 0x|: ipc |: tipc |Debug\.Emulated|exiting|ExceptionRaised|nimplemented')
lines = []
with gzip.open(run + '/log.txt.gz', 'rt', errors='replace') as f:
    for l in f:
        l = l.replace('\0', '')
        if keep.search(l):
            lines.append(re.sub(r' core/\S+: ', ' ', l.rstrip()))
            if stop.search(l):
                break
open(run + '/startup.txt', 'w').write('\n'.join(lines) + '\n')
counts = collections.Counter(); services = []; last_in = None
for l in lines:
    m = re.search(r'svc (0x[0-9a-f]+) in .*\[1\]=(0x[0-9a-f]+)', l)
    if m:
        counts[int(m.group(1), 16)] += 1; last_in = int(m.group(2), 16)
    if 'sm:::1 (GetService)' in l and last_in is not None:
        services.append(last_in.to_bytes(8, 'little').rstrip(b'\0').decode('ascii', 'replace'))
print('services via sm GetService, in order:', ' '.join(services))
for k, v in sorted(counts.items()):
    print(f'  svc {k:#04x} {names.get(k, "?"):28s} {v}')
print('lines:', len(lines), 'window ends:', lines[-1][:60])
