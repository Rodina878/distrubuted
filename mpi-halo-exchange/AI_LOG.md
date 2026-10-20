# AI log

Rule of this course: **AI proposes. Engineer verifies.**

## Round 1: asking an AI for help with the hang

Name of the assistant: Claude (Anthropic)

I started this round after I had my own evidence and my own three hypotheses (INCIDENT.md, section 4).

### The exact prompt I sent (verbatim)
````
I'm running a 3-container Docker Compose cluster (node0, node1, node2) and launching:
mpirun --hostfile hostfile -n 6 python3 img_mpi_v3_naive_halo.py
I'm using mpi4py with Open MPI 4.1.4.
Expected: the job finishes and writes an output file.
Actual: The large image hangs. All six ranks print "before Send", but none prints "after Send". The timeout terminates the job after 20 seconds with exit code 124.
Small image (width 256): The original v3 program completed successfully, wrote small256_out.npy, and exited with code 0.
Large image (width 8192): The program hangs, and timeout returns exit code 124.
Evidence:

1. Open MPI version:

Command:
mpirun --version | head -1
Output:
mpirun (Open MPI) 4.1.4

2. Hostname test:

Command:
mpirun --hostfile hostfile -n 6 hostname
Output:
node0
node0
node2
node2
node1
node1

3. Small image test:

Command:
timeout 20 mpirun --hostfile hostfile -n 6 python3 img_mpi_v3_naive_halo.py small256 small256_out ; echo "exit code: $?"
Output:
[rank 1 on node0] before Send
[rank 1 on node0] after Send
[rank 2 on node1] before Send
[rank 2 on node1] after Send
[rank 3 on node1] before Send
[rank 2 on node1] after Recv
[rank 3 on node1] after Send
[rank 3 on node1] after Recv
[rank 4 on node2] before Send
[rank 4 on node2] after Send
[rank 4 on node2] after Recv
[rank 0 on node0] before Send
[rank 0 on node0] after Send
[rank 1 on node0] after Recv
[rank 0 on node0] after Recv
[rank 5 on node2] before Send
[rank 5 on node2] after Send
[rank 5 on node2] after Recv
[rank 0 on node0] wrote small256_out.npy
exit code: 0

4. Large image test:

Command:
timeout 20 mpirun --hostfile hostfile -n 6 python3 img_mpi_v3_naive_halo.py break_8192 break_8192_out
echo "exit code: $?"
Output:
[rank 1 on node0] before Send
[rank 2 on node1] before Send
[rank 3 on node1] before Send
[rank 4 on node2] before Send
[rank 0 on node0] before Send
[rank 5 on node2] before Send
exit code: 124

5. Synchronous-send test:

Command:
sed 's/comm.Send(/comm.Ssend(/g' img_mpi_v3_naive_halo.py > v3_ssend_test.py
Then:
timeout 20 mpirun --hostfile hostfile -n 6 python3 v3_ssend_test.py small256 small256_ssend_out ; echo "exit code: $?"
Output:
[rank 1 on node0] before Send
[rank 2 on node1] before Send
[rank 3 on node1] before Send
[rank 4 on node2] before Send
[rank 0 on node0] before Send
[rank 5 on node2] before Send
mpirun: Forwarding signal 18 to job
exit code: 124
Relevant code from img_mpi_v3_naive_halo.py:

```
print(f"[rank {rank} on {hostname}] before Send", flush=True)

if rank > 0:
    comm.Send(my_strip[0, :].copy(), dest=rank - 1, tag=TAG_UP)
if rank < size - 1:
    comm.Send(my_strip[-1, :].copy(), dest=rank + 1, tag=TAG_DOWN)

print(f"[rank {rank} on {hostname}] after Send", flush=True)

top_halo = np.empty(width_for_halo, dtype=np.uint8)
bottom_halo = np.empty(width_for_halo, dtype=np.uint8)

if rank > 0:
    comm.Recv(top_halo, source=rank - 1, tag=TAG_DOWN)
else:
    top_halo = my_strip[0, :].copy()

if rank < size - 1:
    comm.Recv(bottom_halo, source=rank + 1, tag=TAG_UP)
else:
    bottom_halo = my_strip[-1, :].copy()

print(f"[rank {rank} on {hostname}] after Recv", flush=True)
```

Please analyze the evidence and provide:

1. My top 2–3 hypotheses, ranked.
2. One command per hypothesis to confirm or reject it.
3. The exact output that would confirm versus reject each hypothesis.
4. The smallest correct fix, not a rewrite.
5. One command that proves the fix worked on the large image.
6. Explain whether increasing the MPI eager-send limit would fix the underlying problem or only hide the symptom.

Do not say "check your MPI setup". Distinguish a genuine fix from a workaround. Base your conclusions on the actual evidence and explain any remaining uncertainty.
````

### The AI's full response (verbatim)
````
## Diagnosis

The program is only correct because small messages are buffered. Every rank does a blocking `Send` before it posts any `Recv`. The MPI standard allows `Send` to block until the matching `Recv` is posted. Below the eager threshold, Open MPI copies the message into an internal buffer and returns early. Above it, `Send` waits for the receiver, and with every rank sending first, nobody ever receives.

The deciding evidence is test 5. With `Ssend` (which never buffers), the 256-wide image hangs the same way as the 8192-wide one. The small run worked only because 256-byte messages were buffered, and the program has been unsafe all along.

The per-rank pattern is a cycle. Rank 1 does `Send` to rank 0 first, while rank 0 is in `Send` to rank 1. Each waits for the other's `Recv`, and ranks 2/3 and 4/5 are stuck the same way. This matches your output: all six ranks print "before Send" and none prints "after Send".

## Ranked hypotheses

### 1. Unsafe ordering plus rendezvous protocol (most likely, nearly certain)

The cyclic send-before-recv pattern deadlocks once the message exceeds the eager limit.

**Command:**
```bash
timeout 20 mpirun --hostfile hostfile -n 6 python3 v3_ssend_test.py small256 x ; echo "exit code: $?"
```

- **Confirms:** hang, exit 124, no "after Send". You already have this result.
- **Rejects:** exit 0 and `wrote ...npy`. That would mean the small case really is deadlock-free regardless of buffering, which would leave the problem unexplained.

### 2. The size threshold is specific to the transport, and 8 KB crosses it

Your rows are only 8192 bytes. I expect Open MPI 4.1.x defaults of about 4096 bytes for the shared-memory transport (`vader`) and about 32768 for `tcp`. If so, 256 bytes is eager and 8192 bytes is rendezvous on every same-node pair (0↔1, 2↔3, 4↔5), which would explain why all six ranks block. I'm recalling these defaults from memory, so check them first:

**Command (check):**
```bash
ompi_info --param btl vader --level 9 | grep eager_limit; ompi_info --param btl tcp --level 9 | grep eager_limit
```

**Command (test):**
```bash
timeout 30 mpirun --hostfile hostfile -n 6 --mca btl_vader_eager_limit 65536 --mca btl_tcp_eager_limit 65536 python3 img_mpi_v3_naive_halo.py break_8192 break_8192_out; echo "exit code: $?"
```

- **Confirms:** `btl_vader_eager_limit` is below 8192 (4096 expected), and the run exits 0 with "after Send" printed on all ranks. A limit raised above the message size makes it complete, so the threshold was the trigger.
- **Rejects:** the limits are already at or above 8192, or the run still hangs after raising them. Then the buffering threshold isn't what separates your two cases.

### 3. Docker or inter-node networking problem (unlikely)

**Command:** a correctly paired large transfer between two nodes, with no deadlock possible:
```bash
timeout 30 mpirun --hostfile hostfile -n 2 --map-by node python3 -c "from mpi4py import MPI;import numpy as np;c=MPI.COMM_WORLD;b=np.zeros(1<<20,'u1');c.Send(b,1) if c.rank==0 else c.Recv(b,0);print(c.rank,'ok',flush=True)"; echo "exit code: $?"
```

- **Rejects (expected):** both ranks print `ok` and the exit code is 0. The network handles large cross-node messages.
- **Confirms:** a hang or a TCP connection error. Then you'd have a separate problem, such as the wrong interface being chosen (`--mca btl_tcp_if_include eth0`).

Hypothesis 3 is already unlikely. Test 2 shows `mpirun` placing and launching ranks on all three nodes. Small messages crossed nodes in test 3. And the `Ssend` hang on 256 bytes is what a pure ordering bug looks like.

## Smallest correct fix

Make the sends nonblocking, do the receives as before, then wait for the sends. Replace the send block and add one `Waitall` after the receives:

```python
reqs = []
if rank > 0:
    up_row = my_strip[0, :].copy()
    reqs.append(comm.Isend(up_row, dest=rank - 1, tag=TAG_UP))
if rank < size - 1:
    down_row = my_strip[-1, :].copy()
    reqs.append(comm.Isend(down_row, dest=rank + 1, tag=TAG_DOWN))

# ... existing Recv block unchanged ...

MPI.Request.Waitall(reqs)
print(f"[rank {rank} on {hostname}] after Recv", flush=True)
```

Two details matter here. `up_row` and `down_row` must stay referenced until `Waitall`, because the buffer must not be freed or modified while the send is in flight. And the old "after Send" print would now only show that the send was posted, not completed, so drop it or move it after `Waitall`. `comm.Sendrecv` with `MPI.PROC_NULL` at the edges would also work, but it's a bigger edit.

## Proof the fix works

```bash
timeout 60 mpirun --hostfile hostfile -n 6 python3 img_mpi_v3_naive_halo.py break_8192 break_8192_out; echo "exit code: $?"; ls -l break_8192_out.npy
```

Expected: exit code 0, `wrote break_8192_out.npy`, and the file exists.

To show the fix doesn't rely on buffering, rerun the same check with the sends made synchronous, which deadlocked before:

```bash
sed 's/comm.Isend(/comm.Issend(/g' img_mpi_v3_naive_halo.py > v3_fixed_issend.py && timeout 60 mpirun --hostfile hostfile -n 6 python3 v3_fixed_issend.py break_8192 b_out; echo "exit code: $?"
```

Exit 0 here proves correctness independent of the eager limit. The original program fails this exact test even at width 256.

## Does raising the eager limit fix it?

It only hides the symptom. The bug is the send-before-recv cycle, and the eager limit only decides how large a message can be before the bug shows up. Raising it makes 8192 pass, as the hypothesis 2 test shows, but:

- Any wider image, or any larger per-rank message, deadlocks again once it exceeds the new limit.
- The behavior depends on transport. It already differs between `vader` and `tcp`, and a different MPI build or network (InfiniBand, UCX) has different thresholds.
- Eager limits are bounded by internal buffer sizes, so you can't raise them indefinitely.

Treat the eager-limit change as a diagnostic that proves hypothesis 2, not as the fix.

## Remaining uncertainty

- I haven't seen your `ompi_info` output, so the 4096 and 32768 defaults are from memory. The `Ssend` result establishes the root cause regardless of the exact numbers.
- I can't see which call each rank is stuck in. The "before Send, no after Send" pattern is consistent with the deadlock, but a stack dump would show it directly. If the fix somehow doesn't resolve it, attach `py-spy dump --pid <pid>` on a hung rank before drawing further conclusions.
- If the `Issend` run of the fixed code still hangs, there is a second problem I can't see from this evidence, most likely in the `Recv` tag or source pairing.
````

### What I tested
| AI claim or suggestion | The command I ran | Real output | Verdict |
|---|---|---|---|
| Ordering plus rendezvous protocol is the cause; `Ssend` on the small image should hang | `timeout 20 mpirun --hostfile hostfile -n 6 python3 v3_ssend_test.py small256 small256_ssend_out ; echo "exit code: $?"` (this was already in my prompt) | six `before Send`, no `after Send`, `exit code: 124` | Confirmed |
| `vader` eager limit is about 4096 bytes, below one 8192-byte row | `ompi_info --param btl vader --level 9 \| grep eager_limit` | `btl_vader_eager_limit` current value `4096`, data source: default | Confirmed (the `tcp` default of 32768 was not checked) |
| Raising the limits above the row size makes the 8192 run finish | `timeout 30 mpirun --hostfile hostfile -n 6 --mca btl_vader_eager_limit 65536 --mca btl_tcp_eager_limit 65536 python3 img_mpi_v3_naive_halo.py break_8192 break_8192_out; echo "exit code: $?"` | all ranks printed `after Send` and `after Recv`, `wrote break_8192_out.npy`, `exit code: 0` | Confirmed, but it is a workaround, not a fix (two limits were raised at once, so it does not show which one mattered) |
| Network check with a paired 1 MiB transfer between two nodes | not run | - | Not tested (the `hostname` run and the cross-node success on small messages already made this unlikely) |
| `Isend` + `Waitall` fix | not run; I used `Sendrecv` (v2) as the fix instead | see INCIDENT.md, section 7 | Not tested |

### Where the AI's hypotheses differed from mine
My hypotheses were Docker/SSH, a logic bug, and a blocking-`Send` deadlock. The AI ranked the deadlock first and added a point I had not considered: the eager threshold is a per-transport setting (`vader` 4096 bytes), which explains why 256 passes and 8192 does not. I checked that number myself with `ompi_info`.

## Round 2: judging the pre-written answer (claims A, B, C)

| Claim | Verdict | The command | Real output (paste) |
|---|---|---|---|
| A (Docker networking is flaky; restart Docker) | REFUTED | `mpirun --hostfile hostfile -n 6 hostname` | `node0`, `node0`, `node2`, `node2`, `node1`, `node1`; the hang happened with all containers healthy, so no restart was needed <add `docker compose ps` output> |
| B (`sshd` is not reachable, so remote ranks cannot start) | REFUTED | `ssh node1 hostname` and `ssh node2 hostname`; `before Send` printed from node1 and node2 | <paste the real output of the two ssh commands> |
| C (raise the eager limit; that will fix it) | HIDES THE SYMPTOM | failing width: step 6 above (8192 passes with limit 65536). Stricter test: `Ssend` on the 256-wide image | with raised limit at 8192: `exit code: 0`. With `Ssend` at 256: six `before Send`, no `after Send`, `exit code: 124`. <optional: bigger-width run, expected to hang again above the new limit; paste it if you run it> |

For C the claim's own flag was `btl_vader_eager_limit 16384`; I used 65536 for both `vader` and `tcp`. The point is the same: a bigger limit lets this input through but leaves the send-before-receive cycle in the code, and the program still hangs under `Ssend`.

## Reflection
The AI saved time by pointing at the eager threshold and by giving ready-to-run commands. It could have misled me in two ways: it quoted the `vader` and `tcp` defaults from memory (I verified the first with `ompi_info`, not the second), and it suggested the eager-limit change, which makes the symptom disappear without removing the cause. In a later exchange it also claimed that v2 contains no `comm.Send(` calls; that was wrong (v2 still uses `Send` to scatter strips and gather results), which I found by reading the file.
