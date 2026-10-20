# Incident report: naive halo exchange hangs on an 8192-wide image

Blameless format: the system and the evidence, not who made the mistake.

## 1. Summary
A 6-rank image-filter job finished on a 256-wide image but hung on an 8192-wide one, with all three containers `Up (healthy)` and no error message. The cause was that every rank called blocking `Send` to both neighbours before posting any `Recv`: a cyclic wait that only disappears when Open MPI buffers the message (below the eager limit).

## 2. Symptoms
- Ran: `timeout 20 mpirun --hostfile hostfile -n 6 python3 img_mpi_v3_naive_halo.py break_8192 break_8192_out; echo "exit code: $?"` (the same run `break_deadlock.sh` does). Expected: the job finishes and writes an output file. Saw: six `before Send` lines, nothing else, then `exit code: 124` (killed by `timeout`).
- Screenshot of the hang ending with `CONFIRMED HANG`: `evidence/hang.png`
- `docker compose ps` taken while it hangs: `evidence/ps_during_hang.png`. All nodes were healthy. Anyone who monitors only container health would see nothing wrong: the containers and sshd are fine, only the MPI job is stuck.
- Who emitted what: `before Send` lines come from my program; `CONFIRMED HANG` comes from `break_deadlock.sh` after `timeout` killed `mpirun`; `Unable to negotiate with 127.0.0.1 ...` lines in `docker compose logs` come from the Docker healthcheck probing sshd inside the container, not from MPI. Confirmed: `docker compose logs node0` showed repeating "Connection closed by 127.0.0.1" and "Unable to negotiate with 127.0.0.1 ... sk-" lines every ~20 seconds. They come from the Docker healthcheck probing sshd inside the container (127.0.0.1 is the container itself), not from my MPI program, and they are not the cause of the hang.

## 3. Timeline
| Step | Command | What I saw |
|---|---|---|
| 1 | `mpirun --hostfile hostfile -n 6 hostname` | 6 lines: `node0` x2, `node1` x2, `node2` x2 |
| 2 | `timeout 20 mpirun ... img_mpi_v3_naive_halo.py small256 small256_out` | all ranks printed `before/after Send` and `after Recv`; `wrote small256_out.npy`; `exit code: 0` |
| 3 | `timeout 20 mpirun ... img_mpi_v3_naive_halo.py break_8192 break_8192_out` | six `before Send`, no `after Send`; `exit code: 124` |
| 4 | same as 2, with `comm.Send(` replaced by `comm.Ssend(` | six `before Send`, no `after Send`; `exit code: 124` |
| 5 | `ompi_info --param btl vader --level 9 \| grep eager_limit` | `btl_vader_eager_limit` current value `4096` (default) |
| 6 | step 3 with `--mca btl_vader_eager_limit 65536 --mca btl_tcp_eager_limit 65536` | finished, `wrote break_8192_out.npy`, `exit code: 0` |
| 7 | `bash verify_fix.sh` (x3) | see section 7 |

## 4. Investigation (three hypotheses, written before changing any code)

| # | Hypothesis | Command that confirms or rejects it | Real output | Verdict |
|---|---|---|---|---|
| H1 | Docker, network or SSH is broken, so ranks did not start or cannot reach each other | `docker compose ps` and `mpirun --hostfile hostfile -n 6 hostname` | `hostname`: 6 lines, 2 per node; `docker compose ps`: node0, node1 and node2 all `Up (healthy)` (full output below the table); `before Send` printed from node0, node1 and node2 | Rejected |
| H2 | Logic bug in the program (wrong tags or buffer sizes) | read the tag/buffer pairs, run v3 on `small256` | tags pair up (`TAG_UP`/`TAG_DOWN`); width 256 finished with `exit code: 0` and the output matched the serial filter (0 differing pixels) | Rejected |
| H3 | Every rank blocks in `Send` waiting for a `Recv` that nobody has reached (cyclic wait) | same program with `Send` replaced by `Ssend`, on the small image | `exit code: 124`, six `before Send`, no `after Send` | Confirmed |

`docker compose ps` output for H1:

```
NAME      IMAGE               COMMAND                  SERVICE   CREATED       STATUS                 PORTS
node0     scx61-week1-node0   "/usr/local/bin/entr…"   node0     5 hours ago   Up 5 hours (healthy)   0.0.0.0:2222->22/tcp, [::]:2222->22/tcp
node1     scx61-week1-node1   "/usr/local/bin/entr…"   node1     5 hours ago   Up 5 hours (healthy)   22/tcp
node2     scx61-week1-node2   "/usr/local/bin/entr…"   node2     5 hours ago   Up 5 hours (healthy)   22/tcp
```

Also:
- `hostname` output rules out SSH, DNS and `mpirun` launching problems: ranks really ran on all three nodes.
- How far each rank got: all six printed `before Send` and none printed `after Send`, so all of them are stuck in the send phase.
- Small versus large: at width 256 the same program finished; at 8192 it hung. A passing small run does not prove correctness: under `Ssend` the 256-wide case hangs too, so it passed only because small messages are buffered.
- `Ssend` on a copy of v3 at width 256 hung (`exit code: 124`). `Ssend` never returns before the receive has started, so this proves the program depends on buffering.

## 5. Root cause
The lines responsible are in the halo phase of `img_mpi_v3_naive_halo.py`, where each rank sends first and receives afterwards:

```python
if rank > 0:
    comm.Send(my_strip[0, :].copy(), dest=rank - 1, tag=TAG_UP)
if rank < size - 1:
    comm.Send(my_strip[-1, :].copy(), dest=rank + 1, tag=TAG_DOWN)
```

Each rank is in `Send` waiting for its neighbour's `Recv`, and the neighbour is in its own `Send`. No rank reaches its first `Recv`. Standard-mode `Send` may complete early by buffering (small messages, eager protocol) or may wait for the receiver (rendezvous). Here one row is `width` bytes: 256 bytes was eager, 8192 bytes exceeds the 4096-byte `btl_vader_eager_limit`.

MPI Standard reference: MPI 4.1, "Communication Modes".

## 6. The fix
The smallest change that removes the cause is to make each rank's send and receive one operation, as in `img_mpi_v2_halo.py`:

```python
comm.Sendrecv(sendbuf=strip[0, :].copy(), dest=rank - 1, sendtag=30,
              recvbuf=top_halo, source=rank - 1, recvtag=31)
comm.Sendrecv(sendbuf=strip[-1, :].copy(), dest=rank + 1, sendtag=31,
              recvbuf=bottom_halo, source=rank + 1, recvtag=30)
```

(each guarded by `rank > 0` / `rank < size - 1`). This removes the cause because no rank can sit in a send whose matching receive it has not posted, so the result no longer depends on buffering. Raising the eager limit (step 6) is only a workaround: the send-before-receive order is still there, and that test changed two limits at once (`vader` and `tcp`), so it does not show which one mattered.

## 7. Verification
- `bash verify_fix.sh`, three runs in a row (each should end with 0 differing pixels). I ran it five times in total and all five ended the same way; the three below are the first three:
  - run 1: `np.array_equal: True`, `Differing pixels: 0 / 2457600`, `0 differing pixels — exact match with the serial oracle.`
  - run 2: `np.array_equal: True`, `Differing pixels: 0 / 2457600`, `0 differing pixels — exact match with the serial oracle.`
  - run 3: `np.array_equal: True`, `Differing pixels: 0 / 2457600`, `0 differing pixels — exact match with the serial oracle.`
- `ssh node1 hostname` from node0: `node1` (and `ssh node2 hostname` printed `node2`).
- Width verified: 8192, the width at which the old code hung. A saved v2 output for 8192 compared against the serial filter gives 0 differing pixels (0 / 2457600).

## 8. Prevention
- Test every communication pattern with `Ssend` (or run it with eager buffering disabled) so reliance on buffering shows up on small inputs.
- Test with an input larger than the eager limit, and always run MPI jobs under `timeout` so a hang becomes a visible exit code 124.
- Do not treat container health as job health.