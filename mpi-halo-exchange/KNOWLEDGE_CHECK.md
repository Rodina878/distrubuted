# Knowledge check

**Q1. Two different causes of "containers healthy but `mpirun` hangs", and the output that tells them apart.**
(a) The launch itself fails (SSH, DNS or hostfile problem). `mpirun --hostfile hostfile -n 6 hostname` hangs or errors, or the remote ranks never print anything. (b) The program deadlocks after a successful launch. `hostname` prints six lines (two per node), every rank prints `before Send`, and none prints `after Send`. `docker compose ps` says `Up (healthy)` in both cases, so it cannot tell them apart.

**Q2. Why did the naive code pass on a small image and hang on a big one? Did the small run "prove" it correct?**
Standard-mode `Send` may return early by copying a small message into an internal buffer (eager protocol), but a larger message uses a rendezvous protocol where `Send` waits for the matching `Recv`. At width 256 a row is 256 bytes (eager), at width 8192 it is 8192 bytes, above the 4096-byte `btl_vader_eager_limit` reported by `ompi_info` here. The small run did not prove correctness: the same program under `Ssend`, which never buffers, hung even at width 256.

**Q3. When does "just raise the eager limit" stop working, and why is it not a fix?**
It stops working as soon as a message is larger than the new limit. Raising the limit to 65536 let the 8192-wide image finish, but the program still sends to both neighbours before receiving, so the cyclic wait is still there. A row wider than the new limit would hang again (I have not yet measured that width). The limit is an implementation and configuration choice, not something the MPI Standard promises.

**Q4. Why can't `mpirun` just use `docker compose exec` instead of SSH?**
SSH is the portable way `mpirun` starts processes on remote machines. `docker compose exec` needs the Docker CLI and access to the Docker daemon, which node0 does not have and which a real cluster would not offer.

**Q5. Why does a 3x3 filter need exactly one halo row on each side? What changes for 5x5?**
Each output pixel depends on neighbours at distance 1 (radius 1), so the last row of a strip needs exactly one row from the neighbour. A 5x5 filter has radius 2, so each rank needs two halo rows from each side.

**Q6. Why do we run MPI as `student` and not `root`?**
Open MPI refuses to run as root by default, because a mistake in a program running as root can damage the whole system. The lab's Dockerfile creates the `student` user for this reason, and the SSH keys and shared work folder are set up for it.
