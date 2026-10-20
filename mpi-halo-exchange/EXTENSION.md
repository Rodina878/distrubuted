# Extension E3: which programs does `Ssend` break?

`Ssend` completes only when the matching receive has started, so it exposes code that works only because the library buffers messages. I already knew v3 hangs under `Ssend` at width 256. Here I apply the same change to copies of v1 and v2.

## Prediction (written before running)

<Write this BEFORE you run the commands. If you ran first, say so here.>

My prediction: both v1 and v2 still finish. `comm.Send(` appears on two lines in each file (v1: the scatter of strips from rank 0 and the send of filtered strips back; v2: the same two places), so `sed` changes two lines in each. In both, the sends go one way only: rank 0 sends to workers that are already waiting in `Recv`, and the workers send back to rank 0, which receives in worker order. There is no cycle of ranks all waiting for each other. v2's halo exchange uses `Sendrecv`, which is not touched by the `sed` and cannot deadlock on its own. I also expect v1 to still be *wrong* (4236 differing pixels), because `Ssend` tests deadlock safety, not correctness.

## Measurement

Run inside node0 as `student`, from `work/`, 6 ranks, `timeout 25`:

```bash
for V in v1_nohalo v2_halo; do
  P=${V%%_*}
  sed 's/comm\.Send(/comm.Ssend(/g' img_mpi_${V}.py > ${P}_ssend_test.py
  echo "=== $P | lines changed by sed: $(diff img_mpi_${V}.py ${P}_ssend_test.py | grep -c '^>')"
  rm -f test_image_filtered_${P}.npy
  timeout 25 mpirun --hostfile hostfile -n 6 python3 ${P}_ssend_test.py > ${P}_ssend.log 2>&1
  echo "$P exit code: $?"
  python3 compare.py --candidate test_image_filtered_${P} | tail -3
done
```

Results (paste the real output of the loop above):

- v1 under `Ssend`: lines changed `<N>`, exit code `<0 or 124>`, `compare.py`: `<result>`.
- v2 under `Ssend`: lines changed `<N>`, exit code `<0 or 124>`, `compare.py`: `<result>`.
- v3 under `Ssend` (measured earlier, width 256): `timeout` killed it after 20 s, exit code 124, all six ranks printed `before Send` and none printed `after Send`.

## Explanation

v3 stalls under `Ssend` because every rank calls `Send` to both neighbours before it posts any `Recv`: each rank waits for a receive that its neighbour will only post after its own send returns. That is a cycle. v1 has no such cycle, since data flows from rank 0 outward and back. v2 combines send and receive in one `Sendrecv` call, so no rank blocks in a send while its receive is still unposted. The small-image success of v3 was an accident of buffering; v1 and v2 do not depend on it.

## What surprised me

<Write one real sentence. If nothing did, write "Nothing surprised me; the results matched my prediction.">
