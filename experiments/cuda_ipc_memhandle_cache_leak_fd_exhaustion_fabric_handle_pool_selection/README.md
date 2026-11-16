# Round 322: CUDA IPC (`cudaIpcOpenMemHandle` vs `cuMemExportToShareableHandle`) MMU Stall, `EMFILE` FD Exhaustion & Base-Offset Pool Selection

Deterministic evaluator comparing 4 cross-process GPU memory sharing architectures (`512` shared tensors/step) across IPC mapping overhead (`ms/step`), open file descriptors per rank, and `VMM / GB200 MNNVL` compatibility.
