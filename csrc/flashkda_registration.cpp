#include "core/registration.h"
#include "flash_kda.h"
#include "flashkda_kcp.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(_flashkda_C, m) {
  m.def(
      "kcp_transition(Tensor workspace, Tensor beta, Tensor cu_seqlens, "
      "Tensor chunk_offsets, Tensor(a!) partial) -> ()");
  m.def(
      "kcp_scan(Tensor v, Tensor beta, Tensor workspace, Tensor initial, "
      "Tensor cu_seqlens, Tensor(a!) out, Tensor(b!) final) -> ()");
  m.def("get_workspace_size(int T_total, int H, int N=1) -> int");
  m.def(
      "fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, "
      "Tensor(a!) out, Tensor(c!) workspace, Tensor A_log, Tensor dt_bias, "
      "float lower_bound, "
      "Tensor? initial_state=None, Tensor(b!)? final_state=None, "
      "Tensor? cu_seqlens=None, Tensor(d!)? checkpoint_state=None, "
      "Tensor? checkpoint_offsets=None) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_flashkda_C, CompositeExplicitAutograd, m) {
  m.impl("get_workspace_size", TORCH_BOX(&get_workspace_size));
}

STABLE_TORCH_LIBRARY_IMPL(_flashkda_C, CUDA, m) {
  m.impl("kcp_transition", TORCH_BOX(&flashkda_kcp::kcp_transition));
  m.impl("kcp_scan", TORCH_BOX(&flashkda_kcp::kcp_scan));
  m.impl("fwd", TORCH_BOX(&fwd));
}

REGISTER_EXTENSION(_flashkda_C)
