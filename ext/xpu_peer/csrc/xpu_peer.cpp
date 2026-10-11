// Cross-process GPU-to-GPU writes for XPU tensor parallelism.
//
// PeerBuffer allocates device memory with Level Zero in torch's SYCL context
// and exports it as an IPC handle. Another process opens that handle as a
// PeerMapping and writes into the buffer over PCIe with a copy on torch's
// current queue. On hosts without PCIe peer atomics (Arc Pro B70 on separate
// root ports) peer writes still run at near link speed, while peer reads are
// slow, so data only ever moves by being written into the receiver.
//
// NEO's IPC handles are opaque: they carry the exporter's pid, and
// zeMemOpenIpcHandle fetches the dma-buf fd itself (pidfd_getfd), so the
// 64 handle bytes can be passed between processes as plain data.

#include <torch/extension.h>

#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

void check(ze_result_t r, const char* what) {
  if (r != ZE_RESULT_SUCCESS) {
    char hex[16];
    std::snprintf(hex, sizeof(hex), "0x%x", static_cast<unsigned>(r));
    throw std::runtime_error(std::string(what) + " failed: " + hex);
  }
}

ze_context_handle_t ze_context() {
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      c10::xpu::get_device_context());
}

ze_device_handle_t ze_device(c10::DeviceIndex index) {
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      c10::xpu::get_raw_device(index));
}

class PeerBuffer {
 public:
  explicit PeerBuffer(int64_t nbytes) : nbytes_(nbytes) {
    if (nbytes <= 0) throw std::invalid_argument("nbytes must be positive");
    device_ = c10::xpu::current_device();
    ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    check(zeMemAllocDevice(ze_context(), &desc, nbytes, 64, ze_device(device_),
                           &ptr_),
          "zeMemAllocDevice");
  }
  ~PeerBuffer() { close(); }

  py::bytes ipc_handle() {
    live();
    ze_ipc_mem_handle_t h = {};
    check(zeMemGetIpcHandle(ze_context(), ptr_, &h), "zeMemGetIpcHandle");
    return py::bytes(h.data, ZE_MAX_IPC_HANDLE_SIZE);
  }

  // The whole buffer as uint8; valid until close().
  torch::Tensor tensor() {
    live();
    return torch::from_blob(
        ptr_, {nbytes_}, [](void*) {},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kXPU, device_));
  }

  int64_t nbytes() const { return nbytes_; }

  void close() {
    if (ptr_) {
      zeMemFree(ze_context(), ptr_);
      ptr_ = nullptr;
    }
  }

 private:
  void live() const {
    if (!ptr_) throw std::runtime_error("PeerBuffer is closed");
  }
  int64_t nbytes_;
  c10::DeviceIndex device_;
  void* ptr_ = nullptr;
};

class PeerMapping {
 public:
  PeerMapping(const py::bytes& handle, int64_t nbytes) : nbytes_(nbytes) {
    std::string h = handle;
    if (h.size() != ZE_MAX_IPC_HANDLE_SIZE)
      throw std::invalid_argument("IPC handle must be 64 bytes");
    ze_ipc_mem_handle_t ipc = {};
    std::memcpy(ipc.data, h.data(), ZE_MAX_IPC_HANDLE_SIZE);
    device_ = c10::xpu::current_device();
    check(zeMemOpenIpcHandle(ze_context(), ze_device(device_), ipc, 0, &ptr_),
          "zeMemOpenIpcHandle");
    flush_ = sycl::malloc_device<uint8_t>(64, c10::xpu::getCurrentXPUStream().queue());
  }
  ~PeerMapping() { close(); }

  // Enqueue a copy of src into the peer buffer at offset, on the current
  // queue, then a small read back from the peer: PCIe does not let a read
  // pass earlier posted writes on the same path, so once the queue has
  // drained, the data is in the peer's memory, not still in flight.
  void write(const torch::Tensor& src, int64_t offset) {
    if (!ptr_) throw std::runtime_error("PeerMapping is closed");
    if (!src.device().is_xpu() || src.device().index() != device_)
      throw std::invalid_argument("src must be on the current XPU device");
    if (!src.is_contiguous()) throw std::invalid_argument("src must be contiguous");
    int64_t n = src.numel() * src.element_size();
    if (offset < 0 || n > nbytes_ || offset > nbytes_ - n)
      throw std::invalid_argument("write past the end of the peer buffer");
    if (n == 0) return;
    auto& q = c10::xpu::getCurrentXPUStream().queue();
    auto* dst = static_cast<uint8_t*>(ptr_) + offset;
    q.memcpy(dst, src.data_ptr(), n);
    int64_t tail = n < 64 ? n : 64;
    q.memcpy(flush_, dst + n - tail, tail);
  }

  int64_t nbytes() const { return nbytes_; }

  void close() {
    if (ptr_) {
      auto& q = c10::xpu::getCurrentXPUStream().queue();
      q.wait();
      zeMemCloseIpcHandle(ze_context(), ptr_);
      ptr_ = nullptr;
      sycl::free(flush_, q);
      flush_ = nullptr;
    }
  }

 private:
  int64_t nbytes_;
  c10::DeviceIndex device_;
  void* ptr_ = nullptr;
  uint8_t* flush_ = nullptr;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<PeerBuffer>(m, "PeerBuffer")
      .def(py::init<int64_t>(), py::arg("nbytes"))
      .def("ipc_handle", &PeerBuffer::ipc_handle)
      .def("tensor", &PeerBuffer::tensor)
      .def_property_readonly("nbytes", &PeerBuffer::nbytes)
      .def("close", &PeerBuffer::close);
  py::class_<PeerMapping>(m, "PeerMapping")
      .def(py::init<const py::bytes&, int64_t>(), py::arg("handle"),
           py::arg("nbytes"))
      .def("write", &PeerMapping::write, py::arg("src"), py::arg("offset"))
      .def_property_readonly("nbytes", &PeerMapping::nbytes)
      .def("close", &PeerMapping::close);
}
