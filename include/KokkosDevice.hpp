#pragma once

#include <Kokkos_Core.hpp>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using Device = typename ExecutionSpace::device_type;
using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using HostDevice = typename HostExecutionSpace::device_type;

template <typename DeviceType = Device>
struct DeviceTraits {
    using execution_space = typename DeviceType::execution_space;
    using memory_space = typename DeviceType::memory_space;
    using array_layout = typename execution_space::array_layout;
};
