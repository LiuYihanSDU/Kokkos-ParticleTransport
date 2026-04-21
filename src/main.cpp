#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "AthenaFieldReader.hpp"
#include "Field.hpp"
#include "FieldInterpolator.hpp"
#include "KokkosDevice.hpp"
#include "RandomManager.hpp"

namespace Reconnection2DCalibration {

inline constexpr int parker_coefficient_component_count = 16;
inline constexpr int coefficient_component_count = 21;
inline constexpr int reconnection_particle_space_dim = 2;
inline constexpr std::int32_t reconnection_particle_custom_species = 3;

enum class TransportModel {
    Parker,
    Focused
};

enum CoefficientComponent : int {
    CoeffVx = 0,
    CoeffVy = 1,
    CoeffVz = 2,
    CoeffBx = 3,
    CoeffBy = 4,
    CoeffBz = 5,
    CoeffBmag = 6,
    CoeffDvxDx = 7,
    CoeffDvxDy = 8,
    CoeffDvyDx = 9,
    CoeffDvyDy = 10,
    CoeffDvzDx = 11,
    CoeffDvzDy = 12,
    CoeffDbxDx = 13,
    CoeffDbxDy = 14,
    CoeffDbyDx = 15,
    CoeffDbyDy = 16,
    CoeffDbzDx = 17,
    CoeffDbzDy = 18,
    CoeffDbmagDx = 19,
    CoeffDbmagDy = 20
};

enum ParkerCoefficientComponent : int {
    ParkerCoeffVx = 0,
    ParkerCoeffVy = 1,
    ParkerCoeffBx = 2,
    ParkerCoeffBy = 3,
    ParkerCoeffBz = 4,
    ParkerCoeffBmag = 5,
    ParkerCoeffDvxDx = 6,
    ParkerCoeffDvyDy = 7,
    ParkerCoeffDbxDx = 8,
    ParkerCoeffDbxDy = 9,
    ParkerCoeffDbyDx = 10,
    ParkerCoeffDbyDy = 11,
    ParkerCoeffDbzDx = 12,
    ParkerCoeffDbzDy = 13,
    ParkerCoeffDbmagDx = 14,
    ParkerCoeffDbmagDy = 15
};

/**
 * Return a display string for a transport model.
 */
std::string transport_model_name(const TransportModel model) {
    switch (model) {
    case TransportModel::Parker:
        return "parker";
    case TransportModel::Focused:
        return "focused";
    }
    return "unknown";
}

/**
 * Parse a command-line transport model name.
 */
TransportModel parse_transport_model_name(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
                   [](const unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (name == "parker" || name == "pte") {
        return TransportModel::Parker;
    }
    if (name == "focused" || name == "focus" || name == "fte") {
        return TransportModel::Focused;
    }
    throw std::runtime_error("Unknown transport model: " + name + ".");
}

/**
 * Host-side controls for the Li Xiaocan reconnection_2d calibration run.
 */
struct Reconnection2DSettings {
    std::string profile_name{"particle-64000"};
    std::string field_dir{"cmake-build-debug/field_fortran_check"};
    std::string output_dir{"cmake-build-debug/reconnection_particle_64000"};
    std::string restart_particle_snapshot_path{};
    TransportModel transport_model{TransportModel::Parker};
    int start_frame{0};
    int end_frame{2};
    std::uint64_t particles_per_frame{64000};
    std::uint64_t rank_scale{1};
    std::uint64_t particle_capacity{1000000};
    std::uint64_t seed{114514};
    int diagnostic_interval{1};
    int histogram_interval{1};
    int particle_snapshot_interval{1};
    int max_particle_steps_per_interval{200000};
    bool split_particles{true};
    bool time_interpolation{true};
    bool overwrite_output{false};
    double walltime_limit_seconds{0.0};
    double walltime_reserve_seconds{0.0};

    double mhd_dt{0.1};
    double p0{0.1};
    double pmin{1.0e-2};
    double pmax{1.0e1};
    double gamma_turb{1.6666667};
    int momentum_dependency{1};
    int magnetic_dependency{1};
    double kpara0{0.00743592};
    double kperp_over_kpara{0.01};
    double dt_min_rel{1.0e-7};
    double dt_max_rel{1.0e-2};
    double drift_param1{850964.408};
    double drift_param2{13575468.975};
    double charge{-1.0};
    double particle_v0{17.20195};
    double duu0{5578.445};
    double split_ratio{2.0};
    double pmin_split_over_p0{2.0};
    double mu_max{0.99};

    /**
     * Return the number of MHD intervals advanced by this run.
     */
    int interval_count() const {
        return end_frame - start_frame;
    }

    /**
     * Return whether this run should stop after reaching a wall-clock limit.
     */
    bool walltime_limit_enabled() const {
        return walltime_limit_seconds > 0.0;
    }

    /**
     * Return whether this run should initialize particles from a frame-boundary snapshot.
     */
    bool restart_enabled() const {
        return !restart_particle_snapshot_path.empty();
    }

    /**
     * Return the elapsed seconds at which the frame loop should stop.
     */
    double walltime_stop_seconds() const {
        return std::max(0.0, walltime_limit_seconds - walltime_reserve_seconds);
    }

    /**
     * Validate host-side controls before launching kernels.
     */
    void validate() const {
        if (end_frame <= start_frame) {
            throw std::runtime_error("Reconnection2DSettings: end frame must exceed start frame.");
        }
        if (start_frame < 0) {
            throw std::runtime_error("Reconnection2DSettings: start frame must be non-negative.");
        }
        if (restart_enabled() &&
            !std::filesystem::exists(restart_particle_snapshot_path)) {
            throw std::runtime_error(
                "Reconnection2DSettings: restart particle snapshot does not exist.");
        }
        if (restart_enabled() && overwrite_output) {
            throw std::runtime_error(
                "Reconnection2DSettings: --overwrite-output cannot be combined with restart.");
        }
        if (particles_per_frame == 0 || rank_scale == 0 || particle_capacity == 0) {
            throw std::runtime_error(
                "Reconnection2DSettings: particle counts and capacity must be positive.");
        }
        if (diagnostic_interval <= 0 || particle_snapshot_interval < 0 ||
            max_particle_steps_per_interval <= 0) {
            throw std::runtime_error(
                "Reconnection2DSettings: diagnostic interval and step guard must be "
                "positive, and particle snapshot interval must be non-negative.");
        }
        if (!std::isfinite(walltime_limit_seconds) ||
            walltime_limit_seconds < 0.0 ||
            !std::isfinite(walltime_reserve_seconds) ||
            walltime_reserve_seconds < 0.0 ||
            (walltime_limit_seconds <= 0.0 && walltime_reserve_seconds > 0.0)) {
            throw std::runtime_error(
                "Reconnection2DSettings: walltime limit must be non-negative, "
                "and reserve requires a positive walltime limit.");
        }
        if (!std::isfinite(mhd_dt) || mhd_dt <= 0.0 ||
            !std::isfinite(p0) || p0 <= 0.0 ||
            !std::isfinite(pmin) || pmin <= 0.0 ||
            !std::isfinite(pmax) || pmax <= pmin ||
            !std::isfinite(gamma_turb) ||
            !std::isfinite(kpara0) || kpara0 <= 0.0 ||
            !std::isfinite(kperp_over_kpara) || kperp_over_kpara < 0.0 ||
            !std::isfinite(dt_min_rel) || dt_min_rel <= 0.0 ||
            !std::isfinite(dt_max_rel) || dt_max_rel < dt_min_rel ||
            !std::isfinite(drift_param1) || !std::isfinite(drift_param2) ||
            !std::isfinite(charge) || charge == 0.0 ||
            !std::isfinite(particle_v0) || particle_v0 <= 0.0 ||
            !std::isfinite(duu0) || duu0 < 0.0 ||
            !std::isfinite(split_ratio) || split_ratio <= 1.0 ||
            !std::isfinite(pmin_split_over_p0) || pmin_split_over_p0 <= 0.0 ||
            !std::isfinite(mu_max) || mu_max <= 0.0) {
            throw std::runtime_error("Reconnection2DSettings: invalid physical parameters.");
        }
    }
};

/**
 * Device-copyable subset of calibration controls used by particle kernels.
 */
struct Reconnection2DDeviceSettings {
    int max_particle_steps_per_interval{200000};
    int momentum_dependency{1};
    int magnetic_dependency{1};
    int split_particles{1};
    int time_interpolation{1};
    int focused_transport{0};

    double x_lower{0.0};
    double x_upper{1.0};
    double y_lower{0.0};
    double y_upper{1.0};
    double x_width{1.0};
    double y_width{1.0};
    double dx{1.0};
    double dy{1.0};
    double interpolation_shift_x{0.5};
    double interpolation_shift_y{0.5};

    double mhd_dt{0.1};
    double p0{0.1};
    double pmax{1.0e1};
    double gamma_turb{1.6666667};
    double pindex{4.0 / 3.0};
    double kpara0{0.00743592};
    double kperp_over_kpara{0.01};
    double dt_min{1.0e-8};
    double dt_max{1.0e-3};
    double drift_param1{850964.408};
    double drift_param2{13575468.975};
    double charge{-1.0};
    double particle_v0{17.20195};
    double duu0{5578.445};
    double pitch_angle_scattering_h0{0.2};
    double split_ratio{2.0};
    double pmin_split{0.2};
    double mu_max{0.99};
};

/**
 * Device-backed particle storage for the Fortran-style 2D calibration driver.
 */
template <
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout
>
struct ReconnectionParticleStorage {
    using execution_space = typename DeviceTraits<DeviceType>::execution_space;
    using memory_space = typename DeviceTraits<DeviceType>::memory_space;
    using layout_type = LayoutType;
    using position_view_type =
        Kokkos::View<double*[reconnection_particle_space_dim], layout_type, memory_space>;
    using scalar_view_type = Kokkos::View<double*, layout_type, memory_space>;
    using int_view_type = Kokkos::View<int*, layout_type, memory_space>;
    using size_type = typename scalar_view_type::size_type;

    std::uint64_t count{0};
    std::uint64_t capacity{0};
    position_view_type position;
    scalar_view_type momentum;
    scalar_view_type time;
    scalar_view_type step_dt;
    scalar_view_type weight;
    scalar_view_type mu;
    int_view_type status;
    int_view_type split_level;

    ReconnectionParticleStorage() = default;

    /**
     * Allocate particle arrays with an initially empty active range.
     */
    explicit ReconnectionParticleStorage(const std::uint64_t input_capacity)
        : count(0),
          capacity(input_capacity),
          position("reconnection_position", input_capacity),
          momentum("reconnection_momentum", input_capacity),
          time("reconnection_time", input_capacity),
          step_dt("reconnection_step_dt", input_capacity),
          weight("reconnection_weight", input_capacity),
          mu("reconnection_mu", input_capacity),
          status("reconnection_status", input_capacity),
          split_level("reconnection_split_level", input_capacity) {
        Kokkos::deep_copy(status, 2);
        Kokkos::deep_copy(split_level, 0);
        Kokkos::deep_copy(weight, 0.0);
    }
};

/**
 * Host-side particle statistics written to the calibration summary.
 */
struct ParticleSummary {
    std::uint64_t stored{0};
    std::uint64_t active{0};
    std::uint64_t inactive{0};
    double weight_sum{0.0};
    double p_min{0.0};
    double p_max{0.0};
    double p_avg{0.0};
    double dt_min{0.0};
    double dt_max{0.0};
    double dt_avg{0.0};
    int max_split_level{0};
};

/**
 * Per-frame timing and split counters.
 */
struct FrameTiming {
    double read_seconds{0.0};
    double inject_seconds{0.0};
    double move_seconds{0.0};
    double split_seconds{0.0};
    std::uint64_t injected{0};
    std::uint64_t split{0};
    std::uint64_t skipped_split_capacity{0};
};

template <typename BackgroundType>
using ParkerReconnectionCoefficientField =
    Field<typename BackgroundType::grid_type,
          parker_coefficient_component_count,
          typename BackgroundType::layout_type>;

template <typename BackgroundType>
using ReconnectionCoefficientField =
    Field<typename BackgroundType::grid_type,
          coefficient_component_count,
          typename BackgroundType::layout_type>;

/**
 * Return a wall-clock timestamp for coarse host timing.
 */
inline auto wall_time_now() {
    return std::chrono::steady_clock::now();
}

/**
 * Return seconds elapsed since a wall-clock timestamp.
 */
template <typename TimePoint>
double elapsed_seconds_since(const TimePoint& start) {
    return std::chrono::duration<double>(wall_time_now() - start).count();
}

/**
 * Write a log block to the terminal and to the run log.
 */
void write_log_block(std::ofstream& log_stream, const std::string& block) {
    std::cout << block;
    if (log_stream) {
        log_stream << block;
        log_stream.flush();
    }
}

/**
 * Write a log line to the terminal and to the run log.
 */
void write_log_line(std::ofstream& log_stream, const std::string& line) {
    write_log_block(log_stream, line + '\n');
}

/**
 * Return a zero-padded compact Athena field path for one frame.
 */
std::string field_file_path(const std::string& field_dir, const int frame) {
    std::ostringstream path;
    path << field_dir << "/field" << std::setw(5) << std::setfill('0') << frame
         << ".bin";
    return path.str();
}

/**
 * Return a zero-padded particle snapshot path for one output frame.
 */
std::string particle_snapshot_file_path(const std::string& output_dir,
                                        const int frame) {
    std::ostringstream path;
    path << output_dir << "/particles_" << std::setw(5) << std::setfill('0')
         << frame << ".bin";
    return path.str();
}

/**
 * Return whether a file name is produced by the reconnection calibration driver.
 */
bool is_reconnection_output_artifact(const std::string& name) {
    constexpr std::string_view particle_prefix{"particles_"};
    constexpr std::string_view histogram_prefix{"momentum_histogram_"};
    constexpr std::string_view csv_suffix{".csv"};
    constexpr std::string_view bin_suffix{".bin"};
    if (name == "summary.csv" || name == "run.log") {
        return true;
    }
    return ((name.size() > particle_prefix.size() + bin_suffix.size()) &&
            name.compare(0, particle_prefix.size(), particle_prefix) == 0 &&
            name.compare(name.size() - bin_suffix.size(), bin_suffix.size(),
                         bin_suffix) == 0) ||
           ((name.size() > histogram_prefix.size() + csv_suffix.size()) &&
            name.compare(0, histogram_prefix.size(), histogram_prefix) == 0 &&
            name.compare(name.size() - csv_suffix.size(), csv_suffix.size(),
                         csv_suffix) == 0);
}

/**
 * Return whether the output directory already contains solver-owned artifacts.
 */
bool output_directory_has_reconnection_outputs(const std::string& output_dir) {
    const std::filesystem::path directory(output_dir);
    if (!std::filesystem::exists(directory)) {
        return false;
    }
    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error(
            "Reconnection output path exists but is not a directory.");
    }
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (is_reconnection_output_artifact(entry.path().filename().string())) {
            return true;
        }
    }
    return false;
}

/**
 * Reject accidental overwrites of existing non-restart solver output.
 */
void require_fresh_output_directory(const Reconnection2DSettings& settings) {
    if (settings.restart_enabled() || settings.overwrite_output) {
        return;
    }
    if (output_directory_has_reconnection_outputs(settings.output_dir)) {
        throw std::runtime_error(
            "Output directory already contains reconnection solver outputs. "
            "Use --restart-particle-snapshot to continue a completed-frame "
            "snapshot, choose a new --output-dir, or pass --overwrite-output "
            "to replace the existing run.");
    }
}

/**
 * Infer the completed frame number from a repository particle snapshot file name.
 */
int infer_particle_snapshot_frame(const std::string& file_path) {
    const std::string name = std::filesystem::path(file_path).filename().string();
    constexpr std::string_view prefix{"particles_"};
    constexpr std::string_view suffix{".bin"};
    if (name.size() <= prefix.size() + suffix.size() ||
        name.compare(0, prefix.size(), prefix) != 0 ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return -1;
    }
    const std::string digits =
        name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
    if (digits.empty() ||
        !std::all_of(digits.begin(), digits.end(), [](const unsigned char c) {
            return std::isdigit(c) != 0;
        })) {
        return -1;
    }
    return std::stoi(digits);
}

/**
 * Return a field directory for the short 64000-particle test available on this
 * workstation. The Fortran-derived directory is preferred because it is known to
 * match the reference `mhd_data_0000..0002` snapshots.
 */
std::string default_particle_test_field_dir() {
    const std::string fortran_check_dir{"cmake-build-debug/field_fortran_check"};
    if (std::filesystem::exists(field_file_path(fortran_check_dir, 0)) &&
        std::filesystem::exists(field_file_path(fortran_check_dir, 2))) {
        return fortran_check_dir;
    }
    return "cmake-build-debug/field";
}

/**
 * Apply a named host-side run profile before command-line overrides.
 */
void apply_run_profile(Reconnection2DSettings& settings,
                       const std::string& profile_name) {
    settings = Reconnection2DSettings{};
    settings.profile_name = profile_name;

    if (profile_name == "particle-64000" || profile_name == "particle-test") {
        settings.field_dir = default_particle_test_field_dir();
        settings.output_dir = "cmake-build-debug/reconnection_particle_64000";
        settings.start_frame = 0;
        settings.end_frame = 2;
        settings.particles_per_frame = 64000;
        settings.rank_scale = 1;
        settings.particle_capacity = 1000000;
        settings.diagnostic_interval = 1;
        settings.histogram_interval = 1;
        settings.particle_snapshot_interval = 1;
        return;
    }

    if (profile_name == "smoke") {
        settings.field_dir = default_particle_test_field_dir();
        settings.output_dir = "cmake-build-debug/reconnection_smoke";
        settings.start_frame = 0;
        settings.end_frame = 1;
        settings.particles_per_frame = 1024;
        settings.rank_scale = 1;
        settings.particle_capacity = 8192;
        settings.diagnostic_interval = 1;
        settings.histogram_interval = 1;
        settings.particle_snapshot_interval = 1;
        return;
    }

    if (profile_name == "fortran-rank") {
        settings.field_dir = "cmake-build-debug/field";
        settings.output_dir = "cmake-build-debug/reconnection_fortran_rank";
        settings.start_frame = 0;
        settings.end_frame = 200;
        settings.particles_per_frame = 1600;
        settings.rank_scale = 1;
        settings.particle_capacity = 1000000;
        settings.diagnostic_interval = 1;
        settings.histogram_interval = 10;
        settings.particle_snapshot_interval = 10;
        return;
    }

    if (profile_name == "fortran-global" || profile_name == "full") {
        settings.field_dir = "cmake-build-debug/field";
        settings.output_dir = "cmake-build-debug/reconnection_fortran_global";
        settings.start_frame = 0;
        settings.end_frame = 200;
        settings.particles_per_frame = 1600;
        settings.rank_scale = 16;
        settings.particle_capacity = 16000000;
        settings.diagnostic_interval = 1;
        settings.histogram_interval = 10;
        settings.particle_snapshot_interval = 10;
        return;
    }

    throw std::runtime_error("Unknown run profile: " + profile_name + ".");
}

/**
 * Return a device settings object derived from host controls and the loaded grid.
 */
template <typename GridType>
Reconnection2DDeviceSettings make_device_settings(
    const Reconnection2DSettings& settings,
    const GridType& grid) {
    Reconnection2DDeviceSettings device_settings;
    device_settings.max_particle_steps_per_interval =
        settings.max_particle_steps_per_interval;
    device_settings.momentum_dependency = settings.momentum_dependency;
    device_settings.magnetic_dependency = settings.magnetic_dependency;
    device_settings.split_particles = settings.split_particles ? 1 : 0;
    device_settings.time_interpolation = settings.time_interpolation ? 1 : 0;
    device_settings.focused_transport =
        settings.transport_model == TransportModel::Focused ? 1 : 0;
    device_settings.x_lower = grid.physical_lower_bound[0];
    device_settings.x_upper = grid.physical_upper_bound[0];
    device_settings.y_lower = grid.physical_lower_bound[1];
    device_settings.y_upper = grid.physical_upper_bound[1];
    device_settings.x_width = device_settings.x_upper - device_settings.x_lower;
    device_settings.y_width = device_settings.y_upper - device_settings.y_lower;
    device_settings.dx = device_settings.x_width / static_cast<double>(grid.extent(0));
    device_settings.dy = device_settings.y_width / static_cast<double>(grid.extent(1));
    device_settings.interpolation_shift_x = 0.5 * device_settings.dx;
    device_settings.interpolation_shift_y = 0.5 * device_settings.dy;
    device_settings.mhd_dt = settings.mhd_dt;
    device_settings.p0 = settings.p0;
    device_settings.pmax = settings.pmax;
    device_settings.gamma_turb = settings.gamma_turb;
    device_settings.pindex = 3.0 - settings.gamma_turb;
    device_settings.kpara0 = settings.kpara0;
    device_settings.kperp_over_kpara = settings.kperp_over_kpara;
    device_settings.dt_min = settings.dt_min_rel * settings.mhd_dt;
    device_settings.dt_max = settings.dt_max_rel * settings.mhd_dt;
    device_settings.drift_param1 = settings.drift_param1;
    device_settings.drift_param2 = settings.drift_param2;
    device_settings.charge = settings.charge;
    device_settings.particle_v0 = settings.particle_v0;
    device_settings.duu0 = settings.duu0;
    device_settings.split_ratio = settings.split_ratio;
    device_settings.pmin_split = settings.pmin_split_over_p0 * settings.p0;
    device_settings.mu_max = settings.mu_max;
    return device_settings;
}

/**
 * Wrap a periodic coordinate into [lower, upper).
 */
KOKKOS_INLINE_FUNCTION
double wrap_periodic_coordinate(const double value,
                                const double lower,
                                const double upper,
                                const double width) {
    double wrapped = value;
    if (wrapped >= upper) {
        wrapped -= width;
        if (wrapped >= upper) {
            wrapped -= Kokkos::floor((wrapped - lower) / width) * width;
        }
    } else if (wrapped < lower) {
        wrapped += width;
        if (wrapped < lower) {
            wrapped -= Kokkos::floor((wrapped - lower) / width) * width;
        }
    }
    if (wrapped >= upper) {
        wrapped -= width;
    }
    if (wrapped < lower) {
        wrapped += width;
    }
    return wrapped;
}

/**
 * Return the smaller of two positive finite values.
 */
KOKKOS_INLINE_FUNCTION
double min_positive(const double left, const double right) {
    return left < right ? left : right;
}

/**
 * Return the square of a scalar without dispatching to the general pow path.
 */
KOKKOS_INLINE_FUNCTION
double square(const double value) {
    return value * value;
}

/**
 * Clamp a pitch-angle cosine to the Fortran focused-transport interval.
 */
KOKKOS_INLINE_FUNCTION
double clamp_pitch_angle_cosine(const double value, const double maximum_abs_mu) {
    if (value > maximum_abs_mu) {
        return maximum_abs_mu;
    }
    if (value < -maximum_abs_mu) {
        return -maximum_abs_mu;
    }
    return value;
}

/**
 * Return |B| at one field index.
 */
template <typename MagneticFieldType>
KOKKOS_INLINE_FUNCTION
double magnetic_magnitude_at(
    const MagneticFieldType& magnetic_field,
    const typename MagneticFieldType::index_array_type& indices) {
    const double bx = magnetic_field(indices, 0);
    const double by = magnetic_field(indices, 1);
    const double bz = magnetic_field(indices, 2);
    return Kokkos::sqrt(bx * bx + by * by + bz * bz);
}

/**
 * Return one finite-difference derivative of a vector-field component.
 */
template <typename VectorFieldType>
KOKKOS_INLINE_FUNCTION
double derivative_component(
    const VectorFieldType& field,
    typename VectorFieldType::index_array_type indices,
    const int derivative_dim,
    const int component) {
    if (derivative_dim >= VectorFieldType::space_dim ||
        field.grid.extent(derivative_dim) <= 1) {
        return 0.0;
    }

    const int lower = field.lower_index(derivative_dim, GridDomain::GhostedDomain);
    const int upper = field.upper_index(derivative_dim, GridDomain::GhostedDomain);
    auto left = indices;
    auto right = indices;

    if (indices[derivative_dim] <= lower) {
        auto next = indices;
        auto next2 = indices;
        next[derivative_dim] = lower + 1;
        next2[derivative_dim] = lower + 2;
        indices[derivative_dim] = lower;
        const double x0 = field.grid.coordinate(derivative_dim, lower,
                                                field.centering(derivative_dim));
        const double x2 = field.grid.coordinate(derivative_dim, lower + 2,
                                                field.centering(derivative_dim));
        return (-3.0 * field(indices, component) +
                4.0 * field(next, component) -
                field(next2, component)) /
               (x2 - x0);
    }

    if (indices[derivative_dim] >= upper) {
        auto prev = indices;
        auto prev2 = indices;
        prev[derivative_dim] = upper - 1;
        prev2[derivative_dim] = upper - 2;
        indices[derivative_dim] = upper;
        const double x0 = field.grid.coordinate(derivative_dim, upper - 2,
                                                field.centering(derivative_dim));
        const double x2 = field.grid.coordinate(derivative_dim, upper,
                                                field.centering(derivative_dim));
        return (3.0 * field(indices, component) -
                4.0 * field(prev, component) +
                field(prev2, component)) /
               (x2 - x0);
    }

    left[derivative_dim] -= 1;
    right[derivative_dim] += 1;
    const double xl = field.grid.coordinate(derivative_dim, left[derivative_dim],
                                            field.centering(derivative_dim));
    const double xr = field.grid.coordinate(derivative_dim, right[derivative_dim],
                                            field.centering(derivative_dim));
    return (field(right, component) - field(left, component)) / (xr - xl);
}

/**
 * Return one finite-difference derivative of |B|.
 */
template <typename MagneticFieldType>
KOKKOS_INLINE_FUNCTION
double derivative_magnetic_magnitude(
    const MagneticFieldType& magnetic_field,
    typename MagneticFieldType::index_array_type indices,
    const int derivative_dim) {
    if (derivative_dim >= MagneticFieldType::space_dim ||
        magnetic_field.grid.extent(derivative_dim) <= 1) {
        return 0.0;
    }

    const int lower =
        magnetic_field.lower_index(derivative_dim, GridDomain::GhostedDomain);
    const int upper =
        magnetic_field.upper_index(derivative_dim, GridDomain::GhostedDomain);
    auto left = indices;
    auto right = indices;

    if (indices[derivative_dim] <= lower) {
        auto next = indices;
        auto next2 = indices;
        next[derivative_dim] = lower + 1;
        next2[derivative_dim] = lower + 2;
        indices[derivative_dim] = lower;
        const double x0 = magnetic_field.grid.coordinate(
            derivative_dim, lower, magnetic_field.centering(derivative_dim));
        const double x2 = magnetic_field.grid.coordinate(
            derivative_dim, lower + 2, magnetic_field.centering(derivative_dim));
        return (-3.0 * magnetic_magnitude_at(magnetic_field, indices) +
                4.0 * magnetic_magnitude_at(magnetic_field, next) -
                magnetic_magnitude_at(magnetic_field, next2)) /
               (x2 - x0);
    }

    if (indices[derivative_dim] >= upper) {
        auto prev = indices;
        auto prev2 = indices;
        prev[derivative_dim] = upper - 1;
        prev2[derivative_dim] = upper - 2;
        indices[derivative_dim] = upper;
        const double x0 = magnetic_field.grid.coordinate(
            derivative_dim, upper - 2, magnetic_field.centering(derivative_dim));
        const double x2 = magnetic_field.grid.coordinate(
            derivative_dim, upper, magnetic_field.centering(derivative_dim));
        return (3.0 * magnetic_magnitude_at(magnetic_field, indices) -
                4.0 * magnetic_magnitude_at(magnetic_field, prev) +
                magnetic_magnitude_at(magnetic_field, prev2)) /
               (x2 - x0);
    }

    left[derivative_dim] -= 1;
    right[derivative_dim] += 1;
    const double xl = magnetic_field.grid.coordinate(
        derivative_dim, left[derivative_dim], magnetic_field.centering(derivative_dim));
    const double xr = magnetic_field.grid.coordinate(
        derivative_dim, right[derivative_dim], magnetic_field.centering(derivative_dim));
    return (magnetic_magnitude_at(magnetic_field, right) -
            magnetic_magnitude_at(magnetic_field, left)) /
           (xr - xl);
}

/**
 * Precompute the field values and gradients consumed by the Parker 2D pusher.
 */
template <typename BackgroundType>
ParkerReconnectionCoefficientField<BackgroundType> make_parker_reconnection_coefficients(
    const BackgroundType& background,
    const char* label) {
    using coefficient_field_type = ParkerReconnectionCoefficientField<BackgroundType>;
    using execution_space = typename coefficient_field_type::execution_space;
    coefficient_field_type coefficients(background.grid, label,
                                        GridCentering::CellCentered);

    const auto magnetic_field = background.magnetic_field;
    const auto velocity_field = background.velocity_field;
    const auto local_coefficients = coefficients;
    auto coefficient_data = coefficients.data;
    Kokkos::parallel_for(
        "Reconnection2DCalibration::make_parker_reconnection_coefficients",
        Kokkos::RangePolicy<execution_space>(0, coefficients.point_count()),
        KOKKOS_LAMBDA(const typename coefficient_field_type::size_type point_index) {
            const auto indices = local_coefficients.logical_indices(point_index);
            const double bx = magnetic_field(indices, 0);
            const double by = magnetic_field(indices, 1);
            const double bz = magnetic_field(indices, 2);
            coefficient_data(point_index, ParkerCoeffVx) =
                velocity_field(indices, 0);
            coefficient_data(point_index, ParkerCoeffVy) =
                velocity_field(indices, 1);
            coefficient_data(point_index, ParkerCoeffBx) = bx;
            coefficient_data(point_index, ParkerCoeffBy) = by;
            coefficient_data(point_index, ParkerCoeffBz) = bz;
            coefficient_data(point_index, ParkerCoeffBmag) =
                Kokkos::sqrt(bx * bx + by * by + bz * bz);
            coefficient_data(point_index, ParkerCoeffDvxDx) =
                derivative_component(velocity_field, indices, 0, 0);
            coefficient_data(point_index, ParkerCoeffDvyDy) =
                derivative_component(velocity_field, indices, 1, 1);
            coefficient_data(point_index, ParkerCoeffDbxDx) =
                derivative_component(magnetic_field, indices, 0, 0);
            coefficient_data(point_index, ParkerCoeffDbxDy) =
                derivative_component(magnetic_field, indices, 1, 0);
            coefficient_data(point_index, ParkerCoeffDbyDx) =
                derivative_component(magnetic_field, indices, 0, 1);
            coefficient_data(point_index, ParkerCoeffDbyDy) =
                derivative_component(magnetic_field, indices, 1, 1);
            coefficient_data(point_index, ParkerCoeffDbzDx) =
                derivative_component(magnetic_field, indices, 0, 2);
            coefficient_data(point_index, ParkerCoeffDbzDy) =
                derivative_component(magnetic_field, indices, 1, 2);
            coefficient_data(point_index, ParkerCoeffDbmagDx) =
                derivative_magnetic_magnitude(magnetic_field, indices, 0);
            coefficient_data(point_index, ParkerCoeffDbmagDy) =
                derivative_magnetic_magnitude(magnetic_field, indices, 1);
        });
    Kokkos::fence("Reconnection2DCalibration::make_parker_reconnection_coefficients");
    return coefficients;
}

/**
 * Precompute the field values and gradients consumed by the focused 2D pusher.
 */
template <typename BackgroundType>
ReconnectionCoefficientField<BackgroundType> make_reconnection_coefficients(
    const BackgroundType& background,
    const char* label) {
    using coefficient_field_type = ReconnectionCoefficientField<BackgroundType>;
    using execution_space = typename coefficient_field_type::execution_space;
    coefficient_field_type coefficients(background.grid, label,
                                        GridCentering::CellCentered);

    const auto magnetic_field = background.magnetic_field;
    const auto velocity_field = background.velocity_field;
    const auto local_coefficients = coefficients;
    auto coefficient_data = coefficients.data;
    Kokkos::parallel_for(
        "Reconnection2DCalibration::make_reconnection_coefficients",
        Kokkos::RangePolicy<execution_space>(0, coefficients.point_count()),
        KOKKOS_LAMBDA(const typename coefficient_field_type::size_type point_index) {
            const auto indices = local_coefficients.logical_indices(point_index);
            const double bx = magnetic_field(indices, 0);
            const double by = magnetic_field(indices, 1);
            const double bz = magnetic_field(indices, 2);
            coefficient_data(point_index, CoeffVx) = velocity_field(indices, 0);
            coefficient_data(point_index, CoeffVy) = velocity_field(indices, 1);
            coefficient_data(point_index, CoeffVz) = velocity_field(indices, 2);
            coefficient_data(point_index, CoeffBx) = bx;
            coefficient_data(point_index, CoeffBy) = by;
            coefficient_data(point_index, CoeffBz) = bz;
            coefficient_data(point_index, CoeffBmag) =
                Kokkos::sqrt(bx * bx + by * by + bz * bz);
            coefficient_data(point_index, CoeffDvxDx) =
                derivative_component(velocity_field, indices, 0, 0);
            coefficient_data(point_index, CoeffDvxDy) =
                derivative_component(velocity_field, indices, 1, 0);
            coefficient_data(point_index, CoeffDvyDx) =
                derivative_component(velocity_field, indices, 0, 1);
            coefficient_data(point_index, CoeffDvyDy) =
                derivative_component(velocity_field, indices, 1, 1);
            coefficient_data(point_index, CoeffDvzDx) =
                derivative_component(velocity_field, indices, 0, 2);
            coefficient_data(point_index, CoeffDvzDy) =
                derivative_component(velocity_field, indices, 1, 2);
            coefficient_data(point_index, CoeffDbxDx) =
                derivative_component(magnetic_field, indices, 0, 0);
            coefficient_data(point_index, CoeffDbxDy) =
                derivative_component(magnetic_field, indices, 1, 0);
            coefficient_data(point_index, CoeffDbyDx) =
                derivative_component(magnetic_field, indices, 0, 1);
            coefficient_data(point_index, CoeffDbyDy) =
                derivative_component(magnetic_field, indices, 1, 1);
            coefficient_data(point_index, CoeffDbzDx) =
                derivative_component(magnetic_field, indices, 0, 2);
            coefficient_data(point_index, CoeffDbzDy) =
                derivative_component(magnetic_field, indices, 1, 2);
            coefficient_data(point_index, CoeffDbmagDx) =
                derivative_magnetic_magnitude(magnetic_field, indices, 0);
            coefficient_data(point_index, CoeffDbmagDy) =
                derivative_magnetic_magnitude(magnetic_field, indices, 1);
        });
    Kokkos::fence("Reconnection2DCalibration::make_reconnection_coefficients");
    return coefficients;
}

/**
 * Read one compact Athena field and build the coefficient field used by the Parker pusher.
 */
template <typename DeviceType = Device,
          typename LayoutType = typename DeviceTraits<DeviceType>::array_layout>
auto load_parker_coefficient_frame(const std::string& field_dir, const int frame) {
    AthenaFieldIO::AthenaBinaryFieldReadOptions<> options;
    options.periodic[0] = 1;
    options.periodic[1] = 1;
    options.ghost_fill_mode = FieldGhostFillMode::PeriodicOrClamped;
    const auto background =
        AthenaFieldIO::read_athena_binary_background_field<DeviceType, LayoutType>(
            field_file_path(field_dir, frame), options);
    return make_parker_reconnection_coefficients(background,
                                                 "parker_reconnection_coefficients");
}

/**
 * Read one compact Athena field and build the coefficient field used by the focused pusher.
 */
template <typename DeviceType = Device,
          typename LayoutType = typename DeviceTraits<DeviceType>::array_layout>
auto load_coefficient_frame(const std::string& field_dir, const int frame) {
    AthenaFieldIO::AthenaBinaryFieldReadOptions<> options;
    options.periodic[0] = 1;
    options.periodic[1] = 1;
    options.ghost_fill_mode = FieldGhostFillMode::PeriodicOrClamped;
    const auto background =
        AthenaFieldIO::read_athena_binary_background_field<DeviceType, LayoutType>(
            field_file_path(field_dir, frame), options);
    return make_reconnection_coefficients(background, "reconnection_coefficients");
}

/**
 * Inject one append-style batch using the Fortran reconnection_2d distribution.
 */
template <typename ParticleStorageType, typename RandomManagerType>
std::uint64_t inject_uniform_particles(
    ParticleStorageType& particles,
    const RandomManagerType& random_manager,
    const Reconnection2DDeviceSettings& settings,
    const std::uint64_t requested_count,
    const double interval_start) {
    if (particles.count >= particles.capacity) {
        return 0;
    }
    const std::uint64_t insert_count =
        std::min(requested_count, particles.capacity - particles.count);
    const std::uint64_t first_index = particles.count;

    auto random_pool = random_manager.pool;
    auto position = particles.position;
    auto momentum = particles.momentum;
    auto time = particles.time;
    auto step_dt = particles.step_dt;
    auto weight = particles.weight;
    auto mu = particles.mu;
    auto status = particles.status;
    auto split_level = particles.split_level;
    const auto local_settings = settings;
    using execution_space = typename ParticleStorageType::execution_space;

    Kokkos::parallel_for(
        "Reconnection2DCalibration::inject_uniform_particles",
        Kokkos::RangePolicy<execution_space>(
            0, static_cast<typename ParticleStorageType::size_type>(insert_count)),
        KOKKOS_LAMBDA(const typename ParticleStorageType::size_type local_index) {
            auto generator = random_pool.get_state();
            const typename ParticleStorageType::size_type particle_index =
                static_cast<typename ParticleStorageType::size_type>(first_index) +
                local_index;
            position(particle_index, 0) =
                generator.drand(local_settings.x_lower, local_settings.x_upper);
            position(particle_index, 1) =
                generator.drand(local_settings.y_lower, local_settings.y_upper);
            momentum(particle_index) = local_settings.p0;
            time(particle_index) =
                interval_start + generator.drand(0.0, local_settings.mhd_dt);
            step_dt(particle_index) = local_settings.dt_min;
            weight(particle_index) = 1.0;
            mu(particle_index) =
                local_settings.mu_max * (2.0 * generator.drand() - 1.0);
            status(particle_index) = 0;
            split_level(particle_index) = 0;
            random_pool.free_state(generator);
        });
    Kokkos::fence("Reconnection2DCalibration::inject_uniform_particles");
    particles.count += insert_count;
    return insert_count;
}

/**
 * Evaluate one interpolated coefficient component array at a particle position and time.
 */
template <typename CoefficientInterpolator>
KOKKOS_INLINE_FUNCTION
typename CoefficientInterpolator::value_array_type sample_time_interpolated_coefficients(
    const CoefficientInterpolator& lower_interpolator,
    const CoefficientInterpolator& upper_interpolator,
    const typename CoefficientInterpolator::coordinate_array_type& position,
    const double rt,
    const int use_time_interpolation) {
    const auto lower_values = lower_interpolator.sample(position);
    if (use_time_interpolation == 0) {
        return lower_values;
    }
    const auto upper_values = upper_interpolator.sample(position);
    typename CoefficientInterpolator::value_array_type values{};
    const double upper_weight = rt < 0.0 ? 0.0 : (rt > 1.0 ? 1.0 : rt);
    const double lower_weight = 1.0 - upper_weight;
    for (int component = 0;
         component < CoefficientInterpolator::field_type::component_dim;
         ++component) {
        values[component] =
            lower_weight * lower_values[component] +
            upper_weight * upper_values[component];
    }
    return values;
}

/**
 * Advance active particles across one MHD interval with the legacy Fortran 2D Parker scheme.
 */
template <typename ParticleStorageType, typename CoefficientFieldType,
          typename RandomManagerType>
void move_parker_particles_one_interval(
    ParticleStorageType& particles,
    const CoefficientFieldType& lower_coefficients,
    const CoefficientFieldType& upper_coefficients,
    const RandomManagerType& random_manager,
    const Reconnection2DDeviceSettings& settings,
    const double interval_start) {
    using execution_space = typename ParticleStorageType::execution_space;
    using coordinate_array_type =
        typename CoefficientFieldType::coordinate_array_type;

    auto random_pool = random_manager.pool;
    auto position = particles.position;
    auto momentum = particles.momentum;
    auto time = particles.time;
    auto step_dt = particles.step_dt;
    auto status = particles.status;
    const auto lower_interpolator =
        FieldInterpolator::make_linear_position_interpolator(
            lower_coefficients, GridDomain::GhostedDomain,
            FieldInterpolator::InterpolationBoundaryPolicy::WrapPeriodic);
    const auto upper_interpolator =
        FieldInterpolator::make_linear_position_interpolator(
            upper_coefficients, GridDomain::GhostedDomain,
            FieldInterpolator::InterpolationBoundaryPolicy::WrapPeriodic);
    const auto local_settings = settings;
    const double interval_end = interval_start + settings.mhd_dt;
    const auto count = particles.count;

    Kokkos::parallel_for(
        "Reconnection2DCalibration::move_parker_particles_one_interval",
        Kokkos::RangePolicy<execution_space>(
            0, static_cast<typename ParticleStorageType::size_type>(count)),
        KOKKOS_LAMBDA(const typename ParticleStorageType::size_type index) {
            if (status(index) != 0) {
                return;
            }

            auto generator = random_pool.get_state();
            double x = position(index, 0);
            double y = position(index, 1);
            double p = momentum(index);
            double t = time(index);
            double dt = step_dt(index);
            double diagnostic_dt = dt;
            int step_guard = 0;

            while (t < interval_end &&
                   step_guard < local_settings.max_particle_steps_per_interval &&
                   status(index) == 0) {
                ++step_guard;
                x = wrap_periodic_coordinate(x, local_settings.x_lower,
                                             local_settings.x_upper,
                                             local_settings.x_width);
                y = wrap_periodic_coordinate(y, local_settings.y_lower,
                                             local_settings.y_upper,
                                             local_settings.y_width);

                coordinate_array_type q{};
                q[0] = x + local_settings.interpolation_shift_x;
                q[1] = y + local_settings.interpolation_shift_y;
                const double rt = (t - interval_start) / local_settings.mhd_dt;
                const auto c = sample_time_interpolated_coefficients(
                    lower_interpolator, upper_interpolator, q, rt,
                    local_settings.time_interpolation);

                const double vx = c[ParkerCoeffVx];
                const double vy = c[ParkerCoeffVy];
                const double bx = c[ParkerCoeffBx];
                const double by = c[ParkerCoeffBy];
                const double bz = c[ParkerCoeffBz];
                const double b = c[ParkerCoeffBmag];
                const double dvx_dx = c[ParkerCoeffDvxDx];
                const double dvy_dy = c[ParkerCoeffDvyDy];
                const double dbx_dx = c[ParkerCoeffDbxDx];
                const double dbx_dy = c[ParkerCoeffDbxDy];
                const double dby_dx = c[ParkerCoeffDbyDx];
                const double dby_dy = c[ParkerCoeffDbyDy];
                const double dbz_dx = c[ParkerCoeffDbzDx];
                const double dbz_dy = c[ParkerCoeffDbzDy];
                const double db_dx = c[ParkerCoeffDbmagDx];
                const double db_dy = c[ParkerCoeffDbmagDy];

                if (!Kokkos::isfinite(p) || p <= 0.0 || !Kokkos::isfinite(b) ||
                    b <= 0.0) {
                    status(index) = 2;
                    break;
                }

                const double ib = 1.0 / b;
                const double ib2 = ib * ib;
                const double ib3 = ib2 * ib;

                double knorm_para = 1.0;
                if (local_settings.magnetic_dependency == 1) {
                    knorm_para *=
                        Kokkos::pow(b, local_settings.gamma_turb - 2.0);
                }
                double knorm = knorm_para;
                if (local_settings.momentum_dependency == 1) {
                    knorm *= Kokkos::pow(p / local_settings.p0,
                                         local_settings.pindex);
                }
                const double kpara = local_settings.kpara0 * knorm;
                const double kperp = kpara * local_settings.kperp_over_kpara;
                if (!Kokkos::isfinite(kpara) || kpara <= 0.0 ||
                    !Kokkos::isfinite(kperp) || kperp < 0.0) {
                    status(index) = 2;
                    break;
                }

                const double skpara = Kokkos::sqrt(2.0 * kpara);
                const double skperp = Kokkos::sqrt(2.0 * kperp);
                const double skpara_perp =
                    Kokkos::sqrt(2.0 * (kpara - kperp));
                const double kpp = kpara - kperp;

                double dkdx = 0.0;
                double dkdy = 0.0;
                if (local_settings.magnetic_dependency == 1) {
                    const double magnetic_exponent =
                        local_settings.gamma_turb - 2.0;
                    dkdx = db_dx * ib * magnetic_exponent;
                    dkdy = db_dy * ib * magnetic_exponent;
                }

                const double dkxx_dx =
                    kperp * dkdx + kpp * dkdx * bx * bx * ib2 +
                    2.0 * kpp * bx * (dbx_dx * b - bx * db_dx) * ib3;
                const double dkyy_dy =
                    kperp * dkdy + kpp * dkdy * by * by * ib2 +
                    2.0 * kpp * by * (dby_dy * b - by * db_dy) * ib3;
                const double dkxy_dx =
                    kpp * dkdx * bx * by * ib2 +
                    kpp * ((dbx_dx * by + bx * dby_dx) * ib2 -
                           2.0 * bx * by * db_dx * ib3);
                const double dkxy_dy =
                    kpp * dkdy * bx * by * ib2 +
                    kpp * ((dbx_dy * by + bx * dby_dy) * ib2 -
                           2.0 * bx * by * db_dy * ib3);

                const double drift_denominator =
                    Kokkos::sqrt(square(local_settings.drift_param1 *
                                        local_settings.p0 / p) +
                                 square(local_settings.drift_param2 *
                                        local_settings.p0 *
                                        local_settings.p0 / (p * p)));
                const double vdp =
                    1.0 / (3.0 * local_settings.charge) / drift_denominator;
                const double vdx =
                    vdp * (dbz_dy * ib2 - 2.0 * bz * db_dy * ib3);
                const double vdy =
                    vdp * (-dbz_dx * ib2 + 2.0 * bz * db_dx * ib3);

                const double dx_dt = vx + vdx + dkxx_dx + dkxy_dy;
                const double dy_dt = vy + vdy + dkxy_dx + dkyy_dy;
                const double divv = dvx_dx + dvy_dy;
                const double dp_dt = -p * divv / 3.0;

                if (dx_dt != 0.0 && dy_dt != 0.0 && dp_dt != 0.0) {
                    const double diffusion_dt_x =
                        square(0.5 * local_settings.dx / skpara);
                    const double diffusion_dt_y =
                        square(0.5 * local_settings.dy / skpara);
                    const double advection_scale =
                        skperp > 0.0 ? skperp : skpara;
                    const double advection_dt_x =
                        square(advection_scale / dx_dt);
                    const double advection_dt_y =
                        square(advection_scale / dy_dt);
                    const double momentum_dt =
                        0.1 * p / Kokkos::abs(dp_dt);
                    dt = min_positive(diffusion_dt_x, diffusion_dt_y);
                    dt = min_positive(dt, advection_dt_x);
                    dt = min_positive(dt, advection_dt_y);
                    dt = min_positive(dt, momentum_dt);
                } else {
                    dt = local_settings.dt_min;
                }
                if (dt < local_settings.dt_min) {
                    dt = local_settings.dt_min;
                }
                if (dt > local_settings.dt_max) {
                    dt = local_settings.dt_max;
                }
                diagnostic_dt = dt;
                const double remaining = interval_end - t;
                if (dt > remaining) {
                    dt = remaining;
                }
                if (!Kokkos::isfinite(dt) || dt <= 0.0) {
                    status(index) = 2;
                    break;
                }

                const double sdt = Kokkos::sqrt(dt);
                const double ran1 = StochasticSampler<>::uniform_sqrt3(generator);
                const double ran2 = StochasticSampler<>::uniform_sqrt3(generator);
                const double ran3 = StochasticSampler<>::uniform_sqrt3(generator);
                const double delta_x =
                    dx_dt * dt + ran1 * skperp * sdt +
                    ran3 * skpara_perp * sdt * bx * ib;
                const double delta_y =
                    dy_dt * dt + ran2 * skperp * sdt +
                    ran3 * skpara_perp * sdt * by * ib;
                const double ignored_momentum_random =
                    StochasticSampler<>::uniform_sqrt3(generator);
                (void)ignored_momentum_random;
                const double delta_p = dp_dt * dt;

                x += delta_x;
                y += delta_y;
                p += delta_p;
                t += dt;
                if (p < 0.25 * local_settings.p0) {
                    p = 0.25 * local_settings.p0;
                }
                if (!Kokkos::isfinite(x) || !Kokkos::isfinite(y) ||
                    !Kokkos::isfinite(p) || !Kokkos::isfinite(t)) {
                    status(index) = 2;
                    break;
                }
            }

            if (step_guard >= local_settings.max_particle_steps_per_interval &&
                t < interval_end) {
                status(index) = 2;
            }

            position(index, 0) =
                wrap_periodic_coordinate(x, local_settings.x_lower,
                                         local_settings.x_upper,
                                         local_settings.x_width);
            position(index, 1) =
                wrap_periodic_coordinate(y, local_settings.y_lower,
                                         local_settings.y_upper,
                                         local_settings.y_width);
            momentum(index) = p;
            time(index) = t;
            step_dt(index) = diagnostic_dt;
            random_pool.free_state(generator);
        });
    Kokkos::fence("Reconnection2DCalibration::move_parker_particles_one_interval");
}

/**
 * Advance active particles across one MHD interval with the Fortran 2D transport scheme.
 */
template <typename ParticleStorageType, typename CoefficientFieldType,
          typename RandomManagerType>
void move_particles_one_interval(
    ParticleStorageType& particles,
    const CoefficientFieldType& lower_coefficients,
    const CoefficientFieldType& upper_coefficients,
    const RandomManagerType& random_manager,
    const Reconnection2DDeviceSettings& settings,
    const double interval_start) {
    using execution_space = typename ParticleStorageType::execution_space;
    using coordinate_array_type =
        typename CoefficientFieldType::coordinate_array_type;

    auto random_pool = random_manager.pool;
    auto position = particles.position;
    auto momentum = particles.momentum;
    auto time = particles.time;
    auto step_dt = particles.step_dt;
    auto mu = particles.mu;
    auto status = particles.status;
    const auto lower_interpolator =
        FieldInterpolator::make_linear_position_interpolator(
            lower_coefficients, GridDomain::GhostedDomain,
            FieldInterpolator::InterpolationBoundaryPolicy::WrapPeriodic);
    const auto upper_interpolator =
        FieldInterpolator::make_linear_position_interpolator(
            upper_coefficients, GridDomain::GhostedDomain,
            FieldInterpolator::InterpolationBoundaryPolicy::WrapPeriodic);
    const auto local_settings = settings;
    const double interval_end = interval_start + settings.mhd_dt;
    const auto count = particles.count;

    Kokkos::parallel_for(
        "Reconnection2DCalibration::move_particles_one_interval",
        Kokkos::RangePolicy<execution_space>(
            0, static_cast<typename ParticleStorageType::size_type>(count)),
        KOKKOS_LAMBDA(const typename ParticleStorageType::size_type index) {
            if (status(index) != 0) {
                return;
            }

            auto generator = random_pool.get_state();
            double x = position(index, 0);
            double y = position(index, 1);
            double p = momentum(index);
            double pitch_mu = clamp_pitch_angle_cosine(mu(index),
                                                       local_settings.mu_max);
            double t = time(index);
            double dt = step_dt(index);
            double diagnostic_dt = dt;
            int step_guard = 0;

            while (t < interval_end &&
                   step_guard < local_settings.max_particle_steps_per_interval &&
                   status(index) == 0) {
                ++step_guard;
                x = wrap_periodic_coordinate(x, local_settings.x_lower,
                                             local_settings.x_upper,
                                             local_settings.x_width);
                y = wrap_periodic_coordinate(y, local_settings.y_lower,
                                             local_settings.y_upper,
                                             local_settings.y_width);

                coordinate_array_type q{};
                q[0] = x + local_settings.interpolation_shift_x;
                q[1] = y + local_settings.interpolation_shift_y;
                const double rt = (t - interval_start) / local_settings.mhd_dt;
                const auto c = sample_time_interpolated_coefficients(
                    lower_interpolator, upper_interpolator, q, rt,
                    local_settings.time_interpolation);

                const double vx = c[CoeffVx];
                const double vy = c[CoeffVy];
                const double vz = c[CoeffVz];
                (void)vz;
                const double bx = c[CoeffBx];
                const double by = c[CoeffBy];
                const double bz = c[CoeffBz];
                const double b = c[CoeffBmag];
                const double dvx_dx = c[CoeffDvxDx];
                const double dvx_dy = c[CoeffDvxDy];
                const double dvy_dx = c[CoeffDvyDx];
                const double dvy_dy = c[CoeffDvyDy];
                const double dvz_dx = c[CoeffDvzDx];
                const double dvz_dy = c[CoeffDvzDy];
                const double dbx_dx = c[CoeffDbxDx];
                const double dbx_dy = c[CoeffDbxDy];
                const double dby_dx = c[CoeffDbyDx];
                const double dby_dy = c[CoeffDbyDy];
                const double dbz_dx = c[CoeffDbzDx];
                const double dbz_dy = c[CoeffDbzDy];
                const double db_dx = c[CoeffDbmagDx];
                const double db_dy = c[CoeffDbmagDy];

                if (!Kokkos::isfinite(p) || p <= 0.0 || !Kokkos::isfinite(b) ||
                    b <= 0.0) {
                    status(index) = 2;
                    break;
                }

                const double ib = 1.0 / b;
                const double ib2 = ib * ib;
                const double ib3 = ib2 * ib;

                double knorm_para = 1.0;
                if (local_settings.magnetic_dependency == 1) {
                    knorm_para *=
                        Kokkos::pow(b, local_settings.gamma_turb - 2.0);
                }
                double knorm = knorm_para;
                if (local_settings.momentum_dependency == 1) {
                    knorm *= Kokkos::pow(p / local_settings.p0,
                                         local_settings.pindex);
                }
                const double kpara = local_settings.kpara0 * knorm;
                const double kperp = kpara * local_settings.kperp_over_kpara;
                if (!Kokkos::isfinite(kpara) || kpara <= 0.0 ||
                    !Kokkos::isfinite(kperp) || kperp < 0.0) {
                    status(index) = 2;
                    break;
                }

                const double skpara = Kokkos::sqrt(2.0 * kpara);
                const double skperp = Kokkos::sqrt(2.0 * kperp);
                const double skpara_perp =
                    kpara > kperp ? Kokkos::sqrt(2.0 * (kpara - kperp)) : 0.0;
                const bool focused_transport =
                    local_settings.focused_transport != 0;
                const double kpp = focused_transport ? -kperp : kpara - kperp;

                double dkdx = 0.0;
                double dkdy = 0.0;
                if (local_settings.magnetic_dependency == 1) {
                    const double magnetic_exponent =
                        local_settings.gamma_turb - 2.0;
                    dkdx = db_dx * ib * magnetic_exponent;
                    dkdy = db_dy * ib * magnetic_exponent;
                }

                const double dkxx_dx =
                    kperp * dkdx + kpp * dkdx * bx * bx * ib2 +
                    2.0 * kpp * bx * (dbx_dx * b - bx * db_dx) * ib3;
                const double dkyy_dy =
                    kperp * dkdy + kpp * dkdy * by * by * ib2 +
                    2.0 * kpp * by * (dby_dy * b - by * db_dy) * ib3;
                const double dkxy_dx =
                    kpp * dkdx * bx * by * ib2 +
                    kpp * ((dbx_dx * by + bx * dby_dx) * ib2 -
                           2.0 * bx * by * db_dx * ib3);
                const double dkxy_dy =
                    kpp * dkdy * bx * by * ib2 +
                    kpp * ((dbx_dy * by + bx * dby_dy) * ib2 -
                           2.0 * bx * by * db_dy * ib3);

                const double drift_denominator =
                    Kokkos::sqrt(square(local_settings.drift_param1 *
                                        local_settings.p0 / p) +
                                 square(local_settings.drift_param2 *
                                        local_settings.p0 *
                                        local_settings.p0 / (p * p)));

                double dx_dt = 0.0;
                double dy_dt = 0.0;
                double dp_dt = 0.0;
                double dmu_dt = 0.0;
                double duu = 0.0;

                if (focused_transport) {
                    pitch_mu = clamp_pitch_angle_cosine(pitch_mu,
                                                        local_settings.mu_max);
                    const double mu2 = pitch_mu * pitch_mu;
                    const double muf1 = 0.5 * (1.0 - mu2);
                    const double muf2 = 0.5 * (3.0 * mu2 - 1.0);
                    const double kx = bx * dbx_dx + by * dbx_dy;
                    const double ky = bx * dby_dx + by * dby_dy;
                    const double kz = bx * dbz_dx + by * dbz_dy;
                    const double bdot_curvb =
                        bx * dbz_dy - by * dbz_dx + bz * (dby_dx - dbx_dy);
                    const double vdp =
                        1.0 / local_settings.charge / drift_denominator;
                    const double vdx =
                        vdp * (muf1 * (-bz * db_dy) * ib2 +
                               mu2 * (by * kz - bz * ky) * ib3 +
                               muf1 * bx * bdot_curvb * ib3);
                    const double vdy =
                        vdp * (muf1 * (bz * db_dx) * ib2 +
                               mu2 * (bz * kx - bx * kz) * ib3 +
                               muf1 * by * bdot_curvb * ib3);
                    const double speed =
                        local_settings.particle_v0 * p / local_settings.p0;
                    if (!Kokkos::isfinite(speed) || speed <= 0.0) {
                        status(index) = 2;
                        break;
                    }
                    const double v_parallel_over_b = speed * pitch_mu * ib;
                    const double vbx = v_parallel_over_b * bx;
                    const double vby = v_parallel_over_b * by;
                    dx_dt = vx + vdx + vbx + dkxx_dx + dkxy_dy;
                    dy_dt = vy + vdy + vby + dkxy_dx + dkyy_dy;

                    const double divv = dvx_dx + dvy_dy;
                    const double bb_gradv =
                        (bx * (bx * dvx_dx + by * dvx_dy) +
                         by * (bx * dvy_dx + by * dvy_dy) +
                         bz * (bx * dvz_dx + by * dvz_dy)) * ib2;
                    const double bv_gradv =
                        (bx * (vx * dvx_dx + vy * dvx_dy) +
                         by * (vx * dvy_dx + vy * dvy_dy) +
                         bz * (vx * dvz_dx + vy * dvz_dy)) * ib;
                    const double acc_rate =
                        -(muf1 * divv + muf2 * bb_gradv +
                          pitch_mu * bv_gradv / speed);
                    dp_dt = p * acc_rate;

                    const double div_bnorm = -(bx * db_dx + by * db_dy) * ib2;
                    dmu_dt =
                        speed * div_bnorm + pitch_mu * divv -
                        3.0 * pitch_mu * bb_gradv - 2.0 * bv_gradv / speed;
                    dmu_dt *= 0.5 * (1.0 - mu2);
                    const double abs_mu = Kokkos::abs(pitch_mu);
                    const double dtmp =
                        Kokkos::pow(abs_mu, local_settings.gamma_turb - 1.0) +
                        local_settings.pitch_angle_scattering_h0;
                    duu = local_settings.duu0 * (1.0 - mu2) * dtmp;
                    double duu_du = 0.0;
                    if (pitch_mu > 0.0) {
                        duu_du =
                            local_settings.duu0 *
                            (-2.0 * pitch_mu * dtmp +
                             (1.0 - mu2) *
                                 Kokkos::pow(abs_mu,
                                             local_settings.gamma_turb - 2.0));
                    } else if (pitch_mu < 0.0) {
                        duu_du =
                            local_settings.duu0 *
                            (-2.0 * pitch_mu * dtmp -
                             (1.0 - mu2) *
                                 Kokkos::pow(abs_mu,
                                             local_settings.gamma_turb - 2.0));
                    }
                    double duu_norm = 1.0;
                    if (local_settings.magnetic_dependency == 1) {
                        duu_norm *=
                            Kokkos::pow(b, 2.0 - local_settings.gamma_turb);
                    }
                    if (local_settings.momentum_dependency == 1) {
                        duu_norm *=
                            Kokkos::pow(p / local_settings.p0,
                                        local_settings.gamma_turb - 1.0);
                    }
                    duu *= duu_norm;
                    duu_du *= duu_norm;
                    dmu_dt += duu_du;
                    if (!Kokkos::isfinite(dmu_dt) ||
                        !Kokkos::isfinite(duu) || duu < 0.0) {
                        status(index) = 2;
                        break;
                    }
                } else {
                    const double vdp =
                        1.0 / (3.0 * local_settings.charge) / drift_denominator;
                    const double vdx =
                        vdp * (dbz_dy * ib2 - 2.0 * bz * db_dy * ib3);
                    const double vdy =
                        vdp * (-dbz_dx * ib2 + 2.0 * bz * db_dx * ib3);
                    dx_dt = vx + vdx + dkxx_dx + dkxy_dy;
                    dy_dt = vy + vdy + dkxy_dx + dkyy_dy;
                    const double divv = dvx_dx + dvy_dy;
                    dp_dt = -p * divv / 3.0;
                }

                if (focused_transport) {
                    const double diffusion_scale =
                        skperp > 0.0 ? skperp : skpara;
                    if (dx_dt != 0.0 && dy_dt != 0.0 && dp_dt != 0.0 &&
                        dmu_dt != 0.0 && diffusion_scale > 0.0) {
                        const double diffusion_dt_x =
                            square(0.5 * local_settings.dx / diffusion_scale);
                        const double diffusion_dt_y =
                            square(0.5 * local_settings.dy / diffusion_scale);
                        const double advection_dt_x =
                            square(diffusion_scale / dx_dt);
                        const double advection_dt_y =
                            square(diffusion_scale / dy_dt);
                        const double momentum_dt =
                            0.1 * p / Kokkos::abs(dp_dt);
                        const double pitch_advection_dt =
                            0.1 / Kokkos::abs(dmu_dt);
                        const double pitch_balance_dt =
                            2.0 * duu / square(dmu_dt);
                        dt = min_positive(diffusion_dt_x, diffusion_dt_y);
                        dt = min_positive(dt, advection_dt_x);
                        dt = min_positive(dt, advection_dt_y);
                        dt = min_positive(dt, momentum_dt);
                        dt = min_positive(dt, pitch_advection_dt);
                        dt = min_positive(dt, pitch_balance_dt);
                    } else {
                        dt = local_settings.dt_min;
                    }
                } else if (dx_dt != 0.0 && dy_dt != 0.0 && dp_dt != 0.0) {
                    const double diffusion_dt_x =
                        square(0.5 * local_settings.dx / skpara);
                    const double diffusion_dt_y =
                        square(0.5 * local_settings.dy / skpara);
                    const double advection_scale =
                        skperp > 0.0 ? skperp : skpara;
                    const double advection_dt_x =
                        square(advection_scale / dx_dt);
                    const double advection_dt_y =
                        square(advection_scale / dy_dt);
                    const double momentum_dt =
                        0.1 * p / Kokkos::abs(dp_dt);
                    dt = min_positive(diffusion_dt_x, diffusion_dt_y);
                    dt = min_positive(dt, advection_dt_x);
                    dt = min_positive(dt, advection_dt_y);
                    dt = min_positive(dt, momentum_dt);
                } else {
                    dt = local_settings.dt_min;
                }
                if (dt < local_settings.dt_min) {
                    dt = local_settings.dt_min;
                }
                if (dt > local_settings.dt_max) {
                    dt = local_settings.dt_max;
                }
                diagnostic_dt = dt;
                const double remaining = interval_end - t;
                if (dt > remaining) {
                    dt = remaining;
                }
                if (!Kokkos::isfinite(dt) || dt <= 0.0) {
                    status(index) = 2;
                    break;
                }

                const double sdt = Kokkos::sqrt(dt);
                double delta_x = 0.0;
                double delta_y = 0.0;
                double delta_p = dp_dt * dt;
                if (focused_transport) {
                    const double ran1 =
                        StochasticSampler<>::uniform_sqrt3(generator);
                    const double ran2 =
                        StochasticSampler<>::uniform_sqrt3(generator);
                    const double bxn = bx * ib;
                    const double byn = by * ib;
                    const double bzn = bz * ib;
                    const double bxy_norm =
                        Kokkos::sqrt(bxn * bxn + byn * byn);
                    double stochastic_x = 0.0;
                    double stochastic_y = 0.0;
                    if (bxy_norm > 0.0) {
                        const double ibxy_norm = 1.0 / bxy_norm;
                        stochastic_x =
                            skperp * ibxy_norm * sdt *
                            (-bxn * bzn * ran1 - by * ran2);
                        stochastic_y =
                            skperp * ibxy_norm * sdt *
                            (-byn * bzn * ran1 + bx * ran2);
                    }
                    delta_x = dx_dt * dt + stochastic_x;
                    delta_y = dy_dt * dt + stochastic_y;
                    const double ignored_momentum_random =
                        StochasticSampler<>::uniform_sqrt3(generator);
                    (void)ignored_momentum_random;
                    const double pitch_random =
                        StochasticSampler<>::uniform_sqrt3(generator);
                    pitch_mu = clamp_pitch_angle_cosine(
                        pitch_mu + dmu_dt * dt +
                            pitch_random * Kokkos::sqrt(2.0 * duu) * sdt,
                        local_settings.mu_max);
                } else {
                    const double ran1 =
                        StochasticSampler<>::uniform_sqrt3(generator);
                    const double ran2 =
                        StochasticSampler<>::uniform_sqrt3(generator);
                    const double ran3 =
                        StochasticSampler<>::uniform_sqrt3(generator);
                    delta_x =
                        dx_dt * dt + ran1 * skperp * sdt +
                        ran3 * skpara_perp * sdt * bx * ib;
                    delta_y =
                        dy_dt * dt + ran2 * skperp * sdt +
                        ran3 * skpara_perp * sdt * by * ib;
                    const double ignored_momentum_random =
                        StochasticSampler<>::uniform_sqrt3(generator);
                    (void)ignored_momentum_random;
                }

                x += delta_x;
                y += delta_y;
                p += delta_p;
                t += dt;
                if (p < 0.25 * local_settings.p0) {
                    p = 0.25 * local_settings.p0;
                }
                if (!Kokkos::isfinite(x) || !Kokkos::isfinite(y) ||
                    !Kokkos::isfinite(p) || !Kokkos::isfinite(t)) {
                    status(index) = 2;
                    break;
                }
            }

            if (step_guard >= local_settings.max_particle_steps_per_interval &&
                t < interval_end) {
                status(index) = 2;
            }

            position(index, 0) =
                wrap_periodic_coordinate(x, local_settings.x_lower,
                                         local_settings.x_upper,
                                         local_settings.x_width);
            position(index, 1) =
                wrap_periodic_coordinate(y, local_settings.y_lower,
                                         local_settings.y_upper,
                                         local_settings.y_width);
            momentum(index) = p;
            mu(index) = pitch_mu;
            time(index) = t;
            step_dt(index) = diagnostic_dt;
            random_pool.free_state(generator);
        });
    Kokkos::fence("Reconnection2DCalibration::move_particles_one_interval");
}

/**
 * Split high-momentum particles with the same threshold sequence as the Fortran code.
 */
template <typename ParticleStorageType>
std::pair<std::uint64_t, std::uint64_t> split_particles(
    ParticleStorageType& particles,
    const Reconnection2DDeviceSettings& settings) {
    if (particles.count >= particles.capacity || settings.split_particles == 0) {
        return {0, 0};
    }

    using execution_space = typename ParticleStorageType::execution_space;
    using memory_space = typename ParticleStorageType::memory_space;
    using layout_type = typename ParticleStorageType::layout_type;
    using counter_view_type = Kokkos::View<std::uint64_t*, layout_type, memory_space>;
    counter_view_type counters("reconnection_split_counters", 3);
    auto counters_host = Kokkos::create_mirror_view(counters);
    counters_host(0) = particles.count;
    counters_host(1) = 0;
    counters_host(2) = 0;
    Kokkos::deep_copy(counters, counters_host);

    auto position = particles.position;
    auto momentum = particles.momentum;
    auto time = particles.time;
    auto step_dt = particles.step_dt;
    auto weight = particles.weight;
    auto mu = particles.mu;
    auto status = particles.status;
    auto split_level = particles.split_level;
    const auto old_count = particles.count;
    const auto capacity = particles.capacity;
    const auto local_settings = settings;

    Kokkos::parallel_for(
        "Reconnection2DCalibration::split_particles",
        Kokkos::RangePolicy<execution_space>(
            0, static_cast<typename ParticleStorageType::size_type>(old_count)),
        KOKKOS_LAMBDA(const typename ParticleStorageType::size_type parent_index) {
            if (status(parent_index) != 0) {
                return;
            }
            const int old_split_level = split_level(parent_index);
            const double threshold =
                local_settings.pmin_split *
                Kokkos::pow(local_settings.split_ratio,
                            static_cast<double>(old_split_level));
            const double p = momentum(parent_index);
            if (!(p > threshold && p <= local_settings.pmax)) {
                return;
            }

            const std::uint64_t child_index =
                Kokkos::atomic_fetch_add(&counters(0), static_cast<std::uint64_t>(1));
            if (child_index >= capacity) {
                Kokkos::atomic_add(&counters(2), static_cast<std::uint64_t>(1));
                return;
            }

            const int new_split_level = old_split_level + 1;
            const double new_weight =
                Kokkos::pow(0.5, static_cast<double>(new_split_level));
            for (int dim = 0; dim < 2; ++dim) {
                position(child_index, dim) = position(parent_index, dim);
            }
            momentum(child_index) = momentum(parent_index);
            time(child_index) = time(parent_index);
            step_dt(child_index) = step_dt(parent_index);
            weight(parent_index) = new_weight;
            weight(child_index) = new_weight;
            mu(child_index) = mu(parent_index);
            status(child_index) = 0;
            split_level(parent_index) = new_split_level;
            split_level(child_index) = new_split_level;
            Kokkos::atomic_add(&counters(1), static_cast<std::uint64_t>(1));
        });
    Kokkos::fence("Reconnection2DCalibration::split_particles");

    counters_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, counters);
    particles.count = std::min(counters_host(0), particles.capacity);
    return {counters_host(1), counters_host(2)};
}

/**
 * Build host-side particle statistics from device arrays.
 */
template <typename ParticleStorageType>
ParticleSummary summarize_particles(const ParticleStorageType& particles) {
    ParticleSummary summary;
    summary.stored = particles.count;
    if (particles.count == 0) {
        return summary;
    }

    const auto momentum_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                            particles.momentum);
    const auto step_dt_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                            particles.step_dt);
    const auto weight_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                            particles.weight);
    const auto status_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                            particles.status);
    const auto split_level_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                            particles.split_level);

    double p_sum = 0.0;
    double dt_sum = 0.0;
    summary.p_min = std::numeric_limits<double>::infinity();
    summary.dt_min = std::numeric_limits<double>::infinity();
    summary.p_max = 0.0;
    summary.dt_max = 0.0;

    for (std::uint64_t index = 0; index < particles.count; ++index) {
        if (status_host(index) == 0) {
            ++summary.active;
            const double p = momentum_host(index);
            const double dt = step_dt_host(index);
            summary.weight_sum += weight_host(index);
            p_sum += p;
            dt_sum += dt;
            summary.p_min = std::min(summary.p_min, p);
            summary.p_max = std::max(summary.p_max, p);
            summary.dt_min = std::min(summary.dt_min, dt);
            summary.dt_max = std::max(summary.dt_max, dt);
            summary.max_split_level =
                std::max(summary.max_split_level, split_level_host(index));
        } else {
            ++summary.inactive;
        }
    }
    if (summary.active > 0) {
        summary.p_avg = p_sum / static_cast<double>(summary.active);
        summary.dt_avg = dt_sum / static_cast<double>(summary.active);
    } else {
        summary.p_min = 0.0;
        summary.dt_min = 0.0;
    }
    return summary;
}

/**
 * Write one momentum histogram for comparison with Fortran distribution diagnostics.
 */
template <typename ParticleStorageType>
void write_momentum_histogram(const ParticleStorageType& particles,
                              const Reconnection2DSettings& settings,
                              const int frame) {
    constexpr int bin_count = 128;
    std::vector<double> histogram(bin_count, 0.0);
    if (particles.count > 0) {
        const auto momentum_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                particles.momentum);
        const auto weight_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                particles.weight);
        const auto status_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                particles.status);
        const double log_pmin = std::log(settings.pmin);
        const double inv_dlogp =
            static_cast<double>(bin_count) / (std::log(settings.pmax) - log_pmin);
        for (std::uint64_t index = 0; index < particles.count; ++index) {
            if (status_host(index) != 0) {
                continue;
            }
            const double p = momentum_host(index);
            if (p < settings.pmin || p > settings.pmax) {
                continue;
            }
            int bin = static_cast<int>((std::log(p) - log_pmin) * inv_dlogp);
            if (bin < 0) {
                bin = 0;
            }
            if (bin >= bin_count) {
                bin = bin_count - 1;
            }
            histogram[static_cast<std::size_t>(bin)] += weight_host(index);
        }
    }

    std::ostringstream name;
    name << settings.output_dir << "/momentum_histogram_" << std::setw(5)
         << std::setfill('0') << frame << ".csv";
    std::ofstream stream(name.str());
    if (!stream) {
        throw std::runtime_error("Failed to open momentum histogram output.");
    }
    stream << "bin,p_left,p_right,weight\n";
    const double dlogp =
        (std::log(settings.pmax) - std::log(settings.pmin)) /
        static_cast<double>(bin_count);
    for (int bin = 0; bin < bin_count; ++bin) {
        const double p_left = std::exp(std::log(settings.pmin) +
                                       static_cast<double>(bin) * dlogp);
        const double p_right = std::exp(std::log(settings.pmin) +
                                        static_cast<double>(bin + 1) * dlogp);
        stream << bin << ',' << std::setprecision(16) << p_left << ','
               << p_right << ',' << histogram[static_cast<std::size_t>(bin)]
               << '\n';
    }
}

/**
 * Write one trivially copyable scalar to a binary particle snapshot stream.
 */
template <typename T>
void write_particle_binary_scalar(std::ofstream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!stream) {
        throw std::runtime_error("Failed to write particle binary scalar.");
    }
}

/**
 * Read one trivially copyable scalar from a binary particle snapshot stream.
 */
template <typename T>
T read_particle_binary_scalar(std::ifstream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream) {
        throw std::runtime_error("Failed to read particle binary scalar.");
    }
    return value;
}

/**
 * Return whether two finite scalar metadata values are equal within roundoff.
 */
bool restart_metadata_matches(const double actual, const double expected) {
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
        return false;
    }
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    return std::abs(actual - expected) <= 1.0e-12 * scale;
}

/**
 * Load a version-4 repository particle snapshot into reconnection particle storage.
 *
 * The repository particle snapshot format does not carry the reconnection pusher's
 * per-particle substep time or the Kokkos random-pool state. Restart therefore resumes
 * from a completed MHD frame boundary: all loaded particle times are reset to
 * restart_frame * dt_out and the next interval receives a fresh RNG sequence.
 */
template <typename ParticleStorageType>
void load_reconnection_particle_snapshot(ParticleStorageType& particles,
                                         const Reconnection2DSettings& settings,
                                         const int restart_frame) {
    std::ifstream stream(settings.restart_particle_snapshot_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open restart particle snapshot.");
    }

    constexpr std::array<char, 8> expected_magic{
        'K', 'P', 'T', 'P', 'R', 'T', '\0', '\0'
    };
    std::array<char, 8> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!stream || magic != expected_magic) {
        throw std::runtime_error(
            "Restart particle snapshot is not a repository KPTPRT file.");
    }

    const auto version = read_particle_binary_scalar<std::uint32_t>(stream);
    const auto endian_marker = read_particle_binary_scalar<std::uint32_t>(stream);
    if (version != 4u) {
        throw std::runtime_error(
            "Restart particle snapshot must use particle binary version 4.");
    }
    if (endian_marker != 0x01020304u) {
        throw std::runtime_error(
            "Restart particle snapshot endian does not match this host.");
    }

    const auto space_dim = read_particle_binary_scalar<std::int32_t>(stream);
    const auto species = read_particle_binary_scalar<std::int32_t>(stream);
    const auto particle_count = read_particle_binary_scalar<std::uint64_t>(stream);
    const auto snapshot_capacity = read_particle_binary_scalar<std::uint64_t>(stream);
    const double rest_mass = read_particle_binary_scalar<double>(stream);
    const double charge = read_particle_binary_scalar<double>(stream);
    const double speed_of_light = read_particle_binary_scalar<double>(stream);
    const double energy_scale_erg = read_particle_binary_scalar<double>(stream);
    const double split_ratio = read_particle_binary_scalar<double>(stream);
    const double minimum_child_weight = read_particle_binary_scalar<double>(stream);
    (void)species;
    (void)rest_mass;
    (void)speed_of_light;
    (void)energy_scale_erg;
    (void)minimum_child_weight;

    if (space_dim != reconnection_particle_space_dim) {
        throw std::runtime_error(
            "Restart particle snapshot dimensionality does not match reconnection storage.");
    }
    if (particle_count > particles.capacity) {
        std::ostringstream message;
        message << "Restart particle snapshot stores " << particle_count
                << " particles, but configured capacity is " << particles.capacity
                << ". Snapshot capacity was " << snapshot_capacity << '.';
        throw std::runtime_error(message.str());
    }
    if (!restart_metadata_matches(charge, settings.charge)) {
        throw std::runtime_error(
            "Restart particle snapshot charge does not match current settings.");
    }
    if (!restart_metadata_matches(split_ratio, settings.split_ratio)) {
        throw std::runtime_error(
            "Restart particle snapshot split ratio does not match current settings.");
    }

    auto position_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.position);
    auto momentum_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.momentum);
    auto time_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.time);
    auto step_dt_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.step_dt);
    auto weight_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.weight);
    auto mu_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.mu);
    auto status_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.status);
    auto split_level_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                            particles.split_level);

    const double restart_time =
        static_cast<double>(restart_frame) * settings.mhd_dt;
    for (std::uint64_t index = 0; index < particle_count; ++index) {
        (void)read_particle_binary_scalar<std::uint64_t>(stream);
        status_host(index) = read_particle_binary_scalar<std::int32_t>(stream);
        split_level_host(index) = read_particle_binary_scalar<std::int32_t>(stream);
        (void)read_particle_binary_scalar<std::int32_t>(stream);
        for (int dim = 0; dim < reconnection_particle_space_dim; ++dim) {
            position_host(index, dim) = read_particle_binary_scalar<double>(stream);
        }
        for (int dim = 0; dim < reconnection_particle_space_dim; ++dim) {
            (void)read_particle_binary_scalar<double>(stream);
        }
        for (int dim = 0; dim < reconnection_particle_space_dim; ++dim) {
            (void)read_particle_binary_scalar<double>(stream);
        }
        momentum_host(index) = read_particle_binary_scalar<double>(stream);
        mu_host(index) = read_particle_binary_scalar<double>(stream);
        weight_host(index) = read_particle_binary_scalar<double>(stream);
        (void)read_particle_binary_scalar<double>(stream);
        time_host(index) = restart_time;
        step_dt_host(index) = settings.dt_min_rel * settings.mhd_dt;
    }

    Kokkos::deep_copy(particles.position, position_host);
    Kokkos::deep_copy(particles.momentum, momentum_host);
    Kokkos::deep_copy(particles.time, time_host);
    Kokkos::deep_copy(particles.step_dt, step_dt_host);
    Kokkos::deep_copy(particles.weight, weight_host);
    Kokkos::deep_copy(particles.mu, mu_host);
    Kokkos::deep_copy(particles.status, status_host);
    Kokkos::deep_copy(particles.split_level, split_level_host);
    particles.count = particle_count;
}

/**
 * Write one repository-format particle binary snapshot from the reconnection storage.
 */
template <typename ParticleStorageType>
void write_reconnection_particle_snapshot(const ParticleStorageType& particles,
                                          const Reconnection2DSettings& settings,
                                          const int frame) {
    const std::string path = particle_snapshot_file_path(settings.output_dir, frame);
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open particle snapshot output.");
    }

    const auto position_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.position);
    const auto momentum_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.momentum);
    const auto mu_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.mu);
    const auto weight_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.weight);
    const auto status_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.status);
    const auto split_level_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                            particles.split_level);

    constexpr char magic[8] = {'K', 'P', 'T', 'P', 'R', 'T', '\0', '\0'};
    constexpr std::uint32_t version = 4u;
    constexpr std::uint32_t endian_marker = 0x01020304u;
    constexpr double rest_mass = 1.0;
    constexpr double speed_of_light = 1.0;
    constexpr double energy_scale_erg = 1.0;
    constexpr double minimum_child_weight = 0.0;
    const double initial_kinetic_energy =
        std::sqrt(1.0 + settings.p0 * settings.p0) - 1.0;

    stream.write(magic, static_cast<std::streamsize>(sizeof(magic)));
    if (!stream) {
        throw std::runtime_error("Failed to write particle binary magic.");
    }
    write_particle_binary_scalar(stream, version);
    write_particle_binary_scalar(stream, endian_marker);
    write_particle_binary_scalar(stream,
                                 static_cast<std::int32_t>(reconnection_particle_space_dim));
    write_particle_binary_scalar(stream, reconnection_particle_custom_species);
    write_particle_binary_scalar(stream, static_cast<std::uint64_t>(particles.count));
    write_particle_binary_scalar(stream, static_cast<std::uint64_t>(particles.capacity));
    write_particle_binary_scalar(stream, rest_mass);
    write_particle_binary_scalar(stream, settings.charge);
    write_particle_binary_scalar(stream, speed_of_light);
    write_particle_binary_scalar(stream, energy_scale_erg);
    write_particle_binary_scalar(stream, settings.split_ratio);
    write_particle_binary_scalar(stream, minimum_child_weight);

    for (std::uint64_t index = 0; index < particles.count; ++index) {
        write_particle_binary_scalar(stream, index);
        write_particle_binary_scalar(stream, status_host(index));
        write_particle_binary_scalar(stream, split_level_host(index));
        write_particle_binary_scalar(stream, static_cast<std::int32_t>(0));
        for (int dim = 0; dim < reconnection_particle_space_dim; ++dim) {
            write_particle_binary_scalar(stream, position_host(index, dim));
        }
        for (int dim = 0; dim < reconnection_particle_space_dim; ++dim) {
            write_particle_binary_scalar(stream, position_host(index, dim));
        }
        for (int dim = 0; dim < reconnection_particle_space_dim; ++dim) {
            write_particle_binary_scalar(stream, position_host(index, dim));
        }
        write_particle_binary_scalar(stream, momentum_host(index));
        write_particle_binary_scalar(stream, mu_host(index));
        write_particle_binary_scalar(stream, weight_host(index));
        write_particle_binary_scalar(stream, initial_kinetic_energy);
    }
}

/**
 * Write the summary CSV header.
 */
void write_summary_header(std::ofstream& stream) {
    stream << "frame,time,stored,active,inactive,weight_sum,p_min,p_max,p_avg,"
              "dt_min,dt_max,dt_avg,max_split_level,injected,split,"
              "skipped_split_capacity,read_seconds,inject_seconds,move_seconds,"
              "split_seconds,total_frame_seconds,total_elapsed_seconds\n";
}

/**
 * Write one summary CSV row.
 */
void write_summary_row(std::ofstream& stream,
                       const int frame,
                       const double time,
                       const ParticleSummary& summary,
                       const FrameTiming& timing,
                       const double total_frame_seconds,
                       const double total_elapsed_seconds) {
    stream << frame << ',' << std::setprecision(16) << time << ','
           << summary.stored << ',' << summary.active << ','
           << summary.inactive << ',' << summary.weight_sum << ','
           << summary.p_min << ',' << summary.p_max << ','
           << summary.p_avg << ',' << summary.dt_min << ','
           << summary.dt_max << ',' << summary.dt_avg << ','
           << summary.max_split_level << ',' << timing.injected << ','
           << timing.split << ',' << timing.skipped_split_capacity << ','
           << timing.read_seconds << ',' << timing.inject_seconds << ','
           << timing.move_seconds << ',' << timing.split_seconds << ','
           << total_frame_seconds << ',' << total_elapsed_seconds << '\n';
}

/**
 * Read a coefficient frame that matches an already-selected transport coefficient type.
 */
template <typename CoefficientFieldType>
auto load_matching_coefficient_frame(const std::string& field_dir, const int frame) {
    if constexpr (CoefficientFieldType::component_dim ==
                  parker_coefficient_component_count) {
        return load_parker_coefficient_frame<>(field_dir, frame);
    } else {
        return load_coefficient_frame<>(field_dir, frame);
    }
}

/**
 * Execute the frame loop once the transport-specific coefficient type is known.
 */
template <typename ParticleStorageType, typename RandomManagerType,
          typename CoefficientFieldType>
int run_transport_loop(const Reconnection2DSettings& settings,
                       std::ofstream& summary_stream,
                       std::ofstream& log_stream,
                       RandomManagerType& random_manager,
                       ParticleStorageType& particles,
                       CoefficientFieldType lower_coefficients,
                       CoefficientFieldType upper_coefficients,
                       const double initial_read_seconds) {
    auto device_settings = make_device_settings(settings, lower_coefficients.grid);

    std::ostringstream header;
    header << "Reconnection 2D calibration\n"
           << "  profile=" << settings.profile_name << '\n'
           << "  transport=" << transport_model_name(settings.transport_model) << '\n'
           << "  field_dir=" << settings.field_dir << '\n'
           << "  output_dir=" << settings.output_dir << '\n'
           << "  restart_particle_snapshot="
           << (settings.restart_enabled()
                   ? settings.restart_particle_snapshot_path
                   : std::string("disabled")) << '\n'
           << "  overwrite_output="
           << (settings.overwrite_output ? "enabled" : "disabled") << '\n'
           << "  frames=" << settings.start_frame << ".."
           << settings.end_frame << " (" << settings.interval_count()
           << " intervals)\n"
           << "  particles_per_frame="
           << settings.particles_per_frame * settings.rank_scale
           << " (" << settings.particles_per_frame << " * rank_scale "
           << settings.rank_scale << ")\n"
           << "  capacity=" << settings.particle_capacity << '\n'
           << "  dt_out=" << settings.mhd_dt << '\n'
           << "  histogram_interval=" << settings.histogram_interval << '\n'
           << "  particle_snapshot_interval="
           << settings.particle_snapshot_interval << '\n'
           << "  walltime_limit_s=";
    if (settings.walltime_limit_enabled()) {
        header << settings.walltime_limit_seconds
               << " walltime_reserve_s=" << settings.walltime_reserve_seconds
               << " walltime_stop_s=" << settings.walltime_stop_seconds() << '\n';
    } else {
        header << "disabled\n";
    }
    header << "  kpara0=" << settings.kpara0
           << " kperp/kpara=" << settings.kperp_over_kpara << '\n'
           << "  particle_v0=" << settings.particle_v0
           << " duu0=" << settings.duu0 << '\n';
    write_log_block(log_stream, header.str());

    const auto run_start = wall_time_now();
    bool stopped_by_walltime = false;
    int last_completed_frame = settings.start_frame;
    for (int frame = settings.start_frame; frame < settings.end_frame; ++frame) {
        const auto frame_start = wall_time_now();
        FrameTiming timing;
        timing.read_seconds = frame == settings.start_frame ? initial_read_seconds : 0.0;
        if (frame > settings.start_frame) {
            lower_coefficients = upper_coefficients;
            const auto read_frame_start = wall_time_now();
            upper_coefficients =
                load_matching_coefficient_frame<CoefficientFieldType>(
                    settings.field_dir, frame + 1);
            timing.read_seconds = elapsed_seconds_since(read_frame_start);
        }

        const double interval_start =
            static_cast<double>(frame) * settings.mhd_dt;
        const std::uint64_t requested_particles =
            settings.particles_per_frame * settings.rank_scale;

        auto inject_start = wall_time_now();
        timing.injected = inject_uniform_particles(
            particles, random_manager, device_settings, requested_particles,
            interval_start);
        timing.inject_seconds = elapsed_seconds_since(inject_start);

        auto move_start = wall_time_now();
        if constexpr (CoefficientFieldType::component_dim ==
                      parker_coefficient_component_count) {
            move_parker_particles_one_interval(
                particles, lower_coefficients, upper_coefficients,
                random_manager, device_settings, interval_start);
        } else {
            move_particles_one_interval(particles, lower_coefficients,
                                        upper_coefficients, random_manager,
                                        device_settings, interval_start);
        }
        timing.move_seconds = elapsed_seconds_since(move_start);

        auto split_start = wall_time_now();
        const auto split_counts = split_particles(particles, device_settings);
        timing.split = split_counts.first;
        timing.skipped_split_capacity = split_counts.second;
        timing.split_seconds = elapsed_seconds_since(split_start);

        const double total_frame_seconds = elapsed_seconds_since(frame_start);
        const double total_elapsed_seconds = elapsed_seconds_since(run_start);
        const bool reached_walltime_limit =
            settings.walltime_limit_enabled() &&
            total_elapsed_seconds >= settings.walltime_stop_seconds();
        const bool final_frame = frame + 1 == settings.end_frame;
        const bool run_is_finishing = final_frame || reached_walltime_limit;

        const bool should_diagnose =
            ((frame + 1 - settings.start_frame) % settings.diagnostic_interval) == 0 ||
            run_is_finishing;
        if (should_diagnose) {
            const ParticleSummary summary = summarize_particles(particles);
            write_summary_row(summary_stream, frame + 1,
                              static_cast<double>(frame + 1) * settings.mhd_dt,
                              summary, timing, total_frame_seconds,
                              total_elapsed_seconds);
            summary_stream.flush();
            std::ostringstream frame_log;
            frame_log << "frame " << frame + 1
                      << " active=" << summary.active
                      << " stored=" << summary.stored
                      << " pmax=" << std::setprecision(6) << summary.p_max
                      << " read_s=" << timing.read_seconds
                      << " inject_s=" << timing.inject_seconds
                      << " move_s=" << timing.move_seconds
                      << " split_s=" << timing.split_seconds
                      << " frame_s=" << total_frame_seconds
                      << " elapsed_s=" << total_elapsed_seconds;
            write_log_line(log_stream, frame_log.str());
        }

        const bool should_write_histogram =
            (settings.histogram_interval > 0 &&
             ((frame + 1 - settings.start_frame) % settings.histogram_interval) == 0) ||
            run_is_finishing;
        if (should_write_histogram) {
            write_momentum_histogram(particles, settings, frame + 1);
        }

        const bool should_write_particle_snapshot =
            settings.particle_snapshot_interval > 0 &&
            (((frame + 1 - settings.start_frame) %
              settings.particle_snapshot_interval) == 0 ||
             run_is_finishing);
        if (should_write_particle_snapshot) {
            const auto snapshot_start = wall_time_now();
            write_reconnection_particle_snapshot(particles, settings, frame + 1);
            std::ostringstream snapshot_log;
            snapshot_log << "wrote particle snapshot "
                         << particle_snapshot_file_path(settings.output_dir, frame + 1)
                         << " in " << elapsed_seconds_since(snapshot_start) << " s";
            write_log_line(log_stream, snapshot_log.str());
        }

        last_completed_frame = frame + 1;
        if (reached_walltime_limit) {
            stopped_by_walltime = true;
            std::ostringstream stop_log;
            stop_log << "Reached walltime stop after frame " << last_completed_frame
                     << " elapsed_s=" << total_elapsed_seconds
                     << " stop_s=" << settings.walltime_stop_seconds()
                     << " limit_s=" << settings.walltime_limit_seconds
                     << " reserve_s=" << settings.walltime_reserve_seconds;
            write_log_line(log_stream, stop_log.str());
            break;
        }
    }

    std::ostringstream footer;
    if (stopped_by_walltime) {
        footer << "Stopped by walltime after frame " << last_completed_frame
               << " in " << elapsed_seconds_since(run_start)
               << " s. Summary: " << settings.output_dir << "/summary.csv\n";
    } else {
        footer << "Finished in " << elapsed_seconds_since(run_start)
               << " s. Summary: " << settings.output_dir << "/summary.csv\n";
    }
    write_log_block(log_stream, footer.str());
    return 0;
}

/**
 * Parse a required value after an option.
 */
std::string require_option_value(const int argc,
                                 char** argv,
                                 int& index,
                                 const std::string& option) {
    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for " + option + ".");
    }
    ++index;
    return argv[index];
}

/**
 * Parse command-line controls after Kokkos has consumed its own options.
 */
Reconnection2DSettings parse_settings(const int argc, char** argv) {
    Reconnection2DSettings settings;
    std::string profile_name = settings.profile_name;
    bool start_frame_was_set = false;
    int requested_frame_count = -1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--profile") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --profile.");
            }
            profile_name = argv[++i];
        }
    }

    apply_run_profile(settings, profile_name);

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: kokkos_particle_transport_app [options]\n"
                << "  --profile NAME                particle-64000(default), smoke, fortran-rank,\n"
                << "                                fortran-global, or full\n"
                << "  --transport MODEL             parker(default) or focused\n"
                << "  --focused-transport           shorthand for --transport focused\n"
                << "  --parker-transport            shorthand for --transport parker\n"
                << "  --restart-particle-snapshot PATH\n"
                << "                                load particles_XXXXX.bin and continue from frame XXXXX\n"
                << "  --overwrite-output           allow a fresh non-restart run to replace existing outputs\n"
                << "  --field-dir PATH              override profile field directory\n"
                << "  --output-dir PATH             override profile output directory\n"
                << "  --start-frame N               default 0\n"
                << "  --end-frame N                 reads through this frame\n"
                << "  --frames N                    sets end-frame = start-frame + N\n"
                << "  --particles-per-frame N       override profile particle injection count\n"
                << "  --rank-scale N                multiply particles per frame\n"
                << "  --capacity N                  override profile particle capacity\n"
                << "  --seed N                      default 114514\n"
                << "  --dt-out X                    default 0.1\n"
                << "  --diagnostic-interval N       profile default is usually 1\n"
                << "  --histogram-interval N        0 writes final histogram only\n"
                << "  --particle-snapshot-interval N\n"
                << "                                0 disables particle binary snapshots\n"
                << "  --max-steps-per-interval N    default 200000\n"
                << "  --walltime-hours X            stop after X hours, disabled at 0\n"
                << "  --walltime-seconds X          stop after X seconds, disabled at 0\n"
                << "  --walltime-reserve-minutes X  subtract X minutes from the stop limit\n"
                << "  --walltime-reserve-seconds X  subtract X seconds from the stop limit\n"
                << "  --particle-v0 X               focused initial speed at p0\n"
                << "  --duu0 X                      focused D_mumu normalization\n"
                << "  --split-ratio X               split threshold ratio between split levels\n"
                << "  --pmin-split-over-p0 X        first split threshold divided by p0\n"
                << "  --no-split                    disable Fortran-style particle splitting\n"
                << "  --no-time-interp              use the lower MHD frame only\n";
            std::exit(0);
        } else if (arg == "--profile") {
            (void)require_option_value(argc, argv, i, arg);
        } else if (arg == "--transport") {
            settings.transport_model =
                parse_transport_model_name(require_option_value(argc, argv, i, arg));
        } else if (arg == "--focused-transport") {
            settings.transport_model = TransportModel::Focused;
        } else if (arg == "--parker-transport") {
            settings.transport_model = TransportModel::Parker;
        } else if (arg == "--restart-particle-snapshot") {
            settings.restart_particle_snapshot_path =
                require_option_value(argc, argv, i, arg);
        } else if (arg == "--overwrite-output") {
            settings.overwrite_output = true;
        } else if (arg == "--field-dir") {
            settings.field_dir = require_option_value(argc, argv, i, arg);
        } else if (arg == "--output-dir") {
            settings.output_dir = require_option_value(argc, argv, i, arg);
        } else if (arg == "--start-frame") {
            settings.start_frame = std::stoi(require_option_value(argc, argv, i, arg));
            start_frame_was_set = true;
            if (requested_frame_count > 0) {
                settings.end_frame = settings.start_frame + requested_frame_count;
            }
        } else if (arg == "--end-frame") {
            settings.end_frame = std::stoi(require_option_value(argc, argv, i, arg));
        } else if (arg == "--frames") {
            requested_frame_count =
                std::stoi(require_option_value(argc, argv, i, arg));
            settings.end_frame = settings.start_frame + requested_frame_count;
        } else if (arg == "--particles-per-frame") {
            settings.particles_per_frame =
                std::stoull(require_option_value(argc, argv, i, arg));
        } else if (arg == "--rank-scale") {
            settings.rank_scale =
                std::stoull(require_option_value(argc, argv, i, arg));
        } else if (arg == "--capacity") {
            settings.particle_capacity =
                std::stoull(require_option_value(argc, argv, i, arg));
        } else if (arg == "--seed") {
            settings.seed = std::stoull(require_option_value(argc, argv, i, arg));
        } else if (arg == "--dt-out") {
            settings.mhd_dt = std::stod(require_option_value(argc, argv, i, arg));
        } else if (arg == "--diagnostic-interval") {
            settings.diagnostic_interval =
                std::stoi(require_option_value(argc, argv, i, arg));
        } else if (arg == "--histogram-interval") {
            settings.histogram_interval =
                std::stoi(require_option_value(argc, argv, i, arg));
        } else if (arg == "--particle-snapshot-interval") {
            settings.particle_snapshot_interval =
                std::stoi(require_option_value(argc, argv, i, arg));
        } else if (arg == "--max-steps-per-interval") {
            settings.max_particle_steps_per_interval =
                std::stoi(require_option_value(argc, argv, i, arg));
        } else if (arg == "--walltime-hours") {
            settings.walltime_limit_seconds =
                3600.0 * std::stod(require_option_value(argc, argv, i, arg));
        } else if (arg == "--walltime-seconds") {
            settings.walltime_limit_seconds =
                std::stod(require_option_value(argc, argv, i, arg));
        } else if (arg == "--walltime-reserve-minutes") {
            settings.walltime_reserve_seconds =
                60.0 * std::stod(require_option_value(argc, argv, i, arg));
        } else if (arg == "--walltime-reserve-seconds") {
            settings.walltime_reserve_seconds =
                std::stod(require_option_value(argc, argv, i, arg));
        } else if (arg == "--particle-v0") {
            settings.particle_v0 =
                std::stod(require_option_value(argc, argv, i, arg));
        } else if (arg == "--duu0") {
            settings.duu0 = std::stod(require_option_value(argc, argv, i, arg));
        } else if (arg == "--split-ratio") {
            settings.split_ratio =
                std::stod(require_option_value(argc, argv, i, arg));
        } else if (arg == "--pmin-split-over-p0") {
            settings.pmin_split_over_p0 =
                std::stod(require_option_value(argc, argv, i, arg));
        } else if (arg == "--no-split") {
            settings.split_particles = false;
        } else if (arg == "--no-time-interp") {
            settings.time_interpolation = false;
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }
    if (settings.restart_enabled()) {
        const int restart_frame =
            infer_particle_snapshot_frame(settings.restart_particle_snapshot_path);
        if (restart_frame < 0) {
            throw std::runtime_error(
                "Cannot infer restart frame from particle snapshot name; expected particles_XXXXX.bin.");
        }
        if (start_frame_was_set && settings.start_frame != restart_frame) {
            throw std::runtime_error(
                "Restart particle snapshot frame does not match --start-frame.");
        }
        if (!start_frame_was_set) {
            settings.start_frame = restart_frame;
            if (requested_frame_count > 0) {
                settings.end_frame = settings.start_frame + requested_frame_count;
            }
        }
    }
    settings.validate();
    return settings;
}

/**
 * Execute the calibration driver.
 */
int run(const Reconnection2DSettings& settings) {
    using particle_storage_type = ReconnectionParticleStorage<>;
    using random_manager_type = RandomManager<>;

    require_fresh_output_directory(settings);
    std::filesystem::create_directories(settings.output_dir);
    const std::string summary_path = settings.output_dir + "/summary.csv";
    const std::string log_path = settings.output_dir + "/run.log";
    const bool append_outputs =
        settings.restart_enabled() && std::filesystem::exists(summary_path);
    const bool summary_has_content =
        append_outputs && std::filesystem::file_size(summary_path) > 0;
    std::ofstream summary_stream(
        summary_path,
        append_outputs ? (std::ios::out | std::ios::app) : std::ios::out);
    if (!summary_stream) {
        throw std::runtime_error("Failed to open summary output.");
    }
    if (!summary_has_content) {
        write_summary_header(summary_stream);
    }
    std::ofstream log_stream(
        log_path,
        settings.restart_enabled() ? (std::ios::out | std::ios::app) : std::ios::out);
    if (!log_stream) {
        throw std::runtime_error("Failed to open run log output.");
    }

    const std::uint64_t effective_seed =
        settings.restart_enabled()
            ? settings.seed +
                  0x9e3779b97f4a7c15ull *
                      static_cast<std::uint64_t>(settings.start_frame + 1)
            : settings.seed;
    random_manager_type random_manager(effective_seed);
    particle_storage_type particles(settings.particle_capacity);
    if (settings.restart_enabled()) {
        load_reconnection_particle_snapshot(particles, settings,
                                            settings.start_frame);
        std::ostringstream restart_log;
        restart_log << "Restarted from particle snapshot "
                    << settings.restart_particle_snapshot_path
                    << " at frame " << settings.start_frame
                    << " loaded_particles=" << particles.count
                    << " configured_capacity=" << particles.capacity
                    << " effective_seed=" << effective_seed
                    << " (RNG state is reseeded, not restored)";
        write_log_line(log_stream, restart_log.str());
    }

    auto read_start = wall_time_now();
    if (settings.transport_model == TransportModel::Parker) {
        auto lower_coefficients =
            load_parker_coefficient_frame<>(settings.field_dir,
                                            settings.start_frame);
        auto upper_coefficients =
            load_parker_coefficient_frame<>(settings.field_dir,
                                            settings.start_frame + 1);
        const double initial_read_seconds = elapsed_seconds_since(read_start);
        return run_transport_loop(settings, summary_stream, log_stream,
                                  random_manager, particles, lower_coefficients,
                                  upper_coefficients, initial_read_seconds);
    }

    auto lower_coefficients = load_coefficient_frame<>(settings.field_dir,
                                                       settings.start_frame);
    auto upper_coefficients = load_coefficient_frame<>(settings.field_dir,
                                                       settings.start_frame + 1);
    const double initial_read_seconds = elapsed_seconds_since(read_start);
    return run_transport_loop(settings, summary_stream, log_stream,
                              random_manager, particles, lower_coefficients,
                              upper_coefficients, initial_read_seconds);
}

} // namespace Reconnection2DCalibration

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    int result = 0;
    try {
        const auto settings =
            Reconnection2DCalibration::parse_settings(argc, argv);
        result = Reconnection2DCalibration::run(settings);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        result = 1;
    }
    Kokkos::finalize();
    return result;
}
