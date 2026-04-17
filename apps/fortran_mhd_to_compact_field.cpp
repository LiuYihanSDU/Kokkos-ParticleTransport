#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/**
 * Minimal subset of the Fortran MHD configuration binary.
 */
struct MhdConfig {
    double dx{1.0};
    double dy{1.0};
    double dz{1.0};
    double xmin{0.0};
    double ymin{0.0};
    double zmin{0.0};
    double xmax{1.0};
    double ymax{1.0};
    double zmax{1.0};
    double lx{1.0};
    double ly{1.0};
    double lz{1.0};
    double dt_out{0.1};
    std::int32_t nx{1};
    std::int32_t ny{1};
    std::int32_t nz{1};
    std::int32_t nxs{1};
    std::int32_t nys{1};
    std::int32_t nzs{1};
    std::int32_t topox{1};
    std::int32_t topoy{1};
    std::int32_t topoz{1};
    std::int32_t nvar{8};
    std::int32_t bcx{0};
    std::int32_t bcy{0};
    std::int32_t bcz{0};
};

/**
 * Command-line controls for the converter.
 */
struct ConverterSettings {
    std::filesystem::path input_dir;
    std::filesystem::path output_dir;
    std::filesystem::path config_name{"mhd_config.dat"};
    int start_frame{0};
    int end_frame{200};
};

/**
 * Read one POD value from a binary stream.
 */
template <typename T>
T read_scalar(std::ifstream& stream, const char* name) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!stream) {
        throw std::runtime_error(std::string("failed to read ") + name);
    }
    return value;
}

/**
 * Return a zero-padded Fortran MHD input path.
 */
std::filesystem::path mhd_frame_path(const std::filesystem::path& input_dir,
                                     const int frame) {
    std::ostringstream name;
    name << "mhd_data_" << std::setw(4) << std::setfill('0') << frame;
    return input_dir / name.str();
}

/**
 * Return a zero-padded compact Kokkos field output path.
 */
std::filesystem::path compact_frame_path(const std::filesystem::path& output_dir,
                                         const int frame) {
    std::ostringstream name;
    name << "field" << std::setw(5) << std::setfill('0') << frame << ".bin";
    return output_dir / name.str();
}

/**
 * Load the Fortran MHD configuration written by mhd_config_module::save_mhd_config.
 */
MhdConfig read_mhd_config(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open MHD config: " + path.string());
    }

    std::array<double, 13> real_values{};
    std::array<std::int32_t, 13> integer_values{};
    stream.read(reinterpret_cast<char*>(real_values.data()),
                static_cast<std::streamsize>(real_values.size() * sizeof(double)));
    stream.read(reinterpret_cast<char*>(integer_values.data()),
                static_cast<std::streamsize>(integer_values.size() * sizeof(std::int32_t)));
    if (!stream) {
        throw std::runtime_error("failed to read MHD config payload: " + path.string());
    }

    MhdConfig config;
    config.dx = real_values[0];
    config.dy = real_values[1];
    config.dz = real_values[2];
    config.xmin = real_values[3];
    config.ymin = real_values[4];
    config.zmin = real_values[5];
    config.xmax = real_values[6];
    config.ymax = real_values[7];
    config.zmax = real_values[8];
    config.lx = real_values[9];
    config.ly = real_values[10];
    config.lz = real_values[11];
    config.dt_out = real_values[12];
    config.nx = integer_values[0];
    config.ny = integer_values[1];
    config.nz = integer_values[2];
    config.nxs = integer_values[3];
    config.nys = integer_values[4];
    config.nzs = integer_values[5];
    config.topox = integer_values[6];
    config.topoy = integer_values[7];
    config.topoz = integer_values[8];
    config.nvar = integer_values[9];
    config.bcx = integer_values[10];
    config.bcy = integer_values[11];
    config.bcz = integer_values[12];
    if (config.nx <= 0 || config.ny <= 0 || config.nz != 1) {
        throw std::runtime_error("only positive 2D MHD configurations are supported");
    }
    return config;
}

/**
 * Return one raw Fortran storage index.
 */
std::size_t fortran_storage_index(const int component,
                                  const int x_storage,
                                  const int y_storage,
                                  const int z_storage,
                                  const int nfields,
                                  const int nxg,
                                  const int nyg) {
    return static_cast<std::size_t>(
        component + nfields * (x_storage + nxg * (y_storage + nyg * z_storage)));
}

/**
 * Write one double vector to a binary stream.
 */
void write_doubles(std::ofstream& stream, const std::vector<double>& values) {
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!stream) {
        throw std::runtime_error("failed while writing double payload");
    }
}

/**
 * Convert one Fortran MHD frame into the compact field format used by AthenaFieldReader.
 */
void convert_frame(const MhdConfig& config,
                   const std::filesystem::path& input_path,
                   const std::filesystem::path& output_path) {
    constexpr int nfields = 8;
    constexpr int vx_component = 0;
    constexpr int vy_component = 1;
    constexpr int vz_component = 2;
    constexpr int bx_component = 4;
    constexpr int by_component = 5;
    constexpr int bz_component = 6;

    const int nx = config.nx;
    const int ny = config.ny;
    const int nxg = nx + 4;
    const int nyg = ny + 4;
    const int nzg = 1;
    const int x_physical_start = 2;
    const int y_physical_start = 2;
    const int z_physical_start = 0;
    const std::size_t raw_count =
        static_cast<std::size_t>(nfields) * static_cast<std::size_t>(nxg) *
        static_cast<std::size_t>(nyg) * static_cast<std::size_t>(nzg);

    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open MHD frame: " + input_path.string());
    }
    std::vector<float> raw(raw_count);
    input.read(reinterpret_cast<char*>(raw.data()),
               static_cast<std::streamsize>(raw.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("failed to read MHD frame: " + input_path.string());
    }

    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open compact frame: " + output_path.string());
    }
    output.write(reinterpret_cast<const char*>(&config.nx), sizeof(std::int32_t));
    output.write(reinterpret_cast<const char*>(&config.ny), sizeof(std::int32_t));

    std::vector<double> x_edges(static_cast<std::size_t>(nx) + 1);
    std::vector<double> y_edges(static_cast<std::size_t>(ny) + 1);
    const double dx = (config.xmax - config.xmin) / static_cast<double>(nx);
    const double dy = (config.ymax - config.ymin) / static_cast<double>(ny);
    for (int i = 0; i <= nx; ++i) {
        x_edges[static_cast<std::size_t>(i)] =
            config.xmin + static_cast<double>(i) * dx;
    }
    for (int j = 0; j <= ny; ++j) {
        y_edges[static_cast<std::size_t>(j)] =
            config.ymin + static_cast<double>(j) * dy;
    }
    write_doubles(output, x_edges);
    write_doubles(output, y_edges);

    std::vector<double> component_values(static_cast<std::size_t>(nx) *
                                         static_cast<std::size_t>(ny));
    const std::array<int, 6> component_order{
        bx_component, by_component, bz_component,
        vx_component, vy_component, vz_component};
    for (const int component : component_order) {
        for (int j = 0; j < ny; ++j) {
            const int y_storage = y_physical_start + j;
            for (int i = 0; i < nx; ++i) {
                const int x_storage = x_physical_start + i;
                const std::size_t output_index =
                    static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
                    static_cast<std::size_t>(i);
                component_values[output_index] =
                    static_cast<double>(raw[fortran_storage_index(
                        component, x_storage, y_storage, z_physical_start,
                        nfields, nxg, nyg)]);
            }
        }
        write_doubles(output, component_values);
    }
}

/**
 * Parse command-line arguments.
 */
ConverterSettings parse_settings(const int argc, char** argv) {
    ConverterSettings settings;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto require_value = [&](const char* option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + option);
            }
            return argv[++i];
        };
        if (arg == "--input-dir") {
            settings.input_dir = require_value("--input-dir");
        } else if (arg == "--output-dir") {
            settings.output_dir = require_value("--output-dir");
        } else if (arg == "--config") {
            settings.config_name = require_value("--config");
        } else if (arg == "--start-frame") {
            settings.start_frame = std::stoi(require_value("--start-frame"));
        } else if (arg == "--end-frame") {
            settings.end_frame = std::stoi(require_value("--end-frame"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: fortran_mhd_to_compact_field --input-dir PATH "
                << "--output-dir PATH [--start-frame N] [--end-frame N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (settings.input_dir.empty() || settings.output_dir.empty()) {
        throw std::runtime_error("--input-dir and --output-dir are required");
    }
    if (settings.end_frame < settings.start_frame) {
        throw std::runtime_error("end frame must be greater than or equal to start frame");
    }
    return settings;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const ConverterSettings settings = parse_settings(argc, argv);
        std::filesystem::create_directories(settings.output_dir);
        const MhdConfig config = read_mhd_config(settings.input_dir / settings.config_name);
        std::cout << "Converting Fortran MHD frames " << settings.start_frame
                  << ".." << settings.end_frame << " to " << settings.output_dir
                  << '\n'
                  << "  nx=" << config.nx << " ny=" << config.ny
                  << " xmin=" << config.xmin << " xmax=" << config.xmax
                  << " ymin=" << config.ymin << " ymax=" << config.ymax << '\n';
        for (int frame = settings.start_frame; frame <= settings.end_frame; ++frame) {
            const auto input_path = mhd_frame_path(settings.input_dir, frame);
            const auto output_path = compact_frame_path(settings.output_dir, frame);
            convert_frame(config, input_path, output_path);
            std::cout << "frame " << frame << " -> " << output_path << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
